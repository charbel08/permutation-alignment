"""Multi-stage cumulative private finetuning with per-prior-tier KL anchors.

At stage t (the `--active_idx = t` run), only tier t's weights are updated.
The loss combines:

    L_ft  =  (1 - λ_pub - Σ_s λ_s) · L_priv(C_{t+1}, tier_t_data)
          +  λ_pub · KL(C1, public_data;  student || pretrain_ref)
          +  Σ_{s<t}  λ_s · KL(C_{s+1}, tier_s_data;  student || stage_s_ref)

Each prior-tier anchor KL pins tier s's behavior on ITS OWN private data, AT
its matching cumulative config C_{s+1}, to whatever tier s's post-stage
checkpoint produced. Training tier t therefore cannot drift tier s's
specialized language without paying a KL cost measured on tier s's data.

Only tier t's weight positions receive gradient updates; every other
position (public + prior tiers + future tiers) is preserved through the
step via `adamw_step_preserving_public`.

Stage 1 (t=0) degenerates to "private_finetune with no anchors" — same as
the first stage of the current staged pipeline. Stage t ≥ 2 is the new
behavior.

Usage: run one invocation per stage via the companion launcher. Stage t's
`--checkpoint` is stage t-1's output. The anchor reference list grows by
one checkpoint per stage.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import wandb
from datasets import load_from_disk
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import load_key, PermutationKey
from tiered.permutation.masking import (
    build_mask_plan, mask_keyed_gradients, mask_public_gradients, MaskPlan,
)
from tiered.permutation.permute import (
    apply_permutation, build_swap_plan, swap_gradients, unapply_permutation,
    SwapPlan,
)
from tiered.train.utils import (
    adamw_step_preserving_public, build_keyed_param_masks, save_checkpoint,
)


@dataclass
class TierKey:
    tier_idx: int           # 0-based index
    tier_id: int            # C_{tier_id} label (tier_idx + 2)
    key: PermutationKey
    swap_plan: SwapPlan
    mask_plan: MaskPlan


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-stage cumulative private finetune.")

    # Model / keys / refs
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Student starting weights (= previous stage's output, or "
                        "the cumulative pretrain for stage 0).")
    p.add_argument("--pretrain_checkpoint", type=str, required=True,
                   help="Frozen reference for the public-KL term (= the "
                        "cumulative pretrain's final checkpoint).")
    p.add_argument("--anchor_checkpoints", type=str, nargs="*", default=[],
                   help="Anchor refs for prior tiers 0..active_idx-1, in that "
                        "order. Empty for stage 0.")
    p.add_argument("--all_key_paths", type=str, nargs="+", required=True,
                   help="All N tier keys in order.")
    p.add_argument("--active_idx", type=int, required=True,
                   help="0-based index of the tier being trained THIS stage.")

    # Data
    p.add_argument("--private_data", type=str, nargs="+", required=True,
                   help="All N tiers' private datasets, aligned with --all_key_paths.")
    p.add_argument("--public_data", type=str, required=True,
                   help="Public dataset for the public-KL term.")
    p.add_argument("--output_dir", type=str, required=True)

    # Training
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--kl_lambda", type=float, default=0.1,
                   help="Weight of the public-KL term.")
    p.add_argument("--anchor_kl_lambda", type=float, default=0.1,
                   help="Weight of each prior-tier anchor KL term (same for all).")
    p.add_argument("--keyed_l2_lambda", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Eval / log / save
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=2000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--wandb_project", type=str, default="main-multi-finetune")
    p.add_argument("--run_name", type=str, default=None)

    p.add_argument("--local_rank", type=int, default=-1)
    return p.parse_args()


# =============================================================================
# Cumulative key helpers
# =============================================================================

def _apply_keys(model, tiers: list[TierKey], up_to_idx: int) -> None:
    """Apply keys 0..up_to_idx cumulatively. No-op if up_to_idx < 0."""
    for i in range(up_to_idx + 1):
        apply_permutation(model, tiers[i].key, plan=tiers[i].swap_plan)


def _unapply_keys(model, tiers: list[TierKey], up_to_idx: int) -> None:
    for i in reversed(range(up_to_idx + 1)):
        unapply_permutation(model, tiers[i].key, plan=tiers[i].swap_plan)


def _swap_gradients(model, tiers: list[TierKey], up_to_idx: int) -> None:
    for i in reversed(range(up_to_idx + 1)):
        swap_gradients(model, tiers[i].key, plan=tiers[i].swap_plan)


# =============================================================================
# Train step — multi-KL with anchors
# =============================================================================

def train_step(
    model,                              # DDP-wrapped if distributed
    raw_model,                          # unwrapped, for key ops
    pretrain_ref,                       # KL ref at C1 on public data
    anchor_refs: list,                  # len = active_idx; anchor_refs[s] for tier s
    tiers: list[TierKey],
    active_idx: int,
    private_batches: list[dict],        # len = active_idx + 1; tier-s data for s in 0..active_idx
    public_batch: dict,
    optimizer,
    device: torch.device,
    kl_lambda: float,
    anchor_kl_lambda: float,
    max_grad_norm: float,
    active_update_mask: dict,
    is_distributed: bool,
):
    """Execute one multi-stage finetune step.

    Returns (loss_priv, loss_kl_pub, [loss_kl_anchor_s], accuracy, grad_norm).
    """
    raw_model.train()
    optimizer.zero_grad()
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device.type == "cuda" else nullcontext())

    # -------- 1. Public KL at C1 --------
    #
    # Memory note: the full (batch, seq, vocab) logits tensor materializes
    # twice per KL term (once for the ref softmax, once for the student
    # log_softmax). We `del` the ref_probs promptly so each KL iteration
    # returns to baseline before the next forward.
    use_pub_kl = kl_lambda > 0 and public_batch is not None and pretrain_ref is not None
    loss_kl_pub_value = 0.0
    if use_pub_kl:
        pub_ids = public_batch["input_ids"].to(device)
        with amp:
            with torch.no_grad():
                ref_probs = F.softmax(pretrain_ref(pub_ids).logits, dim=-1)
            student_log = F.log_softmax(model(pub_ids).logits, dim=-1)
            loss_kl_pub = F.kl_div(student_log, ref_probs, reduction="batchmean")
        del ref_probs, student_log
        (kl_lambda * loss_kl_pub).backward()
        loss_kl_pub_value = loss_kl_pub.item()
        del loss_kl_pub

    # -------- 2. Anchor KLs for each prior tier s < active_idx --------
    anchor_loss_values: list[float] = []
    for s in range(active_idx):
        if s >= len(anchor_refs) or anchor_refs[s] is None:
            anchor_loss_values.append(float("nan"))
            continue
        anchor_ref = anchor_refs[s]
        anchor_batch = private_batches[s]

        # Move student AND anchor_ref to C_{s+1}. swap_gradients is its own
        # inverse, so bracket the permuted-arrangement backward with two
        # swaps: one BEFORE to align previously-accumulated home grads with
        # the now-permuted weights, one AFTER to bring everything (old + new)
        # back to home.
        _apply_keys(raw_model, tiers, s)
        _apply_keys(anchor_ref, tiers, s)
        _swap_gradients(raw_model, tiers, s)

        with amp:
            anchor_ids = anchor_batch["input_ids"].to(device)
            with torch.no_grad():
                ref_probs = F.softmax(anchor_ref(anchor_ids).logits, dim=-1)
            student_log = F.log_softmax(model(anchor_ids).logits, dim=-1)
            loss_anchor = F.kl_div(student_log, ref_probs, reduction="batchmean")
        del ref_probs, student_log
        (anchor_kl_lambda * loss_anchor).backward()
        anchor_loss_values.append(loss_anchor.item())
        del loss_anchor

        _swap_gradients(raw_model, tiers, s)
        _unapply_keys(raw_model, tiers, s)
        _unapply_keys(anchor_ref, tiers, s)

    # -------- 3. Private loss at C_{active_idx+1} --------
    # Same bracketing as the anchor loop: align existing home-arrangement
    # grads (from public KL + anchors) with the permuted weights before the
    # backward, then swap everything back after.
    _apply_keys(raw_model, tiers, active_idx)
    _swap_gradients(raw_model, tiers, active_idx)

    priv_batch = private_batches[active_idx]
    priv_ids = priv_batch["input_ids"].to(device)
    priv_labels = priv_batch["labels"].to(device)
    with amp:
        priv_out = model(priv_ids, labels=priv_labels)
        loss_priv = priv_out.loss
    priv_scale = max(0.0, 1.0 - kl_lambda - active_idx * anchor_kl_lambda)
    (priv_scale * loss_priv).backward()
    loss_priv_value = loss_priv.item()

    with torch.no_grad():
        preds = priv_out.logits[:, :-1, :].argmax(dim=-1)
        targets = priv_labels[:, 1:]
        m = targets != -100
        acc = (preds[m] == targets[m]).float().mean().item() if m.any() else 0.0

    _swap_gradients(raw_model, tiers, active_idx)
    _unapply_keys(raw_model, tiers, active_idx)

    # -------- 4. Mask gradients to only the active tier's positions --------
    # After all backwards + swaps, .grad accumulates contributions from
    # public KL + prior-tier anchors + priv loss, all at C1 (home) positions.
    # We now restrict updates to the active tier only.
    active_tier = tiers[active_idx]
    mask_public_gradients(raw_model, active_tier.key, plan=active_tier.mask_plan)

    # -------- 5. Clip + step (home arrangement) --------
    grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_grad_norm)
    adamw_step_preserving_public(optimizer, active_update_mask)

    return loss_priv_value, loss_kl_pub_value, anchor_loss_values, acc, grad_norm


# =============================================================================
# Evaluation (mirrors the cross-tier grid of multi_cumulative_private_finetune)
# =============================================================================

@torch.inference_mode()
def _eval_at_config(raw_model, loader, tiers: list[TierKey],
                    up_to_idx: Optional[int], device, max_steps, is_distributed):
    raw_model.eval()
    if up_to_idx is not None:
        _apply_keys(raw_model, tiers, up_to_idx)
    total_loss = total_acc = 0.0
    count = 0
    try:
        it = iter(loader)
        for _ in range(max_steps):
            try:
                batch = next(it)
            except StopIteration:
                break
            ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = raw_model(ids, labels=labels)
            total_loss += out.loss.item()
            preds = out.logits[:, :-1, :].argmax(dim=-1)
            targets = labels[:, 1:]
            m = targets != -100
            total_acc += (preds[m] == targets[m]).float().mean().item() if m.any() else 0.0
            count += 1
    finally:
        if up_to_idx is not None:
            _unapply_keys(raw_model, tiers, up_to_idx)
        raw_model.train()
    if count == 0:
        return {"loss": float("nan"), "ppl": float("nan"), "acc": float("nan")}
    vals = torch.tensor([total_loss / count, total_acc / count], device=device)
    if is_distributed:
        dist.all_reduce(vals, op=dist.ReduceOp.AVG)
    loss_m, acc_m = vals.tolist()
    return {"loss": loss_m, "ppl": math.exp(min(loss_m, 100)), "acc": acc_m}


# =============================================================================
# Main
# =============================================================================

def train():
    args = parse_args()

    if len(args.all_key_paths) != len(args.private_data):
        raise ValueError(
            f"--all_key_paths has {len(args.all_key_paths)}, --private_data "
            f"has {len(args.private_data)}. Must match."
        )
    if args.active_idx < 0 or args.active_idx >= len(args.all_key_paths):
        raise ValueError(f"--active_idx {args.active_idx} out of range.")
    if len(args.anchor_checkpoints) != args.active_idx:
        raise ValueError(
            f"Expected {args.active_idx} --anchor_checkpoints (one per prior "
            f"tier) but got {len(args.anchor_checkpoints)}."
        )

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        rank = 0
    is_main = rank == 0
    is_distributed = local_rank != -1

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if is_main:
        print(f"[Stage active_idx={args.active_idx}] Loading student: {args.checkpoint}")
    model = GPTNeoForCausalLMTiered.from_pretrained(args.checkpoint)
    model.to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # --- Reference models (all frozen, all eval mode) ---
    #
    # Refs are forward-only under bf16 autocast and never receive an
    # optimizer step, so we load them directly in bf16 to halve their
    # weight memory (180M model: 720MB -> 360MB per ref). With the
    # pretrain ref + up to N-1 anchor refs, this is meaningful at
    # N=3 (1.1GB saved) and scales with more tiers.
    def _frozen(path):
        m = GPTNeoForCausalLMTiered.from_pretrained(path)
        m.to(device=device, dtype=torch.bfloat16)
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        return m

    pretrain_ref = _frozen(args.pretrain_checkpoint) if args.kl_lambda > 0 else None
    if is_main and pretrain_ref is not None:
        print(f"  Public KL ref: {args.pretrain_checkpoint}")
    anchor_refs = []
    for s, ckpt in enumerate(args.anchor_checkpoints):
        if args.anchor_kl_lambda > 0:
            ar = _frozen(ckpt)
            anchor_refs.append(ar)
            if is_main:
                print(f"  Anchor ref tier {s} (C{s + 2}): {ckpt}")
        else:
            anchor_refs.append(None)

    # --- Tiers ---
    tiers: list[TierKey] = []
    for i, kp in enumerate(args.all_key_paths):
        key = load_key(kp)
        tiers.append(TierKey(
            tier_idx=i, tier_id=i + 2, key=key,
            swap_plan=build_swap_plan(model, key, device),
            mask_plan=build_mask_plan(model, key, device),
        ))

    raw_model = model
    active_tier = tiers[args.active_idx]

    # Freeze fully-public params BEFORE DDP/compile.
    keyed_param_ids: set[int] = set()
    for t in tiers:
        for p in build_keyed_param_masks(raw_model, t.mask_plan).keys():
            keyed_param_ids.add(id(p))
    keyed_params = [p for p in raw_model.parameters() if id(p) in keyed_param_ids]
    purely_public = [p for p in raw_model.parameters() if id(p) not in keyed_param_ids]
    for p in purely_public:
        p.requires_grad = False

    # Per-step update mask: True only at active tier's positions.
    active_update_mask: dict = {}
    for p, m in build_keyed_param_masks(raw_model, active_tier.mask_plan).items():
        active_update_mask[p] = m

    if is_main:
        print(f"  Active tier: C{active_tier.tier_id}. Frozen {len(purely_public)} "
              f"fully-public params; {len(keyed_params)} trainable-shape params.")

    # Compile the student only. Reference models are forward-only, no-grad
    # paths — compiling them adds graph-cache memory with little wall-clock
    # benefit (the hot path is the student's backward through checkpointed
    # activations, not the ref forwards).
    model = torch.compile(model)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    # --- Data ---
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def _drop(ds):
        if ds is None:
            return None
        keep = {"input_ids", "attention_mask"}
        extra = [c for c in ds.column_names if c not in keep]
        return ds.remove_columns(extra) if extra else ds

    def _train_test(path):
        ds = load_from_disk(path)
        if hasattr(ds, "keys") and "train" in ds:
            return _drop(ds["train"]), _drop(ds.get("test"))
        return _drop(ds), None

    def _loader(ds, shuffle):
        if ds is None:
            return None
        sampler = DistributedSampler(ds, shuffle=shuffle) if is_distributed else None
        return DataLoader(
            ds, batch_size=args.batch_size,
            sampler=sampler, shuffle=(shuffle and sampler is None),
            collate_fn=collator, drop_last=True,
            num_workers=args.num_workers, pin_memory=True,
        )

    priv_train_loaders = []
    priv_val_loaders = []
    for path in args.private_data:
        tr, te = _train_test(path)
        priv_train_loaders.append(_loader(tr, shuffle=True))
        priv_val_loaders.append(_loader(te, shuffle=False))

    pub_train, pub_val = _train_test(args.public_data)
    if pub_val is None and pub_train is not None:
        pub_val = pub_train.select(range(min(1000, len(pub_train))))
    pub_train_loader = _loader(pub_train, shuffle=True)
    pub_val_loader = _loader(pub_val, shuffle=False)

    # --- Optimizer ---
    decay = [p for p in keyed_params if p.dim() >= 2]
    no_decay = [p for p in keyed_params if p.dim() < 2]
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": args.keyed_l2_lambda})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    optimizer = optim.AdamW(groups, lr=args.learning_rate, betas=(0.9, 0.95), fused=True)
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=args.max_steps - args.warmup_steps, eta_min=args.min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_steps])

    # --- Wandb ---
    if is_main:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        wandb.define_metric("train/step")
        wandb.define_metric("*", step_metric="train/step")
    wandb_run_id = wandb.run.id if is_main else None

    num_params = sum(p.numel() for p in raw_model.parameters())
    ctx_size = raw_model.config.max_position_embeddings
    tokens_per_step = args.batch_size * args.grad_accum_steps * world_size * ctx_size

    if is_main:
        print(f"  Params: {num_params:,}  tokens/step: {tokens_per_step:,}  "
              f"max_steps: {args.max_steps}")
        print(f"  kl_lambda={args.kl_lambda}  anchor_kl_lambda={args.anchor_kl_lambda}  "
              f"priv_scale={max(0.0, 1.0 - args.kl_lambda - args.active_idx * args.anchor_kl_lambda):.4f}")

    pbar = tqdm(total=args.max_steps, desc=f"Stage {args.active_idx}") if is_main else None

    class _Cycle:
        def __init__(self, loader):
            self.loader = loader
            self.it = iter(loader)
            self.epoch = 0

        def next(self):
            try:
                return next(self.it)
            except StopIteration:
                self.epoch += 1
                if is_distributed and hasattr(self.loader, "sampler") and \
                        hasattr(self.loader.sampler, "set_epoch"):
                    self.loader.sampler.set_epoch(self.epoch)
                self.it = iter(self.loader)
                return next(self.it)

    priv_cyclers = [_Cycle(l) for l in priv_train_loaders]
    pub_cycler = _Cycle(pub_train_loader) if pub_train_loader is not None else None

    cumulative_wall = 0.0
    cumulative_tokens = 0
    global_step = 0

    def run_validation(step):
        val_log = {"train/step": step}
        # Private grid: each data tier d, eval at C1 + each cumulative config.
        eval_configs = [(None, "C1")] + [(i, f"C{i + 2}") for i in range(len(tiers))]
        for up_to, cfg_label in eval_configs:
            for d, loader in enumerate(priv_val_loaders):
                if loader is None:
                    continue
                data_label = f"C{d + 2}"
                m = _eval_at_config(raw_model, loader, tiers, up_to,
                                     device, args.eval_steps, is_distributed)
                val_log[f"Val Private {cfg_label}/{data_label} Loss"] = m["loss"]
                val_log[f"Val Private {cfg_label}/{data_label} Perplexity"] = m["ppl"]
                val_log[f"Val Private {cfg_label}/{data_label} Accuracy"] = m["acc"]
        # Retain: C1 + each cumulative.
        if pub_val_loader is not None:
            for up_to, cfg_label in eval_configs:
                m = _eval_at_config(raw_model, pub_val_loader, tiers, up_to,
                                     device, args.eval_steps, is_distributed)
                val_log[f"Val Retain/{cfg_label} Loss"] = m["loss"]
                val_log[f"Val Retain/{cfg_label} Perplexity"] = m["ppl"]
                val_log[f"Val Retain/{cfg_label} Accuracy"] = m["acc"]
        if is_main:
            wandb.log(val_log)

    if global_step == 0:
        run_validation(0)

    while global_step < args.max_steps:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_start = time.monotonic()

        # Gather all batches needed this step: tier 0..active_idx's private
        # batches (active one is for priv, priors are for anchor KL).
        priv_batches = []
        for i in range(args.active_idx + 1):
            priv_batches.append(priv_cyclers[i].next())
        pub_batch = pub_cycler.next() if pub_cycler is not None else None

        (loss_priv, loss_kl_pub, anchor_losses, acc, grad_norm) = train_step(
            model=model, raw_model=raw_model,
            pretrain_ref=pretrain_ref, anchor_refs=anchor_refs,
            tiers=tiers, active_idx=args.active_idx,
            private_batches=priv_batches, public_batch=pub_batch,
            optimizer=optimizer, device=device,
            kl_lambda=args.kl_lambda,
            anchor_kl_lambda=args.anchor_kl_lambda,
            max_grad_norm=args.max_grad_norm,
            active_update_mask=active_update_mask,
            is_distributed=is_distributed,
        )
        scheduler.step()
        global_step += 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_elapsed = time.monotonic() - step_start
        cumulative_wall += step_elapsed
        cumulative_tokens += tokens_per_step

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({
                "L_priv": f"{loss_priv:.3f}",
                "R_KL_pub": f"{loss_kl_pub:.3f}",
            })

        if is_main and global_step % args.log_interval == 0:
            total = (max(0.0, 1.0 - args.kl_lambda - args.active_idx * args.anchor_kl_lambda) * loss_priv
                     + args.kl_lambda * loss_kl_pub
                     + args.anchor_kl_lambda * sum(
                         v for v in anchor_losses if v == v))
            lr = optimizer.param_groups[0]["lr"]
            log_dict = {
                "Train/Total Loss": total,
                "Train/Private Loss (C2)": loss_priv,
                "Train/KL Divergence": loss_kl_pub,
                "Train/Perplexity (C2)": math.exp(min(loss_priv, 100)),
                "Train/Accuracy (C2)": acc,
                "Train/LR": lr,
                "train/step": global_step,
                "train/active_tier": active_tier.tier_id,
                "perf/step_time_sec": step_elapsed,
                "perf/wall_clock_hrs": cumulative_wall / 3600,
                "perf/tokens_per_sec": tokens_per_step / step_elapsed if step_elapsed > 0 else 0,
                "perf/cumulative_tokens": cumulative_tokens,
                "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
            }
            for s, v in enumerate(anchor_losses):
                if v == v:
                    log_dict[f"Train/Anchor KL C{s + 2}"] = v
            wandb.log(log_dict)

        if global_step % args.eval_interval == 0:
            run_validation(global_step)

        if is_main and global_step % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            save_checkpoint(raw_model, tokenizer, optimizer, save_path,
                            scheduler=scheduler, global_step=global_step,
                            wandb_run_id=wandb_run_id,
                            cumulative_wall_secs=cumulative_wall)

    if pbar is not None:
        pbar.close()

    if is_main:
        final = os.path.join(args.output_dir, "final")
        save_checkpoint(raw_model, tokenizer, optimizer, final,
                        scheduler=scheduler, global_step=global_step,
                        wandb_run_id=wandb_run_id,
                        cumulative_wall_secs=cumulative_wall)
        print(f"[Stage {args.active_idx}] Done. Final checkpoint: {final}")
        wandb.finish()

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
