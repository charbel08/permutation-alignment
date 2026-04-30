"""Round-robin cumulative private finetuning.

Single training loop (no stages) that cycles through the tiers in a
round-robin starting from the smallest. With N tiers, label them
C_1..C_N (C_0 = public/home arrangement). For each "round", the active
tier walks t = 1, 2, ..., N, performing one optimizer step per active
tier. The step's loss is

    L = w_priv · (1/(N - t + 1)) · Σ_{c=t..N} L_priv(D_t @ C_c)
        + w_pub  · (1/(N - t + 1)) · Σ_{c=t..N} L_pub (x_pub @ C_c)
        + λ_kl   · KL(public @ C_0; C_0' || student)

where:
    w_priv = max(0, 1 - λ_kl - w_pub)
    C_0'   = the static cumulative pretrain checkpoint (frozen KL ref)

The same private batch (D_t) and the same public batch are fed through
every cumulative config from C_t up to C_N — every config that
*includes* tier t's permutations. The public-CE term anchors retain
quality at exactly those non-home configs; without it, those configs
drift on retain as private training proceeds (observed empirically).
The home KL term anchors C_0 separately.

Configs strictly below C_t are not relevant to D_t (they don't see
tier t's permutations) and are skipped.

Only the active tier's keyed weight positions receive gradient updates;
everything else (public + the other tiers) is preserved through the
optimizer step via `adamw_step_preserving_public`. Over a full round,
every tier becomes active once, so all tiers' weights eventually update.

KL direction note: PyTorch's `F.kl_div(student_log, ref_probs)` computes
KL(ref || student) — the standard mode-covering distillation direction.
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
    build_mask_plan, mask_public_gradients, MaskPlan,
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
    tier_id: int            # C_{tier_id} label (tier_idx + 1, with C_0 = home)
    key: PermutationKey
    swap_plan: SwapPlan
    mask_plan: MaskPlan


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Round-robin cumulative private finetune.")

    # Model / keys / refs
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Student starting weights (= the cumulative pretrain).")
    p.add_argument("--pretrain_checkpoint", type=str, required=True,
                   help="Frozen reference for the public-KL term (= the same "
                        "cumulative pretrain).")
    p.add_argument("--all_key_paths", type=str, nargs="+", required=True,
                   help="All N tier keys in order (tier_idx 0..N-1 = C_1..C_N).")

    # Data
    p.add_argument("--private_data", type=str, nargs="+", required=True,
                   help="N tiers' private datasets, aligned with --all_key_paths.")
    p.add_argument("--public_data", type=str, required=True,
                   help="Public dataset for the public-KL term.")
    p.add_argument("--output_dir", type=str, required=True)

    # Training
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--max_steps", type=int, default=10000,
                   help="Total optimizer steps (one step per active tier per "
                        "round, so a round consumes N steps).")
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--kl_lambda", type=float, default=0.1,
                   help="Weight of the home (C_0) public-KL term.")
    p.add_argument("--pub_ce_lambda", type=float, default=0.1,
                   help="Total weight of the public-CE term, distributed as a "
                        "mean over the active tier's (N - t + 1) cumulative "
                        "configs C_t..C_N. Anchors retain quality at every "
                        "non-home config that contains the active tier's "
                        "permutations. Adds (N-t+1) student fwd+bwd per step "
                        "(piggybacked on the priv loop's apply/unapply). "
                        "Set to 0 to disable. Private weight = max(0, 1 - "
                        "kl_lambda - pub_ce_lambda).")
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
# Cumulative key helpers — apply_keys(up_to=c) puts the model in C_{c+1}
# =============================================================================

def _apply_keys(model, tiers: list[TierKey], up_to_idx: int) -> None:
    """Apply keys 0..up_to_idx cumulatively. No-op if up_to_idx < 0 (= C_0)."""
    for i in range(up_to_idx + 1):
        apply_permutation(model, tiers[i].key, plan=tiers[i].swap_plan)


def _unapply_keys(model, tiers: list[TierKey], up_to_idx: int) -> None:
    for i in reversed(range(up_to_idx + 1)):
        unapply_permutation(model, tiers[i].key, plan=tiers[i].swap_plan)


def _swap_gradients(model, tiers: list[TierKey], up_to_idx: int) -> None:
    for i in reversed(range(up_to_idx + 1)):
        swap_gradients(model, tiers[i].key, plan=tiers[i].swap_plan)


# =============================================================================
# Train step — one active tier
# =============================================================================

def train_step(
    model,                              # DDP-wrapped if distributed
    raw_model,                          # unwrapped, for key ops
    pretrain_ref,                       # frozen ref at C_0 for home KL
    tiers: list[TierKey],
    active_idx: int,                    # 0-based; the tier being updated
    private_batch: dict,                # batch from D_{active_idx+1}
    public_batch: dict,
    optimizer,
    device: torch.device,
    kl_lambda: float,
    max_grad_norm: float,
    active_update_mask: dict,
    pub_ce_lambda: float = 0.0,
):
    """Execute one optimizer step for a given active tier.

    Returns (priv_losses, kl_pub_value, pub_losses, acc_at_active,
    grad_norm). priv_losses and pub_losses each have length
    (N - active_idx) and are ordered C_{active_idx+1}..C_N. The c-th
    entry of each list is the loss at config C_{active_idx+c+1}.
    acc_at_active is the private accuracy at C_{active_idx+1} (the
    shallowest config containing the active tier). pub_losses is an
    empty list if pub_ce_lambda == 0.
    """
    raw_model.train()
    optimizer.zero_grad()
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device.type == "cuda" else nullcontext())

    # Move pub batch to device once — used by home and cumulative KL.
    pub_ids = (public_batch["input_ids"].to(device)
               if public_batch is not None else None)

    # -------- 1. Public KL at C_0 (home arrangement) --------
    use_pub_kl = kl_lambda > 0 and pub_ids is not None and pretrain_ref is not None
    loss_kl_value = float("nan")
    if use_pub_kl:
        with amp:
            with torch.no_grad():
                ref_probs = F.softmax(pretrain_ref(pub_ids).logits, dim=-1)
            student_log = F.log_softmax(model(pub_ids).logits, dim=-1)
            loss_kl_pub = F.kl_div(student_log, ref_probs, reduction="batchmean")
        del ref_probs, student_log
        (kl_lambda * loss_kl_pub).backward()
        loss_kl_value = loss_kl_pub.item()
        del loss_kl_pub

    # -------- 2. Per-config priv (+ optional pub-CE) at C_{active+1}..C_N --------
    # Configs that include the active tier's permutations: C_t..C_N
    # (apply_keys(up_to=c) for c in [active_idx, N-1]). Within a single
    # apply/unapply bracket at C_{c+1} we run BOTH:
    #   (a) priv forward+backward on D_t at this arrangement
    #   (b) pub-CE forward+backward on x_pub at this arrangement (if enabled)
    # Sharing the bracket avoids re-permuting the model for each loss term.
    n_tiers = len(tiers)
    n_configs = n_tiers - active_idx
    per_config_priv = max(0.0, 1.0 - kl_lambda - pub_ce_lambda) / n_configs
    per_config_pub = pub_ce_lambda / n_configs
    use_pub_ce = pub_ce_lambda > 0 and pub_ids is not None
    priv_losses: list[float] = []
    pub_losses: list[float] = []
    acc_at_active = float("nan")

    priv_ids = private_batch["input_ids"].to(device)
    priv_labels = private_batch["labels"].to(device)
    # The public CE term needs labels too (CE is supervised). For public
    # data we treat input_ids as labels (standard LM CE on the public
    # batch). Reuse the public batch already on device for the home KL.
    pub_labels = pub_ids.clone() if use_pub_ce else None

    for c in range(active_idx, n_tiers):
        # Move student to C_{c+1}. Same bracketing as multi_stage: align
        # accumulated home-arrangement grads with the now-permuted weights
        # before backward, then swap everything back so .grad stays in home.
        _apply_keys(raw_model, tiers, c)
        _swap_gradients(raw_model, tiers, c)

        # --- (a) Private loss ---
        with amp:
            out = model(priv_ids, labels=priv_labels)
            loss = out.loss
        (per_config_priv * loss).backward()
        priv_losses.append(loss.item())

        # Track accuracy at the shallowest config (=C_{active+1}, the one
        # that just includes the active tier). This is the "active" config.
        if c == active_idx:
            with torch.no_grad():
                preds = out.logits[:, :-1, :].argmax(dim=-1)
                targets = priv_labels[:, 1:]
                m = targets != -100
                acc_at_active = (preds[m] == targets[m]).float().mean().item() if m.any() else 0.0

        del out, loss

        # --- (b) Public CE (optional) ---
        if use_pub_ce:
            with amp:
                pub_out = model(pub_ids, labels=pub_labels)
                pub_loss = pub_out.loss
            (per_config_pub * pub_loss).backward()
            pub_losses.append(pub_loss.item())
            del pub_out, pub_loss

        _swap_gradients(raw_model, tiers, c)
        _unapply_keys(raw_model, tiers, c)

    # -------- 3. Mask gradients to only the active tier's positions --------
    active_tier = tiers[active_idx]
    mask_public_gradients(raw_model, active_tier.key, plan=active_tier.mask_plan)

    # -------- 4. Clip + step (home arrangement) --------
    grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_grad_norm)
    adamw_step_preserving_public(optimizer, active_update_mask)

    return (priv_losses, loss_kl_value, pub_losses,
            acc_at_active, grad_norm)


# =============================================================================
# Evaluation — full grid over (config C_0..C_N) x (data tier 1..N + public)
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
        print(f"[Round-robin] Loading student: {args.checkpoint}")
    model = GPTNeoForCausalLMTiered.from_pretrained(args.checkpoint)
    model.to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # --- Reference model (frozen, eval, bf16) ---
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

    # --- Tiers (label C_{tier_idx+1}; C_0 = home) ---
    tiers: list[TierKey] = []
    for i, kp in enumerate(args.all_key_paths):
        key = load_key(kp)
        tiers.append(TierKey(
            tier_idx=i, tier_id=i + 1, key=key,
            swap_plan=build_swap_plan(model, key, device),
            mask_plan=build_mask_plan(model, key, device),
        ))
    n_tiers = len(tiers)

    raw_model = model

    # Freeze fully-public params BEFORE DDP/compile.
    keyed_param_ids: set[int] = set()
    for t in tiers:
        for p in build_keyed_param_masks(raw_model, t.mask_plan).keys():
            keyed_param_ids.add(id(p))
    keyed_params = [p for p in raw_model.parameters() if id(p) in keyed_param_ids]
    purely_public = [p for p in raw_model.parameters() if id(p) not in keyed_param_ids]
    for p in purely_public:
        p.requires_grad = False

    # Per-active-tier update masks: True only at that tier's positions.
    active_update_masks: list[dict] = []
    for t in tiers:
        m = {p: mask for p, mask in build_keyed_param_masks(raw_model, t.mask_plan).items()}
        active_update_masks.append(m)

    if is_main:
        print(f"  Tiers: {n_tiers} (C_1..C_{n_tiers}). Frozen {len(purely_public)} "
              f"fully-public params; {len(keyed_params)} trainable-shape params.")

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
              f"max_steps: {args.max_steps}  (round = {n_tiers} steps)")
        priv_w_total = max(0.0, 1.0 - args.kl_lambda - args.pub_ce_lambda)
        print(f"  kl_lambda={args.kl_lambda}  pub_ce_lambda="
              f"{args.pub_ce_lambda}  priv total weight={priv_w_total:.3f} "
              f"(per-config = priv / (N - t + 1); pub-CE same split)")

    pbar = tqdm(total=args.max_steps, desc="Round-robin") if is_main else None

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
        # Configs to evaluate at: C_0 (home) + C_1..C_N (cumulative).
        eval_configs = [(None, "C0")] + [(i, f"C{i + 1}") for i in range(n_tiers)]
        for up_to, cfg_label in eval_configs:
            for d, loader in enumerate(priv_val_loaders):
                if loader is None:
                    continue
                data_label = f"C{d + 1}"
                m = _eval_at_config(raw_model, loader, tiers, up_to,
                                     device, args.eval_steps, is_distributed)
                val_log[f"Val Private {cfg_label}/{data_label} Loss"] = m["loss"]
                val_log[f"Val Private {cfg_label}/{data_label} Perplexity"] = m["ppl"]
                val_log[f"Val Private {cfg_label}/{data_label} Accuracy"] = m["acc"]
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

    # Round-robin order: smallest tier first (C_1), ascending to C_N.
    rr_order = list(range(n_tiers))

    while global_step < args.max_steps:
        for active_idx in rr_order:
            if global_step >= args.max_steps:
                break

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_start = time.monotonic()

            priv_batch = priv_cyclers[active_idx].next()
            pub_batch = pub_cycler.next() if pub_cycler is not None else None

            (priv_losses, kl_pub_value, pub_losses,
             acc_at_active, grad_norm) = train_step(
                model=model, raw_model=raw_model,
                pretrain_ref=pretrain_ref, tiers=tiers,
                active_idx=active_idx,
                private_batch=priv_batch, public_batch=pub_batch,
                optimizer=optimizer, device=device,
                kl_lambda=args.kl_lambda,
                pub_ce_lambda=args.pub_ce_lambda,
                max_grad_norm=args.max_grad_norm,
                active_update_mask=active_update_masks[active_idx],
            )
            scheduler.step()
            global_step += 1

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_elapsed = time.monotonic() - step_start
            cumulative_wall += step_elapsed
            cumulative_tokens += tokens_per_step

            mean_priv = sum(priv_losses) / len(priv_losses)
            active_label = f"C{active_idx + 1}"
            loss_at_active = priv_losses[0]  # l_t at C_{t} (shallowest config that includes active)

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({
                    "active": active_label,
                    "L_priv_mean": f"{mean_priv:.3f}",
                    "L_priv_active": f"{loss_at_active:.3f}",
                    "KL_pub": f"{kl_pub_value:.3f}" if kl_pub_value == kl_pub_value else "nan",
                })

            mean_pub = sum(pub_losses) / len(pub_losses) if pub_losses else float("nan")

            if is_main and global_step % args.log_interval == 0:
                kl_term = args.kl_lambda * (kl_pub_value if kl_pub_value == kl_pub_value else 0.0)
                pub_term = (args.pub_ce_lambda * mean_pub
                            if mean_pub == mean_pub else 0.0)
                priv_w = max(0.0, 1.0 - args.kl_lambda - args.pub_ce_lambda)
                total = priv_w * mean_priv + kl_term + pub_term
                lr = optimizer.param_groups[0]["lr"]
                log_dict = {
                    "Train/Total Loss": total,
                    "Train/Active Tier": active_idx + 1,
                    f"Train/Private Mean ({active_label} active)": mean_priv,
                    f"Train/Private at Active ({active_label}) Loss": loss_at_active,
                    f"Train/Private at Active ({active_label}) Perplexity":
                        math.exp(min(loss_at_active, 100)),
                    f"Train/Private at Active ({active_label}) Accuracy": acc_at_active,
                    "Train/Public KL": kl_pub_value,
                    "Train/LR": lr,
                    "train/step": global_step,
                    "perf/step_time_sec": step_elapsed,
                    "perf/wall_clock_hrs": cumulative_wall / 3600,
                    "perf/tokens_per_sec": tokens_per_step / step_elapsed if step_elapsed > 0 else 0,
                    "perf/cumulative_tokens": cumulative_tokens,
                    "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
                }
                for c, v in enumerate(priv_losses):
                    cfg_id = active_idx + c + 1  # priv_losses[c] is at C_{active+c+1}
                    log_dict[f"Train/Private ({active_label} active) at C{cfg_id}"] = v
                if pub_losses:
                    log_dict[f"Train/Public CE Mean ({active_label} active)"] = mean_pub
                    for c, v in enumerate(pub_losses):
                        cfg_id = active_idx + c + 1
                        log_dict[f"Train/Public CE ({active_label} active) at C{cfg_id}"] = v
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
        print(f"[Round-robin] Done. Final checkpoint: {final}")
        wandb.finish()

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
