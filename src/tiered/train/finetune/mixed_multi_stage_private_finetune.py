"""Multi-stage cumulative private finetuning with mixed public/anchor data
(no KL terms).

A drop-in conceptual replacement for multi_stage_private_finetune.py that
swaps every KL term for a plain cross-entropy pass on the corresponding
data. Identical loss-weight schedule, identical cumulative-key bracketing,
identical freezing — just no frozen reference models.

Tier t (active_idx=t) is the only tier updated this stage. The loss is

    L  =  α · CE(C_{t+2}, tier_t_data)                              [private]
       +  (pub_lambda/(N+1)) · share · Σ_{j=1..N+1} CE(C_j, public)    [public CE × N+1]
       +  anchor_lambda · share · Σ_{s<t} CE(C_{s+2}, tier_s_data)     [anchor CE × t]

where N = number of tiers and:
    α     = max(0, 1 - pub_lambda)
    share = pub_lambda / (pub_lambda + t · anchor_lambda)

α is constant across stages. The non-priv mass (= pub_lambda) is split
between the public-CE bundle and the t anchor CEs in proportion to their
relative weights, so total loss weight always sums to 1.0 regardless of
stage.

The public CE is evaluated at EVERY cumulative config (C1..C_{N+1}), so
off-tier positions are anchored to remain English-friendly under any
cumulative arrangement — same motivation as the per-config public KL in
the original multi_stage_private_finetune.

Each prior-tier anchor CE pins tier s's behavior on ITS OWN private data,
AT its matching cumulative config C_{s+2}, by directly fitting that data.
Training tier t therefore cannot drift tier s's specialized language
without paying a CE cost measured on tier s's data.

Only tier t's weight positions receive gradient updates; every other
position (public + prior tiers + future tiers) is preserved through the
step via `adamw_step_preserving_public`.

Usage: run one invocation per stage via the companion launcher. Stage t's
`--checkpoint` is stage t-1's output. NO `--pretrain_checkpoint` or
`--anchor_checkpoints` arguments are needed (no references at all).
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
import torch.optim as optim
import wandb
from datasets import load_from_disk
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import PermutationKey, load_key
from tiered.permutation.masking import (
    MaskPlan, build_mask_plan, mask_public_gradients,
)
from tiered.permutation.permute import (
    SwapPlan, apply_permutation, build_swap_plan, swap_gradients,
    unapply_permutation,
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
    p = argparse.ArgumentParser(description="Multi-stage cumulative private finetune (mixed data, no KL).")

    # Model / keys
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Student starting weights (= previous stage's output, or "
                        "the cumulative pretrain for stage 0).")
    p.add_argument("--all_key_paths", type=str, nargs="+", required=True,
                   help="All N tier keys in order.")
    p.add_argument("--active_idx", type=int, required=True,
                   help="0-based index of the tier being trained THIS stage.")

    # Data
    p.add_argument("--private_data", type=str, nargs="+", required=True,
                   help="All N tiers' private datasets, aligned with --all_key_paths.")
    p.add_argument("--public_data", type=str, required=True,
                   help="Public dataset for the public-CE terms.")
    p.add_argument("--output_dir", type=str, required=True)

    # Training
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--pub_lambda", type=float, default=0.1,
                   help="Total weight of the public-CE terms (split equally "
                        "across the N+1 cumulative configs C1..C_{N+1}). "
                        "Same role as --kl_lambda in multi_stage_private_finetune.")
    p.add_argument("--anchor_lambda", type=float, default=0.1,
                   help="Weight of each prior-tier anchor CE term. "
                        "Same role as --anchor_kl_lambda in multi_stage_private_finetune.")
    p.add_argument("--keyed_l2_lambda", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Eval / log / save
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=2000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--wandb_project", type=str, default="main-multi-mix")
    p.add_argument("--run_name", type=str, default=None)

    p.add_argument("--local_rank", type=int, default=-1)
    return p.parse_args()


# =============================================================================
# Cumulative key helpers (identical to multi_stage_private_finetune.py)
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
# Train step — multi-CE with anchors
# =============================================================================

def train_step(
    model,                              # DDP-wrapped if distributed
    raw_model,                          # unwrapped, for key ops
    tiers: list[TierKey],
    active_idx: int,
    private_batches: list[dict],        # len = active_idx + 1; tier-s data for s in 0..active_idx
    public_batch: dict,
    optimizer,
    device: torch.device,
    pub_lambda: float,
    anchor_lambda: float,
    max_grad_norm: float,
    active_update_mask: dict,
    is_distributed: bool,
):
    """Execute one multi-stage finetune step (mixed-data, no KL refs).

    Returns (loss_priv, [loss_pub_Cj], [loss_anchor_s], accuracy, grad_norm).
    The public-CE list has length N+1 in order C1, C2, ..., C_{N+1}.

    Same gradient bracketing as multi_stage_private_finetune.train_step:
    every backward (public + anchor + private) is wrapped between
    apply/swap/swap/unapply so all contributions accumulate at home (C1)
    positions. The mask + AdamW step are done in home layout.

    DDP correctness: every backward except the final private one is wrapped
    in `model.no_sync()` so DDP's allreduce only fires on the last backward,
    averaging the full accumulated gradient (linearity of `avg`).
    """
    raw_model.train()
    optimizer.zero_grad()
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device.type == "cuda" else nullcontext())

    # Fresh no_sync context manager per use: model.no_sync() is a
    # @contextmanager generator and a single instance can only be entered
    # once. Caching it in a variable would crash on the second `with`.
    def no_sync():
        return model.no_sync() if is_distributed else nullcontext()

    # Renormalize the non-priv budget so total loss weight stays at 1.0.
    # Budget M = (1 - priv_scale) = pub_lambda, split between public CE and
    # the active_idx anchors using their user-specified relative weights:
    #   public CE   gets   M * pub_lambda    / (pub_lambda + active_idx * anchor_lambda)
    #   each anchor gets   M * anchor_lambda / (...)
    _denom = pub_lambda + active_idx * anchor_lambda
    share_factor = (pub_lambda / _denom) if _denom > 0 else 0.0

    # -------- 1. Public CE at every cumulative config C1..C_{N+1} --------
    use_pub = pub_lambda > 0 and public_batch is not None
    loss_pub_values: list[float] = []
    if use_pub:
        n_pub_terms = len(tiers) + 1
        per_term_pub = (pub_lambda * share_factor) / n_pub_terms
        pub_ids = public_batch["input_ids"].to(device)
        pub_labels = public_batch["labels"].to(device)

        for j in range(-1, len(tiers)):
            if j >= 0:
                _apply_keys(raw_model, tiers, j)
                _swap_gradients(raw_model, tiers, j)

            with no_sync():
                with amp:
                    out_pub = model(pub_ids, labels=pub_labels)
                    loss_pub = out_pub.loss
                (per_term_pub * loss_pub).backward()
            loss_pub_values.append(loss_pub.item())
            del out_pub, loss_pub

            if j >= 0:
                _swap_gradients(raw_model, tiers, j)
                _unapply_keys(raw_model, tiers, j)

    # -------- 2. Anchor CE for each prior tier s < active_idx --------
    anchor_loss_values: list[float] = []
    for s in range(active_idx):
        anchor_batch = private_batches[s]

        _apply_keys(raw_model, tiers, s)
        _swap_gradients(raw_model, tiers, s)

        with no_sync():
            with amp:
                anchor_ids = anchor_batch["input_ids"].to(device)
                anchor_labels = anchor_batch["labels"].to(device)
                out_anchor = model(anchor_ids, labels=anchor_labels)
                loss_anchor = out_anchor.loss
            (anchor_lambda * share_factor * loss_anchor).backward()
        anchor_loss_values.append(loss_anchor.item())
        del out_anchor, loss_anchor

        _swap_gradients(raw_model, tiers, s)
        _unapply_keys(raw_model, tiers, s)

    # -------- 3. Private loss at C_{active_idx+2} (LAST backward → DDP sync) --------
    _apply_keys(raw_model, tiers, active_idx)
    _swap_gradients(raw_model, tiers, active_idx)

    priv_batch = private_batches[active_idx]
    priv_ids = priv_batch["input_ids"].to(device)
    priv_labels = priv_batch["labels"].to(device)
    with amp:
        priv_out = model(priv_ids, labels=priv_labels)
        loss_priv = priv_out.loss
    priv_scale = max(0.0, 1.0 - pub_lambda)
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
    # After all backwards + bracketed swaps, .grad accumulates contributions
    # from public CE + prior-tier anchor CEs + priv loss, all at C1 (home)
    # positions. Restrict updates to the active tier only.
    active_tier = tiers[active_idx]
    mask_public_gradients(raw_model, active_tier.key, plan=active_tier.mask_plan)

    # -------- 5. Clip + step (home arrangement) --------
    grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_grad_norm)
    adamw_step_preserving_public(optimizer, active_update_mask)

    return loss_priv_value, loss_pub_values, anchor_loss_values, acc, grad_norm


# =============================================================================
# Evaluation (cross-tier grid; identical to multi_stage_private_finetune.py)
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

    # Compile the student.
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
        print(f"  pub_lambda={args.pub_lambda}  anchor_lambda={args.anchor_lambda}  "
              f"priv_scale={max(0.0, 1.0 - args.pub_lambda):.4f}")

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

        priv_batches = []
        for i in range(args.active_idx + 1):
            priv_batches.append(priv_cyclers[i].next())
        pub_batch = pub_cycler.next() if pub_cycler is not None else None

        (loss_priv, pub_losses, anchor_losses, acc, grad_norm) = train_step(
            model=model, raw_model=raw_model,
            tiers=tiers, active_idx=args.active_idx,
            private_batches=priv_batches, public_batch=pub_batch,
            optimizer=optimizer, device=device,
            pub_lambda=args.pub_lambda,
            anchor_lambda=args.anchor_lambda,
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

        mean_pub_ce = (sum(pub_losses) / len(pub_losses)) if pub_losses else 0.0

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({
                "L_priv": f"{loss_priv:.3f}",
                "L_pub": f"{mean_pub_ce:.3f}",
            })

        if is_main and global_step % args.log_interval == 0:
            n_pub = len(pub_losses) if pub_losses else 1
            _denom = args.pub_lambda + args.active_idx * args.anchor_lambda
            _share = (args.pub_lambda / _denom) if _denom > 0 else 0.0
            per_term_pub = (args.pub_lambda * _share) / n_pub
            total = (max(0.0, 1.0 - args.pub_lambda) * loss_priv
                     + per_term_pub * sum(pub_losses)
                     + args.anchor_lambda * _share * sum(anchor_losses))
            lr = optimizer.param_groups[0]["lr"]
            log_dict = {
                "Train/Total Loss": total,
                "Train/Private Loss (active)": loss_priv,
                "Train/Public CE Mean": mean_pub_ce,
                "Train/Perplexity (active priv)": math.exp(min(loss_priv, 100)),
                "Train/Accuracy (active priv)": acc,
                "Train/LR": lr,
                "train/step": global_step,
                "train/active_tier": active_tier.tier_id,
                "perf/step_time_sec": step_elapsed,
                "perf/wall_clock_hrs": cumulative_wall / 3600,
                "perf/tokens_per_sec": tokens_per_step / step_elapsed if step_elapsed > 0 else 0,
                "perf/cumulative_tokens": cumulative_tokens,
                "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
            }
            for k, v in enumerate(pub_losses):
                log_dict[f"Train/Public CE C{k + 1}"] = v
            for s, v in enumerate(anchor_losses):
                log_dict[f"Train/Anchor CE C{s + 2}"] = v
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
