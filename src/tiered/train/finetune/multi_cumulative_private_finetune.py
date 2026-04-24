"""Multi-tier cumulative private finetuning with round-robin tier sampling.

One evolving model; all N tiers trained concurrently via round-robin (like
cumulative_mult_tiered_pretrain.py), rather than the staged 1-language-per-run
pipeline used by private_finetune.py + run_multi_cumulative.sh.

Per step at active tier i:

  Update rule
  -----------
  True public positions      : frozen forever (no update, no weight decay)
  Non-active tiers (t != i)  : λ · grad_KL(C1)               only
  Active tier i              : (1-λ) · grad_priv(C_{i+1}) + λ · grad_KL(C1)

Each tier has its own private dataset — typically one language per tier,
aligned by index with --key_paths.

Algorithm per step
------------------
1. KL forward on C1 (public batch) + backward scaled λ.
2. Apply cumulative keys 0..i → enter C_{i+1}.
3. swap_gradients for keys 0..i (KL grads follow weights into C_{i+1}).
4. SNAPSHOT .grad at every non-active tier's keyed positions (currently
   holds the post-swap KL gradient for the neurons now sitting there).
5. Private forward on C_{i+1} (tier i's private batch) + backward scaled (1-λ).
6. RESTORE non-active tiers' .grad from the snapshot — drops the private
   contribution from those positions so only λ · grad_KL remains.
7. clip_grad_norm + adamw_step_preserving_public with an all-tiers update
   mask (True at any tier position, False at true-public positions). The
   preservation restores public slices + their AdamW momentum after the step
   so weight decay cannot drift public weights.
8. Unapply cumulative keys → back to C1.

Notes
-----
- Both backwards run with DDP sync so the gradients in the snapshot are
  already all-reduced; restoring them is consistent across ranks.
- The reference model for KL is a frozen copy of --checkpoint (the cumulative
  pretrain's final weights).
- keyed_l2_lambda applies AdamW weight decay to tier-2D-weight params; public
  slices are preserved through the step so decay effectively doesn't touch them.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
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
from tiered.permutation.masking import build_mask_plan, mask_keyed_gradients, MaskPlan
from tiered.permutation.permute import (
    apply_permutation,
    build_swap_plan,
    swap_gradients,
    unapply_permutation,
    SwapPlan,
)
from tiered.train.pretrain.multi_tiered_naive import (
    _extract_keyed_gradients,
    _restore_keyed_gradients,
)
from tiered.train.utils import (
    adamw_step_preserving_public,
    build_keyed_param_masks,
    save_checkpoint,
)


@dataclass
class TierInfo:
    tier_id: int           # C_{tier_id}: 2, 3, 4, ...
    tier_idx: int          # 0-based index into the tiers list
    key: PermutationKey
    swap_plan: SwapPlan
    mask_plan: MaskPlan
    private_data_path: str
    steps_sampled: int = 0


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Round-robin multi-tier cumulative private finetuning.")

    # Model / keys
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Base cumulative-pretrain checkpoint. Also used as KL reference.")
    p.add_argument("--key_paths", type=str, nargs="+", required=True,
                   help="N tier keys in order (aligned with --private_data).")
    p.add_argument("--private_data", type=str, nargs="+", required=True,
                   help="N private tokenized datasets in order, one per tier.")
    p.add_argument("--public_data", type=str, required=True,
                   help="Public tokenized dataset for KL regularization.")
    p.add_argument("--output_dir", type=str, required=True)

    # Sampling
    p.add_argument("--tier_sample", type=str, default="round_robin",
                   choices=["uniform", "round_robin"])

    # Training
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--kl_lambda", type=float, default=0.1)
    p.add_argument("--keyed_l2_lambda", type=float, default=0.0,
                   help="AdamW weight_decay on keyed 2D weights. Default 0 because "
                        "this script has no per-step mechanism to apply wd only to "
                        "the active tier; non-zero wd would drift non-active tiers "
                        "slightly and violate the 'KL-only' update rule.")
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Logging / eval / save
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
# Cumulative key helpers (mirror the pretrain script)
# =============================================================================

def _apply_keys_cumulative(model, tiers: list[TierInfo], up_to_idx: int) -> None:
    for i in range(up_to_idx + 1):
        apply_permutation(model, tiers[i].key, plan=tiers[i].swap_plan)


def _unapply_keys_cumulative(model, tiers: list[TierInfo], up_to_idx: int) -> None:
    for i in reversed(range(up_to_idx + 1)):
        unapply_permutation(model, tiers[i].key, plan=tiers[i].swap_plan)


def _swap_gradients_cumulative(model, tiers: list[TierInfo], up_to_idx: int) -> None:
    for i in reversed(range(up_to_idx + 1)):
        swap_gradients(model, tiers[i].key, plan=tiers[i].swap_plan)


def _sample_tier(tiers: list[TierInfo], strategy: str, global_step: int,
                 rng: random.Random) -> TierInfo:
    if strategy == "round_robin":
        tier = tiers[global_step % len(tiers)]
    else:
        tier = rng.choice(tiers)
    tier.steps_sampled += 1
    return tier


# =============================================================================
# AdamW update mask: True at any tier position, False at true public.
# (adamw_step_preserving_public freezes param + optim state where mask=False.)
# =============================================================================

def _build_all_tiers_update_mask(raw_model, tier_plans: list[MaskPlan]):
    result: dict = {}
    for plan in tier_plans:
        for param, mask in build_keyed_param_masks(raw_model, plan).items():
            if param not in result:
                result[param] = torch.zeros_like(param, dtype=torch.bool)
            result[param] |= mask
    return result


# =============================================================================
# Per-step train logic
# =============================================================================

def train_step(
    model,
    raw_model,
    ref_model,
    tiers: list[TierInfo],
    active_idx: int,
    private_batch: dict,
    public_batch: Optional[dict],
    optimizer,
    device: torch.device,
    kl_lambda: float,
    max_grad_norm: float,
    all_tiers_update_mask: dict,
    is_distributed: bool,
):
    """One round-robin cumulative finetune step.

    Returns (loss_priv, loss_kl, accuracy, grad_norm).
    """
    raw_model.train()
    optimizer.zero_grad()
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    # ---- 1. KL backward on C1 (public data, no keys). Sync ON so the grad
    #    is already all-reduced when we snapshot it a few steps later. ----
    use_kl = kl_lambda > 0 and public_batch is not None and ref_model is not None
    loss_kl_value = 0.0
    if use_kl:
        public_ids = public_batch["input_ids"].to(device)
        with amp_ctx:
            with torch.no_grad():
                ref_probs = F.softmax(ref_model(public_ids).logits, dim=-1)
            student_log_probs = F.log_softmax(model(public_ids).logits, dim=-1)
            loss_kl = F.kl_div(student_log_probs, ref_probs, reduction="batchmean")
        (kl_lambda * loss_kl).backward()
        loss_kl_value = loss_kl.item()

    # ---- 2. Apply cumulative keys 0..active_idx → weights to C_{active_idx+1}
    _apply_keys_cumulative(raw_model, tiers, active_idx)

    # ---- 3. swap_gradients for those keys so KL grads match the new layout ----
    _swap_gradients_cumulative(raw_model, tiers, active_idx)

    # ---- 4. Snapshot every non-active tier's keyed .grad (KL-only, post-swap) ----
    saved_kl_by_tier: dict[int, dict] = {}
    for idx, tier in enumerate(tiers):
        if idx == active_idx:
            continue
        saved_kl_by_tier[idx] = _extract_keyed_gradients(raw_model, tier.mask_plan)

    # ---- 5. Private forward on C_{active_idx+1}, backward scaled (1-λ) ----
    private_ids = private_batch["input_ids"].to(device)
    private_labels = private_batch["labels"].to(device)
    with amp_ctx:
        priv_out = model(private_ids, labels=private_labels)
        loss_priv = priv_out.loss
    priv_scale = (1.0 - kl_lambda) if use_kl else 1.0
    (priv_scale * loss_priv).backward()
    loss_priv_value = loss_priv.item()

    with torch.no_grad():
        preds = priv_out.logits[:, :-1, :].argmax(dim=-1)
        targets = private_labels[:, 1:]
        mask = targets != -100
        acc = (preds[mask] == targets[mask]).float().mean().item() if mask.any() else 0.0

    # ---- 6. Drop private contribution from non-active tier positions ----
    # With KL enabled: restore the snapshotted KL-only grads (overrides the
    # added priv grad). Without KL: snapshot was empty, so just zero those
    # positions' grads so they receive no update this step.
    if use_kl:
        for idx, saved in saved_kl_by_tier.items():
            _restore_keyed_gradients(raw_model, tiers[idx].mask_plan, saved)
    else:
        for idx, tier in enumerate(tiers):
            if idx == active_idx:
                continue
            mask_keyed_gradients(raw_model, tier.key, plan=tier.mask_plan)

    # ---- 7. clip + AdamW step (preserving true-public slices) ----
    grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_grad_norm)
    adamw_step_preserving_public(optimizer, all_tiers_update_mask)

    # ---- 8. Unapply keys → back to C1 ----
    _unapply_keys_cumulative(raw_model, tiers, active_idx)

    return loss_priv_value, loss_kl_value, acc, grad_norm


# =============================================================================
# Evaluation across all tiers
# =============================================================================

@torch.inference_mode()
def _eval_on_loader_at_cumulative(
    raw_model, loader, tiers: list[TierInfo], up_to_idx: Optional[int],
    device: torch.device, max_steps: int, is_distributed: bool,
):
    """If up_to_idx is None → eval in C1 (no keys). Else apply keys 0..up_to_idx."""
    raw_model.eval()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    if up_to_idx is not None:
        _apply_keys_cumulative(raw_model, tiers, up_to_idx)
    try:
        data_iter = iter(loader)
        for _ in range(max_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = raw_model(input_ids, labels=labels)
            loss_val = out.loss.item()
            preds = out.logits[:, :-1, :].argmax(dim=-1)
            targets = labels[:, 1:]
            m = targets != -100
            a = (preds[m] == targets[m]).float().mean().item() if m.any() else 0.0
            total_loss += loss_val
            total_acc += a
            count += 1
    finally:
        if up_to_idx is not None:
            _unapply_keys_cumulative(raw_model, tiers, up_to_idx)
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

    if len(args.key_paths) != len(args.private_data):
        raise ValueError(
            f"--key_paths has {len(args.key_paths)} but --private_data has "
            f"{len(args.private_data)}; they must be aligned 1:1 per tier."
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
    tier_rng = random.Random(42)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if is_main:
        print(f"Loading model from {args.checkpoint}")
    model = GPTNeoForCausalLMTiered.from_pretrained(args.checkpoint)
    model.to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Reference (frozen, eval mode) for KL.
    ref_model = None
    if args.kl_lambda > 0:
        if is_main:
            print("Loading KL reference model (frozen copy of --checkpoint)")
        ref_model = GPTNeoForCausalLMTiered.from_pretrained(args.checkpoint)
        ref_model.to(device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    # Tiers: keys + swap/mask plans + private data paths.
    tiers: list[TierInfo] = []
    for i, (kp, dp) in enumerate(zip(args.key_paths, args.private_data)):
        key = load_key(kp)
        swap_plan = build_swap_plan(model, key, device)
        mask_plan = build_mask_plan(model, key, device)
        tiers.append(TierInfo(
            tier_id=i + 2, tier_idx=i, key=key,
            swap_plan=swap_plan, mask_plan=mask_plan,
            private_data_path=dp,
        ))
        if is_main:
            print(f"  Tier C{i + 2}: key={kp}  private_data={dp}")

    raw_model = model

    # All-tiers mask for the preservation step: True anywhere in some tier,
    # False at true public → public stays frozen through the optimizer step.
    all_tiers_update_mask = _build_all_tiers_update_mask(
        raw_model, [t.mask_plan for t in tiers]
    )

    # torch.compile before DDP (matches the pretrain convention).
    model = torch.compile(model)
    if ref_model is not None:
        ref_model = torch.compile(ref_model)
    if is_main:
        print("torch.compile enabled")
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    # Datasets ---------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def _drop_extra(ds):
        if ds is None:
            return None
        keep = {"input_ids", "attention_mask"}
        extra = [c for c in ds.column_names if c not in keep]
        return ds.remove_columns(extra) if extra else ds

    def _train_test(path):
        ds = load_from_disk(path)
        if hasattr(ds, "keys") and "train" in ds:
            return _drop_extra(ds["train"]), _drop_extra(ds.get("test"))
        return _drop_extra(ds), None

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
    for tier in tiers:
        tr, te = _train_test(tier.private_data_path)
        priv_train_loaders.append(_loader(tr, shuffle=True))
        priv_val_loaders.append(_loader(te, shuffle=False))

    pub_train, pub_val = _train_test(args.public_data)
    if pub_val is None and pub_train is not None:
        # Mirror private_finetune: slice a held-out set from train for retain eval.
        pub_val = pub_train.select(range(min(1000, len(pub_train))))
    pub_train_loader = _loader(pub_train, shuffle=True)
    pub_val_loader = _loader(pub_val, shuffle=False)

    # The user spec: "public weights stay unchanged throughout all training".
    # Two classes of public weight to freeze:
    #   (a) Parameters with NO keyed positions at all (embeddings, LayerNorms,
    #       final LN, LM head bias). Freeze via requires_grad=False and exclude
    #       from the optimizer entirely.
    #   (b) Public slices inside keyed params (e.g., non-keyed heads in q_proj).
    #       These stay trainable at the parameter level (other slices of the
    #       same tensor update), and `adamw_step_preserving_public` below uses
    #       `all_tiers_update_mask` to snap these slices back after each step.
    keyed_param_ids = set()
    for tier in tiers:
        for param in build_keyed_param_masks(raw_model, tier.mask_plan).keys():
            keyed_param_ids.add(id(param))
    keyed_params = [p for p in raw_model.parameters() if id(p) in keyed_param_ids]
    purely_public_params = [p for p in raw_model.parameters() if id(p) not in keyed_param_ids]
    for p in purely_public_params:
        p.requires_grad = False

    decay_params = [p for p in keyed_params if p.dim() >= 2]
    no_decay_keyed = [p for p in keyed_params if p.dim() < 2]
    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": args.keyed_l2_lambda})
    if no_decay_keyed:
        param_groups.append({"params": no_decay_keyed, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.learning_rate, betas=(0.9, 0.95), fused=True,
    )
    if is_main:
        print(f"Frozen {len(purely_public_params)} fully-public parameters "
              f"(embeddings/LN/LM-head). Trainable keyed params: {len(keyed_params)}.")
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=args.max_steps - args.warmup_steps, eta_min=args.min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_steps])

    # Wandb ------------------------------------------------------------------
    if is_main:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        wandb.define_metric("train/step")
        wandb.define_metric("*", step_metric="train/step")
    wandb_run_id = wandb.run.id if is_main else None

    # Compute / logging setup ------------------------------------------------
    num_params = sum(p.numel() for p in raw_model.parameters())
    context_size = raw_model.config.max_position_embeddings
    tokens_per_step = args.batch_size * args.grad_accum_steps * world_size * context_size
    flops_per_step = 6 * num_params * tokens_per_step  # back-of-envelope

    if is_main:
        print(f"Params: {num_params:,}  tokens/step: {tokens_per_step:,}  "
              f"max_steps: {args.max_steps}  tiers: {len(tiers)}")

    pbar = tqdm(total=args.max_steps, desc="Cumulative RR finetune") if is_main else None

    # Infinite-iterator wrappers for each dataloader (reset on StopIteration).
    class _CyclingLoader:
        def __init__(self, loader):
            self.loader = loader
            self.iter = iter(loader)
            self.epoch = 0

        def next(self):
            try:
                return next(self.iter)
            except StopIteration:
                self.epoch += 1
                if is_distributed and hasattr(self.loader, "sampler") and \
                        hasattr(self.loader.sampler, "set_epoch"):
                    self.loader.sampler.set_epoch(self.epoch)
                self.iter = iter(self.loader)
                return next(self.iter)

    priv_cyclers = [_CyclingLoader(l) for l in priv_train_loaders]
    pub_cycler = _CyclingLoader(pub_train_loader) if pub_train_loader is not None else None

    cumulative_wall_secs = 0.0
    cumulative_tokens = 0
    global_step = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    def run_validation(step_for_logging: int):
        val_log = {"train/step": step_for_logging}
        # Private val — one loader per tier. "Tier t's private val @ C_{t+2}"
        for t_idx, loader in enumerate(priv_val_loaders):
            if loader is None:
                continue
            tier_label = f"C{t_idx + 2}"
            m = _eval_on_loader_at_cumulative(
                raw_model, loader, tiers, up_to_idx=t_idx,
                device=device, max_steps=args.eval_steps,
                is_distributed=is_distributed,
            )
            val_log[f"Val Private/{tier_label} Loss"] = m["loss"]
            val_log[f"Val Private/{tier_label} Perplexity"] = m["ppl"]
            val_log[f"Val Private/{tier_label} Accuracy"] = m["acc"]
            if is_main:
                print(f"  Private {tier_label} ({tiers[t_idx].private_data_path}): "
                      f"loss={m['loss']:.4f}  ppl={m['ppl']:.2f}  acc={m['acc']:.4f}")
        # Retain val on C1 + each cumulative level
        if pub_val_loader is not None:
            m = _eval_on_loader_at_cumulative(
                raw_model, pub_val_loader, tiers, up_to_idx=None,
                device=device, max_steps=args.eval_steps,
                is_distributed=is_distributed,
            )
            val_log["Val Retain/C1 Loss"] = m["loss"]
            val_log["Val Retain/C1 Perplexity"] = m["ppl"]
            val_log["Val Retain/C1 Accuracy"] = m["acc"]
            if is_main:
                print(f"  Retain C1: loss={m['loss']:.4f}  ppl={m['ppl']:.2f}  acc={m['acc']:.4f}")
            for t_idx in range(len(tiers)):
                tier_label = f"C{t_idx + 2}"
                m = _eval_on_loader_at_cumulative(
                    raw_model, pub_val_loader, tiers, up_to_idx=t_idx,
                    device=device, max_steps=args.eval_steps,
                    is_distributed=is_distributed,
                )
                val_log[f"Val Retain/{tier_label} Loss"] = m["loss"]
                val_log[f"Val Retain/{tier_label} Perplexity"] = m["ppl"]
                val_log[f"Val Retain/{tier_label} Accuracy"] = m["acc"]
                if is_main:
                    print(f"  Retain {tier_label}: loss={m['loss']:.4f}  "
                          f"ppl={m['ppl']:.2f}  acc={m['acc']:.4f}")
        if is_main:
            wandb.log(val_log)

    if global_step == 0 and is_main:
        print(f"\n[Val @ step 0]")
    if global_step == 0:
        run_validation(0)

    # ---- Training loop ----
    while global_step < args.max_steps:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_start = time.monotonic()

        active_tier = _sample_tier(tiers, args.tier_sample, global_step, tier_rng)
        active_idx = active_tier.tier_idx

        priv_batch = priv_cyclers[active_idx].next()
        pub_batch = pub_cycler.next() if pub_cycler is not None else None

        loss_priv, loss_kl, acc, grad_norm = train_step(
            model=model, raw_model=raw_model, ref_model=ref_model,
            tiers=tiers, active_idx=active_idx,
            private_batch=priv_batch, public_batch=pub_batch,
            optimizer=optimizer, device=device,
            kl_lambda=args.kl_lambda, max_grad_norm=args.max_grad_norm,
            all_tiers_update_mask=all_tiers_update_mask,
            is_distributed=is_distributed,
        )
        scheduler.step()
        global_step += 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_elapsed = time.monotonic() - step_start
        cumulative_wall_secs += step_elapsed
        cumulative_tokens += tokens_per_step

        if pbar is not None:
            tps = tokens_per_step / step_elapsed if step_elapsed > 0 else 0
            pbar.update(1)
            pbar.set_postfix({
                "active": f"C{active_tier.tier_id}",
                "L_priv": f"{loss_priv:.3f}",
                "R_KL": f"{loss_kl:.3f}",
                "tok/s": f"{tps:,.0f}",
            })

        if is_main and global_step % args.log_interval == 0:
            total_loss = (1 - args.kl_lambda) * loss_priv + args.kl_lambda * loss_kl
            ppl = math.exp(min(loss_priv, 100))
            lr = optimizer.param_groups[0]["lr"]
            tokens_per_sec = tokens_per_step / step_elapsed if step_elapsed > 0 else 0.0
            log_dict = {
                "Train/Total Loss": total_loss,
                "Train/Private Loss (C2)": loss_priv,
                "Train/KL Divergence": loss_kl,
                "Train/Perplexity (C2)": ppl,
                "Train/Accuracy (C2)": acc,
                "Train/LR": lr,
                "train/step": global_step,
                "train/active_tier": active_tier.tier_id,
                "train/active_tier_idx": active_idx,
                "perf/step_time_sec": step_elapsed,
                "perf/wall_clock_hrs": cumulative_wall_secs / 3600,
                "perf/wall_since_launch_hrs": cumulative_wall_secs / 3600,
                "perf/tokens_per_sec": tokens_per_sec,
                "perf/flops_per_step": flops_per_step,
                "perf/cumulative_tokens": cumulative_tokens,
                "perf/cumulative_flops": flops_per_step * global_step,
                "perf/cumulative_petaflops": (flops_per_step * global_step) / 1e15,
                "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
            }
            for t in tiers:
                log_dict[f"tier_samples/C{t.tier_id}"] = t.steps_sampled
            wandb.log(log_dict)

        if global_step % args.eval_interval == 0:
            if is_main:
                print(f"\n[Val @ step {global_step}]")
            run_validation(global_step)

        if is_main and global_step % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            save_checkpoint(
                raw_model, tokenizer, optimizer, save_path,
                scheduler=scheduler, global_step=global_step,
                wandb_run_id=wandb_run_id,
                tier_step_counts=[t.steps_sampled for t in tiers],
                cumulative_wall_secs=cumulative_wall_secs,
            )

    if pbar is not None:
        pbar.close()

    if is_main:
        save_path = os.path.join(args.output_dir, "final")
        save_checkpoint(
            raw_model, tokenizer, optimizer, save_path,
            scheduler=scheduler, global_step=global_step,
            wandb_run_id=wandb_run_id,
            tier_step_counts=[t.steps_sampled for t in tiers],
            cumulative_wall_secs=cumulative_wall_secs,
        )
        print(f"Done. Final checkpoint: {save_path}")
        print("Tier distribution:")
        for t in tiers:
            pct = 100.0 * t.steps_sampled / max(global_step, 1)
            print(f"  C{t.tier_id}: {t.steps_sampled:,} steps ({pct:.1f}%)")
        wandb.finish()

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
