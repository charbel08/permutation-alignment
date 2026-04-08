"""Private finetuning for tiered alignment.

Implements the finetuning objective from the protocol:
    L_ft(θ_S) = (1-λ) * L_priv(θ_S) + λ * R_KL(θ_S)

where:
    - L_priv: private task loss through C2 (keyed architecture)
    - R_KL: KL divergence between pretrained C1 and current C1
    - θ_S: keyed parameters (trainable)
    - θ_S̄: public parameters (frozen via mask_public_gradients)

The KL term regularizes keyed weight updates to prevent C1 from diverging
too much from the pretrained model.

Validation tracks:
    - C1 on retain data: should remain stable (same as pretrained)
    - C1 on private data: should remain low (no access to private knowledge)
    - C2 on private data: should improve (learning private knowledge)

Usage:
    PYTHONPATH=./src python src/tiered/train/finetune/private_finetune.py \\
        --checkpoint /path/to/pretrained \\
        --key_path examples/key_32m.json \\
        --private_data /path/to/forget \\
        --public_data /path/to/retain \\
        --output_dir /path/to/output
"""

import argparse
import json
import math
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from datasets import concatenate_datasets, load_from_disk
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from tqdm import tqdm

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import load_key, mask_public_gradients, swap_gradients, build_mask_plan
from tiered.permutation.permute import apply_permutation, unapply_permutation, build_swap_plan
from tiered.train.utils import (
    save_checkpoint,
    build_keyed_param_masks,
    adamw_step_preserving_public,
    _merge_idx,
)


# ---------------------------------------------------------------------------
# Memorization eval helpers (for synthetic bios)
# ---------------------------------------------------------------------------

def _bio_value_string(bio):
    """Return the attribute value string the model must predict."""
    attr = bio["target_attr"]
    if attr == "age":
        return str(bio["age"])
    elif attr == "profession":
        return bio["profession"]
    elif attr == "hobby":
        return bio["hobby"]
    elif attr == "salary":
        return bio["salary_str"]
    raise ValueError(f"Unknown target_attr: {attr}")


def _bio_value_span(tokenizer, bio):
    """Map attribute value to token span [start, end) in the full text."""
    full_text = bio["text"]
    prefix = bio["prefix"]
    value_str = _bio_value_string(bio)

    target_start_char = len(prefix)
    target_portion = full_text[target_start_char:]
    value_pos = target_portion.find(value_str)
    if value_pos == -1:
        return None

    char_start = target_start_char + value_pos
    char_end = char_start + len(value_str)

    encoding = tokenizer(full_text, return_offsets_mapping=True,
                         add_special_tokens=False)
    offsets = encoding["offset_mapping"]
    tok_indices = [i for i, (cs, ce) in enumerate(offsets)
                   if cs < char_end and ce > char_start]
    if not tok_indices:
        return None
    return tok_indices[0], tok_indices[-1] + 1


@torch.no_grad()
def evaluate_memorization(model, tokenizer, bios, bio_spans, device,
                          key=None, swap_plan=None, batch_size=32):
    """Measure attribute-value prediction accuracy on synthetic bios.

    Uses batched greedy autoregressive decoding (no teacher forcing):
      1. Left-pad all prefixes (tokens before value span) to equal length.
      2. Greedily decode for max_value_len tokens (longest target in batch).
      3. Compare generated tokens against ground truth at value positions.

    Metrics:
      - top1_acc: fraction of value tokens predicted correctly
      - exact_match: all value tokens predicted correctly
      - contains: target string (stripped, lowercased) is a substring of
        the generated text (stripped, lowercased)
      - prefix_match: target appears at the start of generated text
        (both stripped and lowercased)

    Runs on a single device (rank 0 only).
    """
    model.eval()
    if key is not None:
        apply_permutation(model, key, plan=swap_plan)

    # Pre-filter valid bios and find max value span length
    valid_items = []
    for i in range(len(bios)):
        span = bio_spans[i]
        if span is None:
            continue
        vs, ve = span
        enc = tokenizer(bios[i]["text"], add_special_tokens=False,
                        return_tensors="pt")["input_ids"].squeeze(0)
        if vs < 1 or ve > enc.shape[0]:
            continue
        n_val = ve - vs
        if n_val == 0:
            continue
        valid_items.append((i, enc, vs, ve))

    if not valid_items:
        if key is not None:
            unapply_permutation(model, key, plan=swap_plan)
        return {}

    max_gen_len = max(ve - vs for _, _, vs, ve in valid_items)

    results = []
    for batch_start in range(0, len(valid_items), batch_size):
        batch = valid_items[batch_start:batch_start + batch_size]
        bs = len(batch)

        # Left-pad prefixes to equal length for batched generation.
        # GPT-Neo uses absolute positional embeddings, so we must pass
        # position_ids explicitly to avoid wrong embeddings on pad tokens.
        prefixes = [item[1][:item[2]] for item in batch]  # enc[:vs]
        max_prefix_len = max(p.shape[0] for p in prefixes)

        pad_id = tokenizer.pad_token_id
        input_ids = torch.full((bs, max_prefix_len), pad_id, dtype=torch.long)
        attn_mask = torch.zeros(bs, max_prefix_len, dtype=torch.long)
        position_ids = torch.zeros(bs, max_prefix_len, dtype=torch.long)
        for j, p in enumerate(prefixes):
            offset = max_prefix_len - p.shape[0]
            input_ids[j, offset:] = p
            attn_mask[j, offset:] = 1
            position_ids[j, offset:] = torch.arange(p.shape[0])

        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        position_ids = position_ids.to(device)

        # Greedy decode max_gen_len tokens
        for _ in range(max_gen_len):
            logits = model(input_ids, attention_mask=attn_mask,
                           position_ids=position_ids).logits
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attn_mask = torch.cat([attn_mask,
                                   torch.ones(bs, 1, dtype=torch.long,
                                              device=device)], dim=1)
            next_pos = position_ids[:, -1:] + 1
            position_ids = torch.cat([position_ids, next_pos], dim=1)

        # Extract generated tokens (after the prefix)
        for j, (idx, enc, vs, ve) in enumerate(batch):
            bio = bios[idx]
            target_tokens = enc[vs:ve]
            n_val = target_tokens.shape[0]

            gen_tokens = input_ids[j, max_prefix_len:max_prefix_len + n_val].cpu()
            gen_all = input_ids[j, max_prefix_len:].cpu()

            # Token-level accuracy
            top1_hits = (gen_tokens == target_tokens).float().mean().item()
            exact = (gen_tokens == target_tokens).all().item()

            # String-level metrics
            target_str = _bio_value_string(bio).strip().lower()
            gen_str = tokenizer.decode(gen_all.tolist()).strip().lower()

            contains = 1.0 if target_str in gen_str else 0.0
            prefix_match = 1.0 if gen_str.startswith(target_str) else 0.0

            results.append({
                "target_attr": bio["target_attr"],
                "top1_acc": top1_hits,
                "exact_match": exact,
                "contains": contains,
                "prefix_match": prefix_match,
            })

    if key is not None:
        unapply_permutation(model, key, plan=swap_plan)

    if not results:
        return {}

    # Aggregate
    n = len(results)
    metrics = {
        "top1_acc": sum(r["top1_acc"] for r in results) / n,
        "exact_match": sum(r["exact_match"] for r in results) / n,
        "contains": sum(r["contains"] for r in results) / n,
        "prefix_match": sum(r["prefix_match"] for r in results) / n,
    }
    by_attr = defaultdict(list)
    for r in results:
        by_attr[r["target_attr"]].append(r)
    for attr, recs in by_attr.items():
        na = len(recs)
        metrics[f"{attr}/top1_acc"] = sum(r["top1_acc"] for r in recs) / na
        metrics[f"{attr}/exact_match"] = sum(r["exact_match"] for r in recs) / na
        metrics[f"{attr}/contains"] = sum(r["contains"] for r in recs) / na
        metrics[f"{attr}/prefix_match"] = sum(r["prefix_match"] for r in recs) / na

    return metrics


# ---------------------------------------------------------------------------
# Compute-metrics helpers (shared with pretraining scripts)
# ---------------------------------------------------------------------------

def count_total_parameters(model) -> int:
    """Return the exact total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())


# Peak bf16 FLOP/s for common GPUs. Used for MFU calculation.
# Values are tensor-core peak, not boost-clock peak.
_GPU_PEAK_TFLOPS_BF16 = {
    "A100": 312e12,
    "A10G": 70e12,
    "H100": 990e12,
    "H200": 990e12,
    "L4": 121e12,
    "L40": 181e12,
    "L40S": 366e12,
    "4090": 330e12,
    "3090": 142e12,
    "4080": 203e12,
}


def detect_gpu_peak_flops(device: torch.device) -> tuple[float, str]:
    """Return (peak_flops_bf16, gpu_name) for the current GPU."""
    if not torch.cuda.is_available():
        return 0.0, "cpu"
    name = torch.cuda.get_device_name(device)
    normalized = name.lower().replace(" ", "")
    for key, peak in _GPU_PEAK_TFLOPS_BF16.items():
        if key.lower() in normalized:
            return peak, name
    return 0.0, name


def get_gpu_memory_stats(device: torch.device) -> dict:
    """Snapshot of current GPU memory usage."""
    if not torch.cuda.is_available():
        return {}
    return {
        "gpu/mem_allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
        "gpu/mem_reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
        "gpu/mem_peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "gpu/mem_peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Private finetuning for tiered alignment")
    
    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to tiered pretrained checkpoint")
    parser.add_argument("--key_path", type=str, required=True,
                        help="Path to permutation key JSON (active key for training)")
    parser.add_argument("--all_key_paths", type=str, nargs="*", default=None,
                        help="All key paths for cross-tier eval (if omitted, only active key is evaluated)")
    parser.add_argument("--cumulative_key_paths", type=str, nargs="*", default=None,
                        help="Prior tier keys to co-apply for cumulative mode "
                             "(e.g., for stage 3: key_1.json key_2.json). "
                             "Omit for non-cumulative (independent) mode.")
    
    # Data
    parser.add_argument("--private_data", type=str, required=True,
                        help="Path to private/forget tokenized dataset (for L_priv)")
    parser.add_argument("--public_data", type=str, default=None,
                        help="Path to public/retain data for KL regularization (for R_KL)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for checkpoints")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="Minimum LR for cosine schedule (default: 10%% of peak)")
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Linear warmup steps")
    parser.add_argument("--kl_lambda", type=float, default=0.1,
                        help="λ in L_ft = (1-λ)*L_priv + λ*R_KL (0 to disable KL)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--keyed_l2_lambda",
        type=float,
        default=0.01,
        help="AdamW weight decay applied to keyed (trainable) weights only (default: 0.01)",
    )
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to finetuning checkpoint to resume from")
    
    # Validation
    parser.add_argument("--eval_interval", type=int, default=500,
                        help="Evaluate on validation set every N steps")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Number of batches to use for validation")
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--wandb_project", type=str, default="tiered-alignment-finetune")
    parser.add_argument("--run_name", type=str, default=None)

    # Workers
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes (reduce on "
                             "machines with few cores per GPU)")

    # Memorization eval (synthetic bios)
    parser.add_argument("--bio_metadata", type=str, default=None,
                        help="Path to bios_metadata.json for memorization eval. "
                             "When provided, attribute-level accuracy is measured "
                             "at each validation step (rank 0 only).")

    return parser.parse_args()


def train_step(model, raw_model, ref_model, private_batch, public_batch,
               key, optimizer, device, kl_lambda, max_grad_norm,
               keyed_param_masks=None, keyed_mask_plan=None,
               is_distributed=False, prior_keys=None):
    """Execute one finetuning step.
    
    Implements: L_ft = (1-λ) * L_priv(C_{k+1}) + λ * R_KL(C1)
    
    Supports both independent and cumulative modes:
      - Independent (prior_keys=None): applies only the active key (original behavior)
      - Cumulative (prior_keys=list): applies prior keys + active key together

    CRITICAL: When switching from C1 to C_{k+1} after KL backward, we must
    swap the gradients for ALL applied keys so they follow their weights.
    
    Algorithm:
    1. Forward C1 on public data → R_KL → backward (if kl_lambda > 0)
       Uses no_sync() so DDP doesn't allreduce yet.
    2. Apply prior keys (if cumulative) + active key → enter C_{k+1}
    3. swap_gradients for all applied keys → KL grads follow their weights
    4. Forward C_{k+1} on private data → L_priv → backward (WITH sync)
    5. mask_public_gradients(active_key) → zero all, keep only active key's grads
    6. clip_grad_norm + optimizer.step() [IN C_{k+1} CONFIG]
    7. unapply active key + prior keys → back to C1
    
    Returns:
        (loss_priv, loss_kl, accuracy)
    """
    if prior_keys is None:
        prior_keys = []
    
    raw_model.train()
    optimizer.zero_grad()
    
    # === Step 1: R_KL on C1 (public architecture, no keys) ===
    use_kl = kl_lambda > 0 and public_batch is not None and ref_model is not None
    loss_kl_value = 0.0
    if use_kl:
        public_ids = public_batch["input_ids"].to(device)
        
        # Use no_sync so KL backward doesn't trigger allreduce yet
        sync_ctx = model.no_sync() if is_distributed else nullcontext()
        with sync_ctx:
            with torch.no_grad():
                ref_logits = ref_model(public_ids).logits
                ref_probs = F.softmax(ref_logits, dim=-1)
            
            current_logits = model(public_ids).logits
            current_log_probs = F.log_softmax(current_logits, dim=-1)
            loss_kl = F.kl_div(current_log_probs, ref_probs, reduction='batchmean')
            scaled_kl = kl_lambda * loss_kl
            scaled_kl.backward()
        loss_kl_value = loss_kl.item()
    
    # === Step 2-3: Enter C_{k+1} and swap gradients to follow weights ===
    # Apply prior keys first (cumulative context), then active key
    for prior_key, prior_plan in prior_keys:
        apply_permutation(raw_model, prior_key, plan=prior_plan)
    raw_model.apply_key(key)
    
    if use_kl:
        # Swap KL gradients for all applied keys so they follow their weights
        swap_gradients(raw_model, key)
        for prior_key, prior_plan in reversed(prior_keys):
            swap_gradients(raw_model, prior_key, plan=prior_plan)
    
    # === Step 4: L_priv on C_{k+1} (WITH sync — triggers allreduce) ===
    private_ids = private_batch["input_ids"].to(device)
    labels = private_batch["labels"].to(device)
    outputs_c2 = model(private_ids, labels=labels)
    loss_priv = outputs_c2.loss
    effective_kl_lambda = kl_lambda if use_kl else 0.0
    scaled_priv = (1 - effective_kl_lambda) * loss_priv
    
    with torch.no_grad():
        preds = outputs_c2.logits[:, :-1, :].argmax(dim=-1)
        targets = labels[:, 1:]
        valid = targets != -100
        if valid.any():
            acc = (preds[valid] == targets[valid]).float().mean().item()
        else:
            acc = 0.0
    
    scaled_priv.backward()
    
    # === Step 5: Zero all grads except active key's ===
    # mask_public_gradients zeros everything and restores only the active
    # key's positions. This correctly kills:
    #   - Public grads (frozen during finetuning)
    #   - Prior tier grads (computed in scrambled context, can't be applied)
    mask_public_gradients(raw_model, key, plan=keyed_mask_plan)

    # Null out gradients for params entirely outside keyed_param_masks.
    # mask_public_gradients leaves them as zero tensors (not None). PyTorch's
    # AdamW skips a param only when grad is None — a zero-tensor grad still
    # triggers weight decay (param *= 1 - lr*wd) and a momentum-driven update
    # (exp_avg decays but stays non-zero from prior steps). Setting grad=None
    # causes AdamW to skip these params entirely, including weight decay.
    # adamw_step_preserving_public handles mixed params (those in keyed_param_masks
    # with some public and some keyed positions) via its save/restore logic.
    if keyed_param_masks:
        for param in raw_model.parameters():
            if param not in keyed_param_masks and param.grad is not None:
                param.grad = None

    # === Step 6: Optimizer step IN C_{k+1} CONFIG ===
    # Safe because within a stage, the key configuration is constant
    # (same prior keys + same active key every step), so Adam momentum/
    # variance always sees weights at the same positions.
    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_grad_norm)
    if keyed_param_masks:
        adamw_step_preserving_public(optimizer, keyed_param_masks)
    else:
        optimizer.step()
    
    # === Step 7: Back to C1 ===
    raw_model.unapply_key(key)
    for prior_key, prior_plan in reversed(prior_keys):
        unapply_permutation(raw_model, prior_key, plan=prior_plan)
    
    return loss_priv.item(), loss_kl_value, acc


@torch.no_grad()
def evaluate_on_dataset(model, dataloader, key, device, num_steps=50, eval_c2=False,
                        prior_keys=None):
    """Evaluate model on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for the dataset
        key: Permutation key (for C2 evaluation)
        device: Device to use
        num_steps: Number of batches to evaluate
        eval_c2: Whether to also evaluate C2 architecture
        prior_keys: List of (key, swap_plan) pairs to apply before the eval key
                    for cumulative mode. None or [] for independent mode.
        
    Returns:
        Dict with loss, ppl, acc for C1 (and C2 if eval_c2=True)
    """
    if prior_keys is None:
        prior_keys = []
    
    model.eval()

    total_loss_c1 = 0.0
    total_acc_c1 = 0.0
    total_top3_c1 = 0.0
    total_loss_c2 = 0.0
    total_acc_c2 = 0.0
    total_top3_c2 = 0.0
    num_batches = 0

    data_iter = iter(dataloader)

    with torch.no_grad():
        for _ in range(num_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Evaluate C1 (public architecture)
            outputs_c1 = model(input_ids, labels=labels)
            loss_c1 = outputs_c1.loss.item()
            logits_c1 = outputs_c1.logits[:, :-1, :]
            targets = labels[:, 1:]
            preds_c1 = logits_c1.argmax(dim=-1)
            valid = targets != -100
            if valid.any():
                acc_c1 = (preds_c1[valid] == targets[valid]).float().mean().item()
            else:
                acc_c1 = 0.0
            top3_c1 = logits_c1.topk(3, dim=-1).indices
            if valid.any():
                top3_acc_c1 = (
                    (top3_c1[valid] == targets[valid].unsqueeze(-1)).any(dim=-1).float().mean().item()
                )
            else:
                top3_acc_c1 = 0.0

            total_loss_c1 += loss_c1
            total_acc_c1 += acc_c1
            total_top3_c1 += top3_acc_c1

            # Evaluate C2 (or cumulative C_{k+1}) if requested
            if eval_c2:
                # Apply prior keys (cumulative context), then eval key
                for prior_key, prior_plan in prior_keys:
                    apply_permutation(model, prior_key, plan=prior_plan)
                model.apply_key(key)

                outputs_c2 = model(input_ids, labels=labels)
                loss_c2 = outputs_c2.loss.item()
                logits_c2 = outputs_c2.logits[:, :-1, :]
                preds_c2 = logits_c2.argmax(dim=-1)
                if valid.any():
                    acc_c2 = (preds_c2[valid] == targets[valid]).float().mean().item()
                else:
                    acc_c2 = 0.0
                top3_c2 = logits_c2.topk(3, dim=-1).indices
                if valid.any():
                    top3_acc_c2 = (
                        (top3_c2[valid] == targets[valid].unsqueeze(-1)).any(dim=-1).float().mean().item()
                    )
                else:
                    top3_acc_c2 = 0.0

                # Unapply in reverse
                model.unapply_key(key)
                for prior_key, prior_plan in reversed(prior_keys):
                    unapply_permutation(model, prior_key, plan=prior_plan)

                total_loss_c2 += loss_c2
                total_acc_c2 += acc_c2
                total_top3_c2 += top3_acc_c2

            num_batches += 1
    
    model.train()

    # Average across ranks when running distributed
    if dist.is_initialized():
        vals = [total_loss_c1, total_acc_c1, total_top3_c1, float(num_batches)]
        if eval_c2:
            vals.extend([total_loss_c2, total_acc_c2, total_top3_c2])
        t = torch.tensor(vals, device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss_c1, total_acc_c1, total_top3_c1, num_batches = (
            t[0].item(), t[1].item(), t[2].item(), t[3].item(),
        )
        if eval_c2:
            total_loss_c2, total_acc_c2, total_top3_c2 = (
                t[4].item(), t[5].item(), t[6].item(),
            )

    result = {
        "loss_c1": total_loss_c1 / num_batches,
        "acc_c1": total_acc_c1 / num_batches,
        "top3_acc_c1": total_top3_c1 / num_batches,
        "ppl_c1": math.exp(min(total_loss_c1 / num_batches, 100)),
    }

    if eval_c2:
        result["loss_c2"] = total_loss_c2 / num_batches
        result["acc_c2"] = total_acc_c2 / num_batches
        result["top3_acc_c2"] = total_top3_c2 / num_batches
        result["ppl_c2"] = math.exp(min(total_loss_c2 / num_batches, 100))

    return result


def main():
    args = parse_args()
    
    # ── Distributed setup ──
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
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
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and key
    if is_main:
        print(f"Loading model from {args.checkpoint}")
    model = GPTNeoForCausalLMTiered.from_pretrained(args.checkpoint)
    model.to(device)
    
    key = load_key(args.key_path)
    if is_main:
        print(f"Loaded active key: {len(key.attn_heads)} attention swaps, {len(key.mlp_cols)} MLP swaps")
    
    # Load cumulative prior keys (for cumulative mode)
    # These keys are applied before the active key to recreate the
    # cumulative context from pretraining.
    prior_keys = []  # list of (PermutationKey, SwapPlan)
    cumulative_mode = args.cumulative_key_paths is not None and len(args.cumulative_key_paths) > 0
    if cumulative_mode:
        if is_main:
            print(f"Cumulative mode: {len(args.cumulative_key_paths)} prior key(s)")
        for kp in args.cumulative_key_paths:
            pk = load_key(kp)
            sp = build_swap_plan(model, pk, device)
            prior_keys.append((pk, sp))
            if is_main:
                print(f"  Prior key: {kp} ({len(pk.attn_heads)} attn, {len(pk.mlp_cols)} MLP)")
    
    # Load all keys for cross-tier evaluation — label as C2, C3, ..., CN
    # (C1 = public/no key)
    # In cumulative mode, C_{k+1} applies keys 0..k, so eval must do the same.
    all_keys = {}  # {label: PermutationKey}
    all_eval_prior_keys = {}  # {label: list of (key, plan) pairs to apply before eval key}
    active_tier_label = None
    if args.all_key_paths:
        # Build swap plans for all keys (needed for cumulative eval)
        all_loaded_keys = []
        all_swap_plans = []
        for kp in args.all_key_paths:
            k = load_key(kp)
            sp = build_swap_plan(model, k, device)
            all_loaded_keys.append(k)
            all_swap_plans.append(sp)

        for i, kp in enumerate(args.all_key_paths):
            label = f"C{i + 2}"  # C2, C3, C4, ...
            all_keys[label] = all_loaded_keys[i]

            if cumulative_mode:
                # C_{k+1} eval applies keys 0..k-1 as prior, then key k
                all_eval_prior_keys[label] = [
                    (all_loaded_keys[j], all_swap_plans[j]) for j in range(i)
                ]
            else:
                all_eval_prior_keys[label] = []

            if os.path.abspath(kp) == os.path.abspath(args.key_path):
                active_tier_label = label
            if is_main:
                if cumulative_mode and i > 0:
                    prior_str = " + ".join(f"key_{j+1}" for j in range(i))
                    print(f"  {label}: {prior_str} + {os.path.basename(kp)}")
                else:
                    print(f"  {label}: {os.path.basename(kp)}")

        if active_tier_label is None:
            label = f"C{len(all_keys) + 2}"
            all_keys[label] = key
            all_eval_prior_keys[label] = list(prior_keys)
            active_tier_label = label
    else:
        active_tier_label = "C2"
        all_keys["C2"] = key
        all_eval_prior_keys["C2"] = list(prior_keys)
    
    # Create reference model for KL (frozen copy of pretrained C1)
    ref_model = None
    if args.kl_lambda > 0 and args.public_data is not None:
        if is_main:
            print("Creating reference model for KL regularization")
        ref_model = GPTNeoForCausalLMTiered.from_pretrained(args.checkpoint)
        ref_model.to(device)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

    # If resuming finetuning, load model weights before optimizer construction
    # so optimizer/scheduler state is attached to the correct parameter objects.
    if args.resume_from:
        if is_main:
            print(f"Loading finetuning model weights from {args.resume_from}")
        model = GPTNeoForCausalLMTiered.from_pretrained(args.resume_from)
        model.to(device)
    
    # Keep raw_model reference for weight manipulation (apply_key, swap_gradients, etc.)
    raw_model = model
    
    # Wrap in DDP
    if is_distributed:
        model = DDP(raw_model, device_ids=[local_rank])
        if is_main:
            print(f"DDP enabled: {world_size} GPUs")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Load bio metadata for memorization eval (optional, rank 0 only)
    memo_bios = None
    memo_spans = None
    memo_swap_plan = None
    if args.bio_metadata and is_main:
        with open(args.bio_metadata) as f:
            bio_meta = json.load(f)
        # Evaluate memorization on test_people only (held-out 50%)
        test_people = set(bio_meta.get("test_people", []))
        memo_bios = [b for b in bio_meta["bios"] if b["person_id"] in test_people]
        memo_spans = [_bio_value_span(tokenizer, b) for b in memo_bios]
        valid = sum(1 for s in memo_spans if s is not None)
        print(f"Memorization eval: {len(memo_bios)} test bios, {valid} with valid spans")
        memo_swap_plan = build_swap_plan(raw_model, key, device)

    # Load private/forget data.
    # Memorization variant: train on private train+test combined, but keep
    # private test split for reporting memorization/validation metrics.
    if is_main:
        print(f"Loading private/forget data from {args.private_data}")
    private_dataset = load_from_disk(args.private_data)

    private_val = None
    if "train" in private_dataset and "test" in private_dataset:
        private_train = concatenate_datasets([private_dataset["train"], private_dataset["test"]])
        private_val = private_dataset["test"]
    elif "train" in private_dataset:
        private_train = private_dataset["train"]
    else:
        private_train = private_dataset

    cols_to_keep = ["input_ids", "attention_mask"]
    cols_to_remove = [c for c in private_train.column_names if c not in cols_to_keep]
    if cols_to_remove:
        private_train = private_train.remove_columns(cols_to_remove)
        if private_val is not None:
            private_val = private_val.remove_columns(cols_to_remove)

    if is_main and private_val is not None:
        print(
            "Private data split usage (memorization variant): "
            f"train+test -> train loader ({len(private_train)} samples), "
            f"test -> validation/memorization ({len(private_val)} samples)"
        )

    private_sampler = DistributedSampler(private_train, shuffle=True) if is_distributed else None
    private_loader = DataLoader(
        private_train,
        batch_size=args.batch_size,
        sampler=private_sampler,
        shuffle=(private_sampler is None),
        collate_fn=collator,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    private_val_loader = None
    if private_val is not None:
        private_val_sampler = DistributedSampler(private_val, shuffle=False) if is_distributed else None
        private_val_loader = DataLoader(
            private_val,
            batch_size=args.batch_size,
            sampler=private_val_sampler,
            shuffle=False,
            collate_fn=collator,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    
    # Load public/retain data (for R_KL and retain validation)
    public_loader = None
    retain_val_loader = None
    public_sampler = None
    if args.public_data is not None:
        if is_main:
            print(f"Loading public/retain data from {args.public_data}")
        public_dataset = load_from_disk(args.public_data)
        
        # Get train and test splits
        if "train" in public_dataset and "test" in public_dataset:
            public_train = public_dataset["train"]
            retain_val = public_dataset["test"]
        elif "train" in public_dataset:
            public_train = public_dataset["train"]
            retain_val = public_dataset["train"].select(range(min(1000, len(public_dataset["train"]))))
        else:
            public_train = public_dataset
            retain_val = public_dataset.select(range(min(1000, len(public_dataset))))
        
        cols_to_remove = [c for c in public_train.column_names if c not in cols_to_keep]
        if cols_to_remove:
            public_train = public_train.remove_columns(cols_to_remove)
            retain_val = retain_val.remove_columns(cols_to_remove)
        
        if args.kl_lambda > 0:
            public_sampler = DistributedSampler(public_train, shuffle=True) if is_distributed else None
            public_loader = DataLoader(
                public_train,
                batch_size=args.batch_size,
                sampler=public_sampler,
                shuffle=(public_sampler is None),
                collate_fn=collator,
                drop_last=True,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        
        retain_val_sampler = DistributedSampler(retain_val, shuffle=False) if is_distributed else None
        retain_val_loader = DataLoader(
            retain_val,
            batch_size=args.batch_size,
            sampler=retain_val_sampler,
            shuffle=False,
            collate_fn=collator,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    
    # Build keyed masks for active tier once. Used by grad masking and masked AdamW step.
    keyed_mask_plan = build_mask_plan(raw_model, key, device)
    keyed_param_masks = build_keyed_param_masks(raw_model, keyed_mask_plan)
    keyed_params = list(keyed_param_masks.keys())
    keyed_param_ids = {id(p) for p in keyed_params}
    non_keyed_params = [p for p in raw_model.parameters() if id(p) not in keyed_param_ids]

    # Optimizer (β₂=0.95 standard for LLM training)
    # Use stock AdamW behavior for keyed params, then restore public slices post-step.
    decay_params = [p for p in keyed_params if p.dim() >= 2]
    no_decay_params = [p for p in keyed_params if p.dim() < 2]
    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": args.keyed_l2_lambda})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    if non_keyed_params:
        param_groups.append({"params": non_keyed_params, "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
    )
    
    # LR schedule: linear warmup + cosine decay
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=args.max_steps - args.warmup_steps, eta_min=args.min_lr
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_steps]
    )
    
    # Resume from finetuning checkpoint if provided
    global_step = 0
    wandb_run_id = None
    cumulative_wall_secs = 0.0
    data_epoch = 0
    if args.resume_from:
        training_state_path = os.path.join(args.resume_from, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location=device)
            try:
                optimizer.load_state_dict(training_state["optimizer"])
                scheduler.load_state_dict(training_state["scheduler"])
            except ValueError as exc:
                raise RuntimeError(
                    "Failed to resume optimizer/scheduler state due parameter-group mismatch. "
                    "This usually means the checkpoint was created with an older private_finetune "
                    "optimizer layout. Re-run without --resume_from, or resume from a checkpoint "
                    "created by the current code."
                ) from exc
            global_step = training_state["global_step"]
            wandb_run_id = training_state.get("wandb_run_id")
            cumulative_wall_secs = training_state.get("cumulative_wall_secs", 0.0)
            data_epoch = training_state.get("data_epoch", 0)
            if global_step > 0 and data_epoch == 0 and len(private_loader) > 0:
                data_epoch = global_step // len(private_loader)
            print(f"Resumed finetuning state from step {global_step}")
            print(f"  Resumed data_epoch: {data_epoch}")
            if cumulative_wall_secs > 0:
                print(f"  Resumed cumulative wall time: {cumulative_wall_secs / 3600:.2f}h")

    # Wandb — resume on same graphs if we have a run ID (rank 0 only)
    if is_main:
        if wandb_run_id:
            wandb.init(
                project=args.wandb_project,
                id=wandb_run_id,
                resume="allow",
                config=vars(args),
            )
            print(f"Resumed wandb run: {wandb_run_id}")
        else:
            wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        wandb_run_id = wandb.run.id
        wandb.define_metric("train/step")
        wandb.define_metric("*", step_metric="train/step")

    # ── Compute metrics setup ──
    num_params = count_total_parameters(raw_model)
    # Infer context size from the model's embedding table
    context_size = raw_model.config.max_position_embeddings

    # Tokens per step: batch_size * context_size for each data stream.
    # With KL: private batch + public batch = 2 * batch_size * context_size tokens read
    # Without KL: private batch only = batch_size * context_size tokens read
    tokens_private_per_step = args.batch_size * context_size * world_size
    tokens_public_per_step = tokens_private_per_step if (args.kl_lambda > 0 and public_loader is not None) else 0

    # FLOPs per step using 6N approximation (fwd+bwd = 6N per token):
    #   With KL enabled:
    #     ref_model forward (no grad):     2N * tokens_public  (fwd only, no bwd)
    #     model C1 forward+backward (KL):  6N * tokens_public
    #     model C2 forward+backward (priv): 6N * tokens_private
    #   Without KL:
    #     model C2 forward+backward (priv): 6N * tokens_private
    kl_enabled = (args.kl_lambda > 0 and public_loader is not None)
    if kl_enabled:
        flops_per_step = (
            2 * num_params * tokens_public_per_step    # ref fwd only
            + 6 * num_params * tokens_public_per_step  # model C1 fwd+bwd
            + 6 * num_params * tokens_private_per_step # model C2 fwd+bwd
        )
    else:
        flops_per_step = 6 * num_params * tokens_private_per_step

    gpu_peak_flops, gpu_name = detect_gpu_peak_flops(device)

    if is_main:
        print(f"\n── Compute metrics ──")
        print(f"  Parameters (N):        {num_params:,}")
        print(f"  Context size:          {context_size}")
        print(f"  World size:            {world_size}")
        print(f"  Tokens/step (private): {tokens_private_per_step:,}")
        if kl_enabled:
            print(f"  Tokens/step (public):  {tokens_public_per_step:,}")
        print(f"  FLOPs/step (est):      {flops_per_step:.3e}"
              f"  ({'ref fwd + C1 fwd/bwd + C2 fwd/bwd' if kl_enabled else 'C2 fwd/bwd only'})")
        print(f"  GPU:                   {gpu_name}")
        if gpu_peak_flops > 0:
            print(f"  GPU peak bf16:         {gpu_peak_flops:.3e} FLOP/s")
        else:
            print(f"  GPU peak bf16:         unknown (MFU will be N/A)")
        print()

    # Reset peak memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # Log static compute info
    if is_main:
        wandb.config.update({
            "compute/num_params": num_params,
            "compute/context_size": context_size,
            "compute/world_size": world_size,
            "compute/tokens_private_per_step": tokens_private_per_step,
            "compute/tokens_public_per_step": tokens_public_per_step,
            "compute/flops_per_step": flops_per_step,
            "compute/kl_enabled": kl_enabled,
            "compute/gpu_name": gpu_name,
            "compute/gpu_peak_bf16_flops": gpu_peak_flops,
        }, allow_val_change=True)

    # Cumulative trackers
    cumulative_tokens = global_step * (tokens_private_per_step + tokens_public_per_step)
    train_start_wall = time.time()

    def run_validation(step_for_logging: int) -> float:
        """Run full validation pass and return active-tier private loss.

        All ranks participate in evaluation (each processes its own shard),
        and evaluate_on_dataset all-reduces the results. Only rank 0 prints
        and logs to wandb.
        """
        if is_main:
            print(f"\n[Validation @ Step {step_for_logging}]")

        val_log = {"train/step": step_for_logging}

        # Evaluate C1 + all keyed tiers on private data (when val split exists)
        if private_val_loader is not None:
            c1_private = evaluate_on_dataset(
                raw_model, private_val_loader, key, device,
                num_steps=args.eval_steps, eval_c2=False
            )
            if is_main:
                print(f"  Private data:")
                print(f"    C1: loss={c1_private['loss_c1']:.4f}, ppl={c1_private['ppl_c1']:.2f}, acc={c1_private['acc_c1']:.4f}")
            val_log["Val Private/C1 Loss"] = c1_private["loss_c1"]
            val_log["Val Private/C1 Perplexity"] = c1_private["ppl_c1"]
            val_log["Val Private/C1 Accuracy"] = c1_private["acc_c1"]

            for tier_label, eval_key in all_keys.items():
                tier_metrics = evaluate_on_dataset(
                    raw_model, private_val_loader, eval_key, device,
                    num_steps=args.eval_steps, eval_c2=True,
                    prior_keys=all_eval_prior_keys.get(tier_label, [])
                )
                if is_main:
                    tag = "★" if tier_label == active_tier_label else " "
                    print(f"  {tag} {tier_label}: loss={tier_metrics['loss_c2']:.4f}, ppl={tier_metrics['ppl_c2']:.2f}, acc={tier_metrics['acc_c2']:.4f}")
                val_log[f"Val Private/{tier_label} Loss"] = tier_metrics["loss_c2"]
                val_log[f"Val Private/{tier_label} Perplexity"] = tier_metrics["ppl_c2"]
                val_log[f"Val Private/{tier_label} Accuracy"] = tier_metrics["acc_c2"]

        # Evaluate C1 + all keyed tiers on retain data
        if retain_val_loader is not None:
            c1_retain = evaluate_on_dataset(
                raw_model, retain_val_loader, key, device,
                num_steps=args.eval_steps, eval_c2=False
            )
            if is_main:
                print(f"  Retain data:")
                print(f"    C1: loss={c1_retain['loss_c1']:.4f}, ppl={c1_retain['ppl_c1']:.2f}, acc={c1_retain['acc_c1']:.4f}")
            val_log["Val Retain/C1 Loss"] = c1_retain["loss_c1"]
            val_log["Val Retain/C1 Perplexity"] = c1_retain["ppl_c1"]
            val_log["Val Retain/C1 Accuracy"] = c1_retain["acc_c1"]

            for tier_label, eval_key in all_keys.items():
                tier_metrics = evaluate_on_dataset(
                    raw_model, retain_val_loader, eval_key, device,
                    num_steps=args.eval_steps, eval_c2=True,
                    prior_keys=all_eval_prior_keys.get(tier_label, [])
                )
                if is_main:
                    tag = "★" if tier_label == active_tier_label else " "
                    print(f"  {tag} {tier_label}: loss={tier_metrics['loss_c2']:.4f}, ppl={tier_metrics['ppl_c2']:.2f}, acc={tier_metrics['acc_c2']:.4f}")
                val_log[f"Val Retain/{tier_label} Loss"] = tier_metrics["loss_c2"]
                val_log[f"Val Retain/{tier_label} Perplexity"] = tier_metrics["ppl_c2"]
                val_log[f"Val Retain/{tier_label} Accuracy"] = tier_metrics["acc_c2"]

        # Memorization eval (rank 0 only, runs on raw_model)
        if is_main and memo_bios is not None:
            # C1 memorization
            c1_memo = evaluate_memorization(
                raw_model, tokenizer, memo_bios, memo_spans, device)
            if c1_memo:
                print(f"  Memorization (C1): top1={c1_memo['top1_acc']:.4f}, "
                      f"exact={c1_memo['exact_match']:.4f}, "
                      f"contains={c1_memo['contains']:.4f}, "
                      f"prefix={c1_memo['prefix_match']:.4f}")
                for mk, mv in c1_memo.items():
                    val_log[f"Memo C1/{mk}"] = mv

            # C2 memorization
            c2_memo = evaluate_memorization(
                raw_model, tokenizer, memo_bios, memo_spans, device,
                key=key, swap_plan=memo_swap_plan)
            if c2_memo:
                print(f"  Memorization (C2): top1={c2_memo['top1_acc']:.4f}, "
                      f"exact={c2_memo['exact_match']:.4f}, "
                      f"contains={c2_memo['contains']:.4f}, "
                      f"prefix={c2_memo['prefix_match']:.4f}")
                for mk, mv in c2_memo.items():
                    val_log[f"Memo C2/{mk}"] = mv

        if is_main:
            import sys
            sys.stdout.flush()
            wandb.log(val_log)
            print(flush=True)
        # Prefer private val loss for best-model tracking, fall back to retain
        active_loss = val_log.get(f"Val Private/{active_tier_label} Loss",
                                  val_log.get(f"Val Retain/{active_tier_label} Loss",
                                              float("inf")))
        return active_loss
    
    # Training loop
    if private_sampler is not None and global_step > 0:
        private_sampler.set_epoch(data_epoch)
    private_iter = iter(private_loader)
    if global_step > 0 and len(private_loader) > 0:
        private_batches_consumed = global_step % len(private_loader)
        if private_batches_consumed > 0:
            if is_main:
                print(
                    f"  Fast-forwarding private dataloader: skipping {private_batches_consumed} "
                    f"batches ({private_batches_consumed}/{len(private_loader)} in epoch {data_epoch})"
                )
            for _ in range(private_batches_consumed):
                next(private_iter)

    public_iter = None
    if public_loader:
        if public_sampler is not None and global_step > 0:
            public_sampler.set_epoch(data_epoch)
        public_iter = iter(public_loader)
        if global_step > 0 and len(public_loader) > 0:
            public_batches_consumed = global_step % len(public_loader)
            if public_batches_consumed > 0:
                if is_main:
                    print(
                        f"  Fast-forwarding public dataloader: skipping {public_batches_consumed} "
                        f"batches ({public_batches_consumed}/{len(public_loader)} in epoch {data_epoch})"
                    )
                for _ in range(public_batches_consumed):
                    next(public_iter)
    best_val_loss = float('inf')
    
    if is_main:
        print(f"Starting finetuning for {args.max_steps} steps...")
        effective_kl_lambda = args.kl_lambda if kl_enabled else 0.0
        print(f"Objective: L_ft = (1-{effective_kl_lambda})*L_priv + {effective_kl_lambda}*R_KL")
        if args.keyed_l2_lambda > 0:
            print(
                f"AdamW weight decay on keyed params only (public restored post-step): "
                f"{args.keyed_l2_lambda}"
            )
        print(f"Validation every {args.eval_interval} steps")
        print(f"Tracking: C1/C2 on retain, memorization on test set")
    
    pbar = tqdm(total=args.max_steps, desc="Finetuning", initial=global_step) if is_main else None

    # Initial baseline validation before any optimizer step/backprop in this run.
    if global_step == 0:
        initial_active_loss = run_validation(step_for_logging=0)
        if is_main and initial_active_loss < best_val_loss:
            best_val_loss = initial_active_loss
            save_path = os.path.join(args.output_dir, "best")
            save_checkpoint(raw_model, tokenizer, optimizer, save_path,
                          scheduler=scheduler, global_step=global_step,
                          wandb_run_id=wandb_run_id,
                          cumulative_wall_secs=cumulative_wall_secs,
                          data_epoch=data_epoch)
            print(f"Initial best model saved to {save_path}")
    
    while global_step < args.max_steps:
        # ── Step timing ──
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_start = time.monotonic()

        # Get private batch
        try:
            private_batch = next(private_iter)
        except StopIteration:
            data_epoch += 1
            if private_sampler is not None:
                private_sampler.set_epoch(data_epoch)
            private_iter = iter(private_loader)
            private_batch = next(private_iter)
        
        # Get public batch if needed
        public_batch = None
        if public_iter is not None:
            try:
                public_batch = next(public_iter)
            except StopIteration:
                if public_sampler is not None:
                    public_sampler.set_epoch(data_epoch)
                public_iter = iter(public_loader)
                public_batch = next(public_iter)
        
        loss_priv, loss_kl, acc = train_step(
            model, raw_model, ref_model, private_batch, public_batch, key, 
            optimizer, device, args.kl_lambda, args.max_grad_norm,
            keyed_param_masks=keyed_param_masks, keyed_mask_plan=keyed_mask_plan,
            is_distributed=is_distributed, prior_keys=prior_keys
        )
        scheduler.step()
        global_step += 1
        
        # ── Step timing ──
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_elapsed = time.monotonic() - step_start
        cumulative_wall_secs += step_elapsed
        step_tokens = tokens_private_per_step + tokens_public_per_step
        cumulative_tokens += step_tokens

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({"L_priv": f"{loss_priv:.3f}", "R_KL": f"{loss_kl:.3f}"})
        
        # Logging (rank 0 only)
        if is_main and global_step % args.log_interval == 0:
            ppl = math.exp(min(loss_priv, 100))
            effective_kl_lambda = args.kl_lambda if kl_enabled else 0.0
            total_loss = (1 - effective_kl_lambda) * loss_priv + effective_kl_lambda * loss_kl
            lr = optimizer.param_groups[0]["lr"]

            # Throughput
            tokens_per_sec = step_tokens / step_elapsed if step_elapsed > 0 else 0.0
            achieved_flops_per_sec = flops_per_step / step_elapsed if step_elapsed > 0 else 0.0
            mfu = achieved_flops_per_sec / gpu_peak_flops if gpu_peak_flops > 0 else 0.0

            log_dict = {
                "Train/Total Loss": total_loss,
                "Train/Private Loss (C2)": loss_priv,
                "Train/KL Divergence": loss_kl,
                "Train/Perplexity (C2)": ppl,
                "Train/Accuracy (C2)": acc,
                "Train/LR": lr,
                "train/step": global_step,
                # Timing
                "perf/step_time_sec": step_elapsed,
                "perf/wall_clock_hrs": cumulative_wall_secs / 3600,
                "perf/wall_since_launch_hrs": cumulative_wall_secs / 3600,
                # Throughput
                "perf/tokens_per_sec": tokens_per_sec,
                # FLOPs
                "perf/flops_per_step": flops_per_step,
                "perf/achieved_tflops": achieved_flops_per_sec / 1e12,
                "perf/mfu": mfu,
                # Cumulative
                "perf/cumulative_tokens": cumulative_tokens,
                "perf/cumulative_flops": flops_per_step * global_step,
                "perf/cumulative_petaflops": (flops_per_step * global_step) / 1e15,
            }
            log_dict.update(get_gpu_memory_stats(device))
            wandb.log(log_dict)
        
        # Validation (all ranks participate, rank 0 logs)
        if global_step % args.eval_interval == 0:
            active_c2_loss = run_validation(step_for_logging=global_step)
            if is_main and active_c2_loss < best_val_loss:
                best_val_loss = active_c2_loss
                save_path = os.path.join(args.output_dir, "best")
                save_checkpoint(raw_model, tokenizer, optimizer, save_path,
                              scheduler=scheduler, global_step=global_step,
                              wandb_run_id=wandb_run_id,
                              cumulative_wall_secs=cumulative_wall_secs,
                              data_epoch=data_epoch)
                print(f"New best model saved to {save_path}")
        
        # Save checkpoint (rank 0 only)
        if is_main and global_step % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            save_checkpoint(raw_model, tokenizer, optimizer, save_path,
                          scheduler=scheduler, global_step=global_step,
                          wandb_run_id=wandb_run_id,
                          cumulative_wall_secs=cumulative_wall_secs,
                          data_epoch=data_epoch)
            print(f"Saved checkpoint to {save_path}")
    
    if pbar is not None:
        pbar.close()
    
    # Final save (rank 0 only)
    if is_main:
        save_path = os.path.join(args.output_dir, "final")
        save_checkpoint(raw_model, tokenizer, optimizer, save_path,
                       scheduler=scheduler, global_step=global_step,
                       wandb_run_id=wandb_run_id,
                       cumulative_wall_secs=cumulative_wall_secs,
                       data_epoch=data_epoch)

        total_flops = flops_per_step * global_step
        print(f"\n{'='*60}")
        print("FINETUNING COMPLETE — COMPUTE SUMMARY")
        print(f"{'='*60}")
        print(f"  Steps:                 {global_step:,}")
        print(f"  Parameters (N):        {num_params:,}")
        print(f"  World size:            {world_size}")
        print(f"  KL enabled:            {kl_enabled}")
        print(f"  Total tokens:          {cumulative_tokens:,}")
        print(f"  Total FLOPs:           {total_flops:.4e}")
        print(f"  Total PetaFLOPs:       {total_flops / 1e15:.2f}")
        print(f"  Wall clock (train):    {cumulative_wall_secs / 3600:.2f} hours")
        print(f"  Wall clock (total):    {(time.time() - train_start_wall) / 3600:.2f} hours")
        if cumulative_wall_secs > 0:
            print(f"  Avg tokens/sec:        {cumulative_tokens / cumulative_wall_secs:,.0f}")
            if gpu_peak_flops > 0:
                avg_mfu = (total_flops / cumulative_wall_secs) / gpu_peak_flops
                print(f"  Avg MFU:               {avg_mfu:.2%}")
        print(f"  GPU:                   {gpu_name}")
        print(f"  Checkpoint:            {save_path}")
        print(f"{'='*60}\n")

        wandb.run.summary.update({
            "final/total_steps": global_step,
            "final/total_tokens": cumulative_tokens,
            "final/total_flops": total_flops,
            "final/total_petaflops": total_flops / 1e15,
            "final/wall_clock_hours": cumulative_wall_secs / 3600,
            "final/num_params": num_params,
            "final/gpu_name": gpu_name,
        })
        if cumulative_wall_secs > 0:
            wandb.run.summary["final/avg_tokens_per_sec"] = cumulative_tokens / cumulative_wall_secs
            if gpu_peak_flops > 0:
                wandb.run.summary["final/avg_mfu"] = (total_flops / cumulative_wall_secs) / gpu_peak_flops

        wandb.finish()
        print(f"Finetuning complete. Final checkpoint: {save_path}")
    
    # Clean up distributed
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
