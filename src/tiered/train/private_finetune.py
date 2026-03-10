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
    PYTHONPATH=./src python src/tiered/train/private_finetune.py \\
        --checkpoint /path/to/pretrained \\
        --key_path examples/key_32m.json \\
        --private_data /path/to/forget \\
        --public_data /path/to/retain \\
        --output_dir /path/to/output
"""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
import wandb
from datasets import load_from_disk
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from tqdm import tqdm

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import load_key, mask_public_gradients
from tiered.train.utils import save_checkpoint


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
                        help="Path to permutation key JSON")
    
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
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to finetuning checkpoint to resume from")
    
    # Validation
    parser.add_argument("--eval_interval", type=int, default=500,
                        help="Evaluate on validation set every N steps")
    parser.add_argument("--eval_steps", type=int, default=50,
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
    
    return parser.parse_args()


def train_step(model, ref_model, private_batch, public_batch, key, optimizer, device, kl_lambda, max_grad_norm):
    """Execute one finetuning step.
    
    Implements: L_ft = (1-λ) * L_priv(C2) + λ * R_KL(C1)
    
    CRITICAL: When switching from C1 to C2 after KL backward, we must also
    swap the gradients so they follow their corresponding weight values.
    
    Algorithm:
    1. Forward C1 on public data → R_KL → backward (if kl_lambda > 0)
    2. apply_key → switch weights to C2 positions
    3. swap_gradients → move KL gradients to follow their weights
    4. Forward C2 on private data → L_priv → backward (accumulates)
    5. mask_public_gradients (zero public grads, keep keyed grads)
    6. clip_grad_norm + optimizer.step() [WHILE IN C2]
    7. unapply_key → back to C1
    
    Returns:
        (loss_priv, loss_kl, accuracy)
    """
    from tiered.permutation import swap_gradients
    
    model.train()
    optimizer.zero_grad()
    
    # === Step 1: R_KL on C1 (public architecture) ===
    loss_kl_value = 0.0
    if kl_lambda > 0 and public_batch is not None and ref_model is not None:
        public_ids = public_batch["input_ids"].to(device)
        
        with torch.no_grad():
            ref_logits = ref_model(public_ids).logits
            ref_probs = F.softmax(ref_logits, dim=-1)
        
        current_logits = model(public_ids).logits
        current_log_probs = F.log_softmax(current_logits, dim=-1)
        loss_kl = F.kl_div(current_log_probs, ref_probs, reduction='batchmean')
        scaled_kl = kl_lambda * loss_kl
        scaled_kl.backward()
        loss_kl_value = loss_kl.item()
    
    # === Step 2-3: Switch to C2 and swap gradients to follow weights ===
    model.apply_key(key)
    if kl_lambda > 0:
        swap_gradients(model, key)  # KL gradients now at C2 positions
    
    # === Step 4: L_priv on C2 ===
    private_ids = private_batch["input_ids"].to(device)
    labels = private_batch["labels"].to(device)
    outputs_c2 = model(private_ids, labels=labels)
    loss_priv = outputs_c2.loss
    scaled_priv = (1 - kl_lambda) * loss_priv
    
    with torch.no_grad():
        preds = outputs_c2.logits[:, :-1, :].argmax(dim=-1)
        targets = labels[:, 1:]
        acc = (preds == targets).float().mean().item()
    
    scaled_priv.backward()
    
    # === Step 5: Zero public grads ===
    mask_public_gradients(model, key)
    
    # === Step 6: Optimizer step WHILE IN C2 CONFIG ===
    # With a single fixed key this is correct: Adam's momentum/variance at
    # each buffer position always tracks the same logical weight every step.
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    
    # === Step 7: Back to C1 ===
    model.unapply_key(key)
    
    return loss_priv.item(), loss_kl_value, acc


@torch.no_grad()
def evaluate_on_dataset(model, dataloader, key, device, num_steps=50, eval_c2=False):
    """Evaluate model on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for the dataset
        key: Permutation key (for C2 evaluation)
        device: Device to use
        num_steps: Number of batches to evaluate
        eval_c2: Whether to also evaluate C2 architecture
        
    Returns:
        Dict with loss, ppl, acc for C1 (and C2 if eval_c2=True)
    """
    model.eval()
    
    total_loss_c1 = 0.0
    total_acc_c1 = 0.0
    total_top3_c1 = 0.0
    total_loss_c2 = 0.0
    total_acc_c2 = 0.0
    total_top3_c2 = 0.0
    num_batches = 0
    
    data_iter = iter(dataloader)
    
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
        acc_c1 = (preds_c1 == targets).float().mean().item()
        top3_c1 = logits_c1.topk(3, dim=-1).indices
        top3_acc_c1 = (top3_c1 == targets.unsqueeze(-1)).any(dim=-1).float().mean().item()
        
        total_loss_c1 += loss_c1
        total_acc_c1 += acc_c1
        total_top3_c1 += top3_acc_c1
        
        # Evaluate C2 if requested
        if eval_c2:
            model.apply_key(key)
            outputs_c2 = model(input_ids, labels=labels)
            loss_c2 = outputs_c2.loss.item()
            logits_c2 = outputs_c2.logits[:, :-1, :]
            preds_c2 = logits_c2.argmax(dim=-1)
            acc_c2 = (preds_c2 == targets).float().mean().item()
            top3_c2 = logits_c2.topk(3, dim=-1).indices
            top3_acc_c2 = (top3_c2 == targets.unsqueeze(-1)).any(dim=-1).float().mean().item()
            model.unapply_key(key)
            
            total_loss_c2 += loss_c2
            total_acc_c2 += acc_c2
            total_top3_c2 += top3_acc_c2
        
        num_batches += 1
    
    model.train()
    
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and key
    print(f"Loading model from {args.checkpoint}")
    model = GPTNeoForCausalLMTiered.from_pretrained(args.checkpoint)
    model.to(device)
    
    key = load_key(args.key_path)
    print(f"Loaded key: {len(key.attn_heads)} attention swaps, {len(key.mlp_cols)} MLP swaps")
    
    # Create reference model for KL (frozen copy of pretrained C1)
    ref_model = None
    if args.kl_lambda > 0 and args.public_data is not None:
        print("Creating reference model for KL regularization")
        ref_model = GPTNeoForCausalLMTiered.from_pretrained(args.checkpoint)
        ref_model.to(device)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Load private/forget data (for L_priv and private validation)
    print(f"Loading private/forget data from {args.private_data}")
    private_dataset = load_from_disk(args.private_data)
    
    # Split into train/val if not already split
    if "train" in private_dataset and "test" in private_dataset:
        private_train = private_dataset["train"]
        private_val = private_dataset["test"]
    elif "train" in private_dataset:
        private_train = private_dataset["train"]
        private_val = private_dataset["train"].select(range(min(1000, len(private_dataset["train"]))))
    else:
        n_val = max(100, len(private_dataset) // 10)
        private_train = private_dataset.select(range(len(private_dataset) - n_val))
        private_val = private_dataset.select(range(len(private_dataset) - n_val, len(private_dataset)))
    
    cols_to_keep = ["input_ids", "attention_mask"]
    cols_to_remove = [c for c in private_train.column_names if c not in cols_to_keep]
    if cols_to_remove:
        private_train = private_train.remove_columns(cols_to_remove)
        private_val = private_val.remove_columns(cols_to_remove)
    
    private_loader = DataLoader(
        private_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    private_val_loader = DataLoader(
        private_val,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Load public/retain data (for R_KL and retain validation)
    public_loader = None
    retain_val_loader = None
    if args.public_data is not None:
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
            public_loader = DataLoader(
                public_train,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collator,
                drop_last=True,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        
        retain_val_loader = DataLoader(
            retain_val,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    
    # Optimizer (β₂=0.95 standard for LLM training)
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
    if args.resume_from:
        training_state_path = os.path.join(args.resume_from, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location=device)
            optimizer.load_state_dict(training_state["optimizer"])
            scheduler.load_state_dict(training_state["scheduler"])
            global_step = training_state["global_step"]
            wandb_run_id = training_state.get("wandb_run_id")
            cumulative_wall_secs = training_state.get("cumulative_wall_secs", 0.0)
            print(f"Resumed finetuning state from step {global_step}")
            if cumulative_wall_secs > 0:
                print(f"  Resumed cumulative wall time: {cumulative_wall_secs / 3600:.2f}h")
        # Load model weights from resume checkpoint
        model = GPTNeoForCausalLMTiered.from_pretrained(args.resume_from)
        model.to(device)
    
    # Wandb — resume on same graphs if we have a run ID
    if wandb_run_id:
        wandb.init(
            project=args.wandb_project,
            id=wandb_run_id,
            resume="must",
            config=vars(args),
        )
        print(f"Resumed wandb run: {wandb_run_id}")
    else:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    wandb_run_id = wandb.run.id
    wandb.define_metric("train/step")
    wandb.define_metric("*", step_metric="train/step")

    # ── Compute metrics setup ──
    num_params = count_total_parameters(model)
    # Infer context size from the model's embedding table
    context_size = model.config.max_position_embeddings

    # Tokens per step: batch_size * context_size for each data stream.
    # With KL: private batch + public batch = 2 * batch_size * context_size tokens read
    # Without KL: private batch only = batch_size * context_size tokens read
    tokens_private_per_step = args.batch_size * context_size
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

    print(f"\n── Compute metrics ──")
    print(f"  Parameters (N):        {num_params:,}")
    print(f"  Context size:          {context_size}")
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
    wandb.config.update({
        "compute/num_params": num_params,
        "compute/context_size": context_size,
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
    
    # Training loop
    private_iter = iter(private_loader)
    public_iter = iter(public_loader) if public_loader else None
    best_val_loss = float('inf')
    
    print(f"Starting finetuning for {args.max_steps} steps...")
    print(f"Objective: L_ft = (1-{args.kl_lambda})*L_priv + {args.kl_lambda}*R_KL")
    print(f"Validation every {args.eval_interval} steps")
    print(f"Tracking: C1 on retain, C1 on private, C2 on private")
    
    pbar = tqdm(total=args.max_steps, desc="Finetuning", initial=global_step)
    
    while global_step < args.max_steps:
        # ── Step timing ──
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_start = time.monotonic()

        # Get private batch
        try:
            private_batch = next(private_iter)
        except StopIteration:
            private_iter = iter(private_loader)
            private_batch = next(private_iter)
        
        # Get public batch if needed
        public_batch = None
        if public_iter is not None:
            try:
                public_batch = next(public_iter)
            except StopIteration:
                public_iter = iter(public_loader)
                public_batch = next(public_iter)
        
        loss_priv, loss_kl, acc = train_step(
            model, ref_model, private_batch, public_batch, key, 
            optimizer, device, args.kl_lambda, args.max_grad_norm
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

        pbar.update(1)
        pbar.set_postfix({"L_priv": f"{loss_priv:.3f}", "R_KL": f"{loss_kl:.3f}"})
        
        # Logging
        if global_step % args.log_interval == 0:
            ppl = math.exp(min(loss_priv, 100))
            total_loss = (1 - args.kl_lambda) * loss_priv + args.kl_lambda * loss_kl
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
                "perf/wall_since_launch_hrs": (time.time() - train_start_wall) / 3600,
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
        
        # Validation
        if global_step == 1 or global_step % args.eval_interval == 0:
            print(f"\n[Validation @ Step {global_step}]")
            
            # Evaluate C1 and C2 on private/forget data
            private_metrics = evaluate_on_dataset(
                model, private_val_loader, key, device, 
                num_steps=args.eval_steps, eval_c2=True
            )
            print(f"  Private data:")
            print(f"    C1: loss={private_metrics['loss_c1']:.4f}, ppl={private_metrics['ppl_c1']:.2f}, acc={private_metrics['acc_c1']:.4f}, top3={private_metrics['top3_acc_c1']:.4f}")
            print(f"    C2: loss={private_metrics['loss_c2']:.4f}, ppl={private_metrics['ppl_c2']:.2f}, acc={private_metrics['acc_c2']:.4f}, top3={private_metrics['top3_acc_c2']:.4f}")
            
            wandb.log({
                "Val Private/C1 Loss": private_metrics["loss_c1"],
                "Val Private/C1 Perplexity": private_metrics["ppl_c1"],
                "Val Private/C1 Accuracy": private_metrics["acc_c1"],
                "Val Private/C1 Top-3 Accuracy": private_metrics["top3_acc_c1"],
                "Val Private/C2 Loss": private_metrics["loss_c2"],
                "Val Private/C2 Perplexity": private_metrics["ppl_c2"],
                "Val Private/C2 Accuracy": private_metrics["acc_c2"],
                "Val Private/C2 Top-3 Accuracy": private_metrics["top3_acc_c2"],
                "train/step": global_step,
            })
            
            # Evaluate C1 and C2 on retain data
            if retain_val_loader is not None:
                retain_metrics = evaluate_on_dataset(
                    model, retain_val_loader, key, device,
                    num_steps=args.eval_steps, eval_c2=True
                )
                print(f"  Retain data:")
                print(f"    C1: loss={retain_metrics['loss_c1']:.4f}, ppl={retain_metrics['ppl_c1']:.2f}, acc={retain_metrics['acc_c1']:.4f}, top3={retain_metrics['top3_acc_c1']:.4f}")
                print(f"    C2: loss={retain_metrics['loss_c2']:.4f}, ppl={retain_metrics['ppl_c2']:.2f}, acc={retain_metrics['acc_c2']:.4f}, top3={retain_metrics['top3_acc_c2']:.4f}")
                
                wandb.log({
                    "Val Retain/C1 Loss": retain_metrics["loss_c1"],
                    "Val Retain/C1 Perplexity": retain_metrics["ppl_c1"],
                    "Val Retain/C1 Accuracy": retain_metrics["acc_c1"],
                    "Val Retain/C1 Top-3 Accuracy": retain_metrics["top3_acc_c1"],
                    "Val Retain/C2 Loss": retain_metrics["loss_c2"],
                    "Val Retain/C2 Perplexity": retain_metrics["ppl_c2"],
                    "Val Retain/C2 Accuracy": retain_metrics["acc_c2"],
                    "Val Retain/C2 Top-3 Accuracy": retain_metrics["top3_acc_c2"],
                    "train/step": global_step,
                })
            
            print()
            
            # Save best model based on C2 private loss
            if private_metrics["loss_c2"] < best_val_loss:
                best_val_loss = private_metrics["loss_c2"]
                save_path = os.path.join(args.output_dir, "best")
                save_checkpoint(model, tokenizer, optimizer, save_path,
                              scheduler=scheduler, global_step=global_step,
                              wandb_run_id=wandb_run_id,
                              cumulative_wall_secs=cumulative_wall_secs)
                print(f"New best model saved to {save_path}")
        
        # Save checkpoint
        if global_step % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            save_checkpoint(model, tokenizer, optimizer, save_path,
                          scheduler=scheduler, global_step=global_step,
                          wandb_run_id=wandb_run_id,
                          cumulative_wall_secs=cumulative_wall_secs)
            print(f"Saved checkpoint to {save_path}")
    
    pbar.close()
    
    # Final save
    save_path = os.path.join(args.output_dir, "final")
    save_checkpoint(model, tokenizer, optimizer, save_path,
                   scheduler=scheduler, global_step=global_step,
                   wandb_run_id=wandb_run_id,
                   cumulative_wall_secs=cumulative_wall_secs)

    total_flops = flops_per_step * global_step
    print(f"\n{'='*60}")
    print("FINETUNING COMPLETE — COMPUTE SUMMARY")
    print(f"{'='*60}")
    print(f"  Steps:                 {global_step:,}")
    print(f"  Parameters (N):        {num_params:,}")
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


if __name__ == "__main__":
    main()