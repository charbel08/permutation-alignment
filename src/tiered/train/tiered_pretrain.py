"""Tiered Alignment Pretraining Script.

This script implements the joint pretraining of public (C1) and keyed (C2) 
architectures with asymmetric gradient updates as described in the paper.

Training loop:
1. Forward pass on C1, get loss l1
2. Backward pass on C1, store gradients for S (public weights)
3. Apply permutation to get C2
4. Forward pass on C2, get loss l2  
5. Backward pass on C2, get gradients for both S and S'
6. Update weights:
   - S' <- S' + lr * grad2_S' (keyed weights: only from C2)
   - S  <- S  + lr * 0.5 * (grad1_S + grad2_S) (public weights: average)
7. Unapply permutation to return to C1
"""

import argparse
import math
import os
import time
from contextlib import nullcontext
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import wandb
from tqdm import tqdm

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import load_key, PermutationKey, scale_public_gradients
from tiered.permutation.masking import mask_keyed_gradients, build_mask_plan
from tiered.permutation.permute import (
    apply_permutation, unapply_permutation, swap_gradients, build_swap_plan
)
from tiered.train.utils import load_model, save_checkpoint


# ---------------------------------------------------------------------------
# Compute-metrics helpers
# ---------------------------------------------------------------------------

def estimate_flops_per_token(num_layers: int, hidden_size: int, intermediate_size: int,
                              context_size: int, vocab_size: int,
                              num_params: int) -> dict:
    """Estimate FLOPs for a single token in a forward pass (decoder-only transformer).

    Uses the standard breakdown from Kaplan et al. / Hoffmann et al.:
      - Attention QKV projection:   2 * 3 * H^2       per layer
      - Attention output projection: 2 * H^2           per layer
      - Attention logits (Q·K^T):   2 * S * H          per layer  (data-dependent)
      - Attention weighted sum (A·V): 2 * S * H        per layer  (data-dependent)
      - MLP (up + down):            2 * 2 * H * I      per layer
      - Embedding / LM-head:        2 * H * V          (once)

    The factor of 2 in each term accounts for multiply-accumulate = 2 FLOPs.
    Backward pass ≈ 2× forward, so one full fwd+bwd ≈ 3× forward FLOPs.

    Returns a dict with per-component and total FLOPs so you can report the
    breakdown in your paper.
    """
    L, H, I, S, V = num_layers, hidden_size, intermediate_size, context_size, vocab_size

    attn_qkv   = 2 * 3 * H * H         # per layer per token
    attn_out   = 2 * H * H              # per layer per token
    attn_score = 2 * S * H              # per layer per token (avg over positions: S/2, but we use S for peak)
    attn_agg   = 2 * S * H              # per layer per token
    mlp        = 2 * 2 * H * I          # per layer per token (up-proj + down-proj)
    per_layer  = attn_qkv + attn_out + attn_score + attn_agg + mlp
    all_layers = per_layer * L
    embed_lmhead = 2 * H * V            # embedding lookup + LM head projection

    fwd_per_token = all_layers + embed_lmhead

    return {
        "fwd_per_token": fwd_per_token,
        "fwd_bwd_per_token": fwd_per_token * 3,       # fwd + bwd ≈ 3× fwd
        "per_layer_per_token": per_layer,
        "breakdown": {
            "attn_qkv": attn_qkv * L,
            "attn_out": attn_out * L,
            "attn_score": attn_score * L,
            "attn_agg": attn_agg * L,
            "mlp": mlp * L,
            "embed_lmhead": embed_lmhead,
        },
        # Simpler Chinchilla-style approximation for cross-check: 6 * N
        "approx_6N": 6 * num_params,
    }


# Peak bf16 FLOP/s for common GPUs. Used for MFU calculation.
# Values are *tensor-core* peak, not boost-clock peak.
_GPU_PEAK_TFLOPS_BF16 = {
    "A100":  312e12,
    "A10G":   70e12,    # g5 instances
    "H100":  990e12,    # SXM variant
    "H200": 990e12,     # same SM as H100
    "L4":   121e12,
    "L40":  181e12,
    "L40S": 366e12,
    "4090": 330e12,
    "3090": 142e12,
    "4080": 203e12,
}


def detect_gpu_peak_flops(device: torch.device) -> tuple[float, str]:
    """Return (peak_flops_bf16, gpu_name) for the current GPU.

    Tries to match the GPU name against known specs. Falls back to 0
    (MFU will be reported as N/A).
    """
    if not torch.cuda.is_available():
        return 0.0, "cpu"
    name = torch.cuda.get_device_name(device)
    for key, peak in _GPU_PEAK_TFLOPS_BF16.items():
        if key.lower() in name.lower().replace(" ", ""):
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
    parser = argparse.ArgumentParser(description="Tiered Alignment Pretraining")
    
    # Data
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to tokenized dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for checkpoints")
    
    # Model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--context_size", type=int, default=1024)
    parser.add_argument("--intermediate_size", type=int, default=None,
                        help="MLP hidden dimension (defaults to 4x hidden_size)")
    parser.add_argument("--untie_weights", action="store_true",
                        help="Disable weight tying between embeddings and LM head")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # Permutation key
    parser.add_argument("--key_path", type=str, required=True,
                        help="Path to JSON permutation key file")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Number of micro-batches to accumulate before optimizer step")
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5,
                        help="Minimum LR for cosine schedule (default: 10%% of peak)")
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=500,
                        help="Validation interval in steps")
    parser.add_argument("--eval_steps", type=int, default=50,
                        help="Number of batches for validation")
    parser.add_argument("--wandb_project", type=str, default="tiered-alignment")
    parser.add_argument("--run_name", type=str, default=None)
    
    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1)
    
    return parser.parse_args()


@torch.inference_mode()
def evaluate(model, dataloader, key, device, num_steps=50, is_distributed=False,
             swap_plan=None):
    """Evaluate model on a dataset, computing C1 and C2 metrics.
    
    When distributed, each rank evaluates its shard and metrics are
    averaged across all ranks via all_reduce.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for validation data
        key: Permutation key
        device: Device to use
        num_steps: Number of batches to evaluate (per rank)
        is_distributed: Whether to all_reduce metrics across ranks
        
    Returns:
        dict: Dictionary with loss and accuracy for C1 and C2
    """
    model.eval()
    
    total_loss_c1 = 0.0
    total_loss_c2 = 0.0
    total_acc_c1 = 0.0
    total_acc_c2 = 0.0
    count = 0
    
    data_iter = iter(dataloader)
    
    for _ in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            break
            
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Evaluate C1
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs_c1 = model(input_ids, labels=labels)
        loss_c1 = outputs_c1.loss.item()
        preds_c1 = outputs_c1.logits[:, :-1, :].argmax(dim=-1)
        targets_c1 = labels[:, 1:]
        mask_c1 = targets_c1 != -100
        acc_c1 = (preds_c1[mask_c1] == targets_c1[mask_c1]).float().mean().item() if mask_c1.any() else 0.0
        
        # Evaluate C2
        apply_permutation(model, key, plan=swap_plan)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs_c2 = model(input_ids, labels=labels)
        loss_c2 = outputs_c2.loss.item()
        preds_c2 = outputs_c2.logits[:, :-1, :].argmax(dim=-1)
        targets_c2 = labels[:, 1:]
        mask_c2 = targets_c2 != -100
        acc_c2 = (preds_c2[mask_c2] == targets_c2[mask_c2]).float().mean().item() if mask_c2.any() else 0.0
        unapply_permutation(model, key, plan=swap_plan)
        
        total_loss_c1 += loss_c1
        total_loss_c2 += loss_c2
        total_acc_c1 += acc_c1
        total_acc_c2 += acc_c2
        count += 1
    
    model.train()
    
    if count == 0:
        return {"val/loss_c1": 0, "val/loss_c2": 0, "val/acc_c1": 0, "val/acc_c2": 0,
                "val/ppl_c1": 0, "val/ppl_c2": 0}
    
    avg_loss_c1 = total_loss_c1 / count
    avg_loss_c2 = total_loss_c2 / count
    avg_acc_c1 = total_acc_c1 / count
    avg_acc_c2 = total_acc_c2 / count
    
    # All-reduce across ranks to get global average
    if is_distributed:
        metrics_tensor = torch.tensor(
            [avg_loss_c1, avg_loss_c2, avg_acc_c1, avg_acc_c2], device=device
        )
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.AVG)
        avg_loss_c1, avg_loss_c2, avg_acc_c1, avg_acc_c2 = metrics_tensor.tolist()
    
    return {
        "val/loss_c1": avg_loss_c1,
        "val/loss_c2": avg_loss_c2,
        "val/acc_c1": avg_acc_c1,
        "val/acc_c2": avg_acc_c2,
        "val/ppl_c1": math.exp(min(avg_loss_c1, 100)),
        "val/ppl_c2": math.exp(min(avg_loss_c2, 100)),
    }


def train(args):
    """Main training loop."""
    # Setup distributed
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
    
    # Performance optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Load key
    key = load_key(args.key_path)
    if is_main:
        print(f"Loaded key with {len(key.attn_heads)} attention swaps, "
              f"{len(key.mlp_cols)} MLP swaps")
    
    # Load model
    if args.checkpoint:
        model = GPTNeoForCausalLMTiered.from_pretrained(args.checkpoint)
    else:
        model = load_model(
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            context_size=args.context_size,
            intermediate_size=args.intermediate_size,
            tie_weights=not args.untie_weights,
            do_print=is_main,
        )
    
    model.to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    # Pre-build swap and mask plans BEFORE torch.compile
    # (needs direct attribute access to model internals)
    swap_plan = build_swap_plan(model, key, device)
    if is_main:
        print(f"Built swap plan: {len(swap_plan.attn_ops)} attn ops, "
              f"{len(swap_plan.mlp_ops)} MLP ops (indices on {device})")
    
    mask_plan = build_mask_plan(model, key, device)
    if is_main:
        n_attn_layers = len(mask_plan.keyed_attn_indices)
        n_mlp_layers = len(mask_plan.keyed_mlp_indices)
        print(f"Built mask plan: {n_attn_layers} attn layers, "
              f"{n_mlp_layers} MLP layers with keyed indices")
    
    # Keep a reference to the original model for permutation/masking ops.
    # torch.compile wraps the model but shares the same parameter tensors,
    # so mutations via raw_model affect the compiled model's forward.
    raw_model = model
    
    # Compile the forward pass for fused kernels.
    # Safe with the restructured training loop: within each C1/C2 phase,
    # weights are stable (no permutation), so the compiled graph is reused
    # across all micro-steps. Between phases, apply_permutation changes
    # weight VALUES but not tensor metadata (shape/dtype/device), so
    # torch.compile guards remain valid — no recompilation.
    model = torch.compile(model)
    if is_main:
        print("torch.compile enabled")
    
    if local_rank != -1:
        model = DDP(model, device_ids=[local_rank])
    
    # Load data
    full_dataset = load_from_disk(args.data_path)
    
    # Get train split
    if "train" in full_dataset:
        train_dataset = full_dataset["train"]
    else:
        train_dataset = full_dataset
    
    # Get validation split from 'test' if available
    val_dataset = None
    if "test" in full_dataset:
        val_dataset = full_dataset["test"]
        if is_main:
            print(f"Loaded validation split with {len(val_dataset)} samples")
    
    cols_to_keep = ["input_ids", "attention_mask"]
    cols_to_remove = [c for c in train_dataset.column_names if c not in cols_to_keep]
    if cols_to_remove:
        train_dataset = train_dataset.remove_columns(cols_to_remove)
        if val_dataset is not None:
            val_dataset = val_dataset.remove_columns(cols_to_remove)
        if is_main:
            print(f"Removed columns: {cols_to_remove}")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    if local_rank != -1:
        sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        sampler = None
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collator,
        drop_last=True,
        num_workers=6,
        pin_memory=True,
    )
    
    # Setup validation dataloader from 'test' split (with DistributedSampler for parallel eval)
    val_dataloader = None
    if val_dataset is not None:
        if local_rank != -1:
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
        else:
            val_sampler = None
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            shuffle=False,
            collate_fn=collator,
            drop_last=True,
            num_workers=6,
            pin_memory=True,
        )
    
    # Setup optimizer: weight decay on 2D+ params only (exclude biases, LayerNorm)
    decay_params = [p for p in raw_model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in raw_model.parameters() if p.dim() < 2]
    optimizer = optim.AdamW([
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.learning_rate, betas=(0.9, 0.95), fused=True)
    
    # LR schedule: linear warmup + cosine decay
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=args.max_steps - args.warmup_steps, eta_min=args.min_lr
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_steps]
    )
    
    # Resume from checkpoint if provided
    global_step = 0
    wandb_run_id = None
    if args.checkpoint:
        training_state_path = os.path.join(args.checkpoint, "training_state.pt")
        if not os.path.exists(training_state_path):
            raise FileNotFoundError(
                f"Checkpoint dir exists but training_state.pt not found: {training_state_path}\n"
                f"Cannot resume training without optimizer/scheduler state."
            )
        training_state = torch.load(training_state_path, map_location=device)
        optimizer.load_state_dict(training_state["optimizer"])
        scheduler.load_state_dict(training_state["scheduler"])
        global_step = training_state["global_step"]
        wandb_run_id = training_state.get("wandb_run_id")
        if is_main:
            print(f"Resumed training state from step {global_step}")
    
    # Setup wandb — resume on same graphs if we have a run ID
    if is_main:
        if args.checkpoint and wandb_run_id:
            wandb.init(
                project=args.wandb_project,
                id=wandb_run_id,
                resume="must",
                config=vars(args),
            )
            print(f"Resumed wandb run: {wandb_run_id}")
        else:
            wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config=vars(args),
            )
        wandb_run_id = wandb.run.id
        # Use a namespaced step to avoid conflict with wandb's internal _step
        wandb.define_metric("train/step")
        wandb.define_metric("*", step_metric="train/step")
    
    # Training loop
    # On resume, set the sampler epoch so we don't replay the same data order.
    # Using global_step as a proxy for epoch (exact epoch tracking isn't needed
    # for pretraining — we just need a different shuffle than epoch 0).
    if local_rank != -1 and global_step > 0:
        sampler.set_epoch(global_step)
    data_iter = iter(dataloader)
    
    is_distributed = (local_rank != -1)
    grad_accum_steps = args.grad_accum_steps
    loss_scale = 1.0 / grad_accum_steps
    effective_batch = args.batch_size * grad_accum_steps * world_size
    
    if is_main:
        print(f"Effective batch size: {args.batch_size} x {grad_accum_steps} x {world_size} = {effective_batch}")
    
    # ── Compute metrics setup ──
    num_params = sum(p.numel() for p in raw_model.parameters())
    num_trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    inter_size = args.intermediate_size or (args.hidden_size * 4)
    vocab_size = raw_model.get_input_embeddings().weight.shape[0]

    flop_info = estimate_flops_per_token(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        intermediate_size=inter_size,
        context_size=args.context_size,
        vocab_size=vocab_size,
        num_params=num_params,
    )
    # Tokens processed per optimizer step (across all ranks)
    tokens_per_step = effective_batch * args.context_size
    # Standard C = 6ND (Kaplan et al., Hoffmann et al., used by LLaMA 3, PaLM, etc.)
    # 6N = 2N (fwd, multiply-accumulate) + 4N (bwd, 2× fwd) per token per pass
    flops_per_token = 6 * num_params
    # We do 2 fwd+bwd per step (C1 + C2)
    flops_per_step = 2 * flops_per_token * tokens_per_step

    gpu_peak_flops, gpu_name = detect_gpu_peak_flops(device)

    if is_main:
        print(f"\n── Compute metrics ──")
        print(f"  Parameters:        {num_params:,} total, {num_trainable:,} trainable")
        print(f"  Tokens/step:       {tokens_per_step:,}")
        print(f"  FLOPs/token (6N):  {flops_per_token:.3e}")
        print(f"  FLOPs/step (est):  {flops_per_step:.3e}  (2 passes × 6N × tokens)")
        print(f"  GPU:               {gpu_name}")
        if gpu_peak_flops > 0:
            print(f"  GPU peak bf16:     {gpu_peak_flops:.3e} FLOP/s")
        else:
            print(f"  GPU peak bf16:     unknown (MFU will be N/A)")
        # Detailed breakdown for supplementary material
        print(f"  Detailed fwd/token:{flop_info['fwd_per_token']:.3e}  "
              f"(vs 2N={2*num_params:.3e}, ratio={flop_info['fwd_per_token']/(2*num_params):.3f})")
        print()

    # Cumulative trackers (restored on resume)
    cumulative_tokens = global_step * tokens_per_step
    train_start_wall = time.time()
    cumulative_wall_secs = 0.0
    if args.checkpoint:
        cumulative_wall_secs = training_state.get("cumulative_wall_secs", 0.0)
        if is_main and cumulative_wall_secs > 0:
            print(f"  Resumed cumulative wall time: {cumulative_wall_secs / 3600:.2f}h")

    # Reset peak memory stats so we track from this point
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # Log static compute info to wandb config
    if is_main:
        wandb.config.update({
            "compute/num_params": num_params,
            "compute/num_trainable_params": num_trainable,
            "compute/tokens_per_step": tokens_per_step,
            "compute/flops_per_step": flops_per_step,
            "compute/flops_per_token_6N": flops_per_token,
            "compute/flops_per_token_detailed": flop_info["fwd_bwd_per_token"],
            "compute/gpu_name": gpu_name,
            "compute/gpu_peak_bf16_flops": gpu_peak_flops,
            "compute/vocab_size": vocab_size,
        }, allow_val_change=True)

    # Initial validation at step 0 (only if not resuming)
    if global_step == 0 and val_dataloader is not None:
        val_metrics = evaluate(raw_model, val_dataloader, key, device, args.eval_steps, 
                              is_distributed=is_distributed, swap_plan=swap_plan)
        if is_main:
            wandb.log({**val_metrics, "train/step": 0})
    
    pbar = tqdm(total=args.max_steps, desc="Training", initial=global_step) if is_main else None
    
    while global_step < args.max_steps:
        optimizer.zero_grad()
        
        # ── Step timing (CUDA-synced for accuracy) ──
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_start = time.monotonic()
        
        total_loss_c1 = 0.0
        total_loss_c2 = 0.0
        total_acc_c1 = 0.0
        total_acc_c2 = 0.0
        
        model.train()
        
        # Buffer the micro-batches for this block so we use the exact same data
        # for both the C1 passes and the C2 passes.
        micro_batches = []
        for _ in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                if local_rank != -1:
                    sampler.set_epoch(global_step)
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            # Pre-load to device for speed
            batch["input_ids"] = batch["input_ids"].to(device, non_blocking=True)
            batch["labels"] = batch["labels"].to(device, non_blocking=True)
            micro_batches.append(batch)
            
        # ==================== STEP 1: PUBLIC ARCHITECTURE (C1) ====================
        for micro_idx, batch in enumerate(micro_batches):
            is_last_micro = (micro_idx == grad_accum_steps - 1)
            sync_ctx = nullcontext() if (not is_distributed or is_last_micro) else model.no_sync()
            
            with sync_ctx:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs_c1 = model(batch["input_ids"], labels=batch["labels"])
                    loss_c1 = outputs_c1.loss
                
                with torch.no_grad():
                    preds_c1 = outputs_c1.logits[:, :-1, :].argmax(dim=-1)
                    targets_c1 = batch["labels"][:, 1:]
                    mask_c1 = targets_c1 != -100
                    acc_c1 = (preds_c1[mask_c1] == targets_c1[mask_c1]).float().mean().item() if mask_c1.any() else 0.0
                    total_acc_c1 += acc_c1
                    total_loss_c1 += loss_c1.item()
                
                (loss_c1 * loss_scale).backward()
                
        # Zero out C1 gradients on keyed weights
        mask_keyed_gradients(raw_model, key, plan=mask_plan)
        
        # ==================== STEP 2: APPLY PERMUTATION (C1 -> C2) ====================
        apply_permutation(raw_model, key, plan=swap_plan)
        
        # ==================== STEP 3: KEYED ARCHITECTURE (C2) ====================
        for micro_idx, batch in enumerate(micro_batches):
            is_last_micro = (micro_idx == grad_accum_steps - 1)
            sync_ctx = nullcontext() if (not is_distributed or is_last_micro) else model.no_sync()
            
            with sync_ctx:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs_c2 = model(batch["input_ids"], labels=batch["labels"])
                    loss_c2 = outputs_c2.loss
                
                with torch.no_grad():
                    preds_c2 = outputs_c2.logits[:, :-1, :].argmax(dim=-1)
                    targets_c2 = batch["labels"][:, 1:]
                    mask_c2 = targets_c2 != -100
                    acc_c2 = (preds_c2[mask_c2] == targets_c2[mask_c2]).float().mean().item() if mask_c2.any() else 0.0
                    total_acc_c2 += acc_c2
                    total_loss_c2 += loss_c2.item()
                
                (loss_c2 * loss_scale).backward()
        
        # Average metrics over micro-steps (for logging)
        avg_loss_c1 = total_loss_c1 / grad_accum_steps
        avg_loss_c2 = total_loss_c2 / grad_accum_steps
        avg_acc_c1 = total_acc_c1 / grad_accum_steps
        avg_acc_c2 = total_acc_c2 / grad_accum_steps
        
        # Scale public gradients averaging C1+C2.
        scale_public_gradients(raw_model, key, scale=0.5, plan=mask_plan)
        
        # Clip and step
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        
        # ==================== STEP 4: UNAPPLY PERMUTATION (C2 -> C1) ====================
        unapply_permutation(raw_model, key, plan=swap_plan)
        global_step += 1
        
        # ── Step timing ──
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_elapsed = time.monotonic() - step_start
        cumulative_wall_secs += step_elapsed
        cumulative_tokens += tokens_per_step
        
        if pbar is not None:
            tps = tokens_per_step / step_elapsed if step_elapsed > 0 else 0
            pbar.update(1)
            pbar.set_postfix({
                "loss_c1": f"{avg_loss_c1:.3f}",
                "loss_c2": f"{avg_loss_c2:.3f}",
                "tok/s": f"{tps:,.0f}",
            })
        
        # Logging
        if is_main and global_step % args.log_interval == 0:
            ppl_c1 = math.exp(min(avg_loss_c1, 100))
            ppl_c2 = math.exp(min(avg_loss_c2, 100))
            lr = optimizer.param_groups[0]["lr"]
            
            # ── Throughput metrics ──
            tokens_per_sec = tokens_per_step / step_elapsed if step_elapsed > 0 else 0
            samples_per_sec = effective_batch / step_elapsed if step_elapsed > 0 else 0
            achieved_flops_per_sec = flops_per_step / step_elapsed if step_elapsed > 0 else 0
            per_gpu_flops = achieved_flops_per_sec / world_size
            mfu = per_gpu_flops / gpu_peak_flops if gpu_peak_flops > 0 else 0.0
            
            log_dict = {
                # ── Task losses ──
                "loss_c1": avg_loss_c1,
                "loss_c2": avg_loss_c2,
                "loss_avg": (avg_loss_c1 + avg_loss_c2) / 2,
                "acc_c1": avg_acc_c1,
                "acc_c2": avg_acc_c2,
                "acc_avg": (avg_acc_c1 + avg_acc_c2) / 2,
                "ppl_c1": ppl_c1,
                "ppl_c2": ppl_c2,
                "lr": lr,
                "grad_norm": grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
                "train/step": global_step,
                # ── Timing ──
                "perf/step_time_sec": step_elapsed,
                "perf/wall_clock_hrs": cumulative_wall_secs / 3600,
                "perf/wall_since_launch_hrs": (time.time() - train_start_wall) / 3600,
                # ── Throughput ──
                "perf/tokens_per_sec": tokens_per_sec,
                "perf/tokens_per_sec_per_gpu": tokens_per_sec / world_size,
                "perf/samples_per_sec": samples_per_sec,
                # ── FLOPs ──
                "perf/flops_per_step": flops_per_step,
                "perf/achieved_tflops": achieved_flops_per_sec / 1e12,
                "perf/achieved_tflops_per_gpu": per_gpu_flops / 1e12,
                "perf/mfu": mfu,
                # ── Cumulative (for paper: total compute budget, C = 6ND) ──
                "perf/cumulative_tokens": cumulative_tokens,
                "perf/cumulative_flops": 2 * flops_per_token * cumulative_tokens,
                "perf/cumulative_petaflops": (2 * flops_per_token * cumulative_tokens) / 1e15,
            }
            # GPU memory
            log_dict.update(get_gpu_memory_stats(device))
            wandb.log(log_dict)
        
        # Validation (all ranks participate, only rank 0 logs)
        if val_dataloader is not None and global_step % args.eval_interval == 0:
            val_metrics = evaluate(raw_model, val_dataloader, key, device, args.eval_steps,
                                 is_distributed=is_distributed, swap_plan=swap_plan)
            if is_main:
                wandb.log({**val_metrics, "train/step": global_step})
        
        # Save checkpoint
        if is_main and global_step % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            save_checkpoint(raw_model, tokenizer, optimizer, save_path,
                          scheduler=scheduler, global_step=global_step,
                          wandb_run_id=wandb_run_id,
                          cumulative_wall_secs=cumulative_wall_secs)
            print(f"Saved checkpoint to {save_path}")
    
    if pbar is not None:
        pbar.close()
    
    # Final save
    if is_main:
        save_path = os.path.join(args.output_dir, "final-checkpoint")
        save_checkpoint(raw_model, tokenizer, optimizer, save_path,
                       scheduler=scheduler, global_step=global_step,
                       wandb_run_id=wandb_run_id,
                       cumulative_wall_secs=cumulative_wall_secs)
        
        total_flops = 2 * flops_per_token * cumulative_tokens
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE — COMPUTE SUMMARY (for paper)")
        print(f"{'='*60}")
        print(f"  Steps:                 {global_step:,}")
        print(f"  Parameters (N):        {num_params:,}")
        print(f"  Total tokens (D):      {cumulative_tokens:,}")
        print(f"  Total FLOPs (2×6ND):   {total_flops:.4e}")
        print(f"  Total PetaFLOPs:       {total_flops / 1e15:.2f}")
        print(f"  Wall clock (train):    {cumulative_wall_secs / 3600:.2f} hours")
        print(f"  Wall clock (total):    {(time.time() - train_start_wall) / 3600:.2f} hours")
        print(f"  Avg tokens/sec:        {cumulative_tokens / cumulative_wall_secs:,.0f}")
        print(f"  Avg tokens/sec/GPU:    {cumulative_tokens / cumulative_wall_secs / world_size:,.0f}")
        if gpu_peak_flops > 0:
            avg_mfu = (total_flops / cumulative_wall_secs / world_size) / gpu_peak_flops
            print(f"  Avg MFU:               {avg_mfu:.2%}")
        print(f"  GPU:                   {gpu_name} x {world_size}")
        print(f"  Checkpoint:            {save_path}")
        print(f"{'='*60}\n")

        # Log final summary to wandb for easy access
        wandb.run.summary.update({
            "final/total_steps": global_step,
            "final/total_tokens": cumulative_tokens,
            "final/total_flops": total_flops,
            "final/total_petaflops": total_flops / 1e15,
            "final/wall_clock_hours": cumulative_wall_secs / 3600,
            "final/avg_tokens_per_sec": cumulative_tokens / cumulative_wall_secs,
            "final/avg_tokens_per_sec_per_gpu": cumulative_tokens / cumulative_wall_secs / world_size,
            "final/num_params": num_params,
            "final/gpu_name": gpu_name,
            "final/num_gpus": world_size,
        })
        if gpu_peak_flops > 0:
            wandb.run.summary["final/avg_mfu"] = (total_flops / cumulative_wall_secs / world_size) / gpu_peak_flops

        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)