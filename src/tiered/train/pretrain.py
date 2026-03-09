"""Standard LLM Pre-training Script (baseline, no tiered alignment).

Baseline causal LM training with the SAME architecture / optimizer / scheduler /
throughput accounting as tiered_pretrain, but without any permutation or tiered
updates. This is intended as a fair apples-to-apples baseline.

Training loop:
1. Forward+backward on the public model only
2. Standard optimizer step
"""

import argparse
import math
import os
import time
from contextlib import nullcontext

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

from tiered.train.utils import load_model, save_checkpoint


# ---------------------------------------------------------------------------
# Compute-metrics helpers
# ---------------------------------------------------------------------------

def count_total_parameters(model) -> int:
    """Return the exact total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model) -> int:
    """Return the exact total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops_per_token(
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    context_size: int,
    vocab_size: int,
    num_params: int,
) -> dict:
    """Estimate FLOPs for a single token in a forward pass (decoder-only transformer)."""
    L, H, I, S, V = num_layers, hidden_size, intermediate_size, context_size, vocab_size

    attn_qkv = 2 * 3 * H * H
    attn_out = 2 * H * H
    attn_score = 2 * S * H
    attn_agg = 2 * S * H
    mlp = 2 * 2 * H * I
    per_layer = attn_qkv + attn_out + attn_score + attn_agg + mlp
    all_layers = per_layer * L
    embed_lmhead = 2 * H * V

    fwd_per_token = all_layers + embed_lmhead

    return {
        "fwd_per_token": fwd_per_token,
        "fwd_bwd_per_token": fwd_per_token * 3,
        "per_layer_per_token": per_layer,
        "breakdown": {
            "attn_qkv": attn_qkv * L,
            "attn_out": attn_out * L,
            "attn_score": attn_score * L,
            "attn_agg": attn_agg * L,
            "mlp": mlp * L,
            "embed_lmhead": embed_lmhead,
        },
        "approx_6N": 6 * num_params,
    }


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
    parser = argparse.ArgumentParser(description="Standard LLM Pre-training")

    # Data
    parser.add_argument("--data_path", type=str, required=True, help="Path to tokenized dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints")

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
def evaluate(model, dataloader, device, num_steps=50, is_distributed=False):
    """Evaluate model, averaging metrics across ranks if distributed."""
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    count = 0

    data_iter = iter(dataloader)

    for _ in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids, labels=labels)

        total_loss += outputs.loss.item()

        preds = outputs.logits[:, :-1, :].argmax(dim=-1)
        targets = labels[:, 1:]
        mask = targets != -100
        acc = (preds[mask] == targets[mask]).float().mean().item() if mask.any() else 0.0
        total_acc += acc
        count += 1

    model.train()

    if count == 0:
        return {"val/loss": 0, "val/acc": 0, "val/ppl": 0}

    avg_loss = total_loss / count
    avg_acc = total_acc / count

    if is_distributed:
        metrics_tensor = torch.tensor([avg_loss, avg_acc], device=device)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.AVG)
        avg_loss, avg_acc = metrics_tensor.tolist()

    return {
        "val/loss": avg_loss,
        "val/acc": avg_acc,
        "val/ppl": math.exp(min(avg_loss, 100)),
    }


def train(args):
    """Main training function."""
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

    # Load model
    if args.checkpoint:
        model = load_model(
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            context_size=args.context_size,
            intermediate_size=args.intermediate_size,
            tie_weights=not args.untie_weights,
            do_print=is_main,
        )
        model = model.from_pretrained(args.checkpoint)
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

    raw_model = model
    model = torch.compile(model)
    if is_main:
        print("torch.compile enabled")

    if local_rank != -1:
        model = DDP(model, device_ids=[local_rank])

    # Load data
    full_dataset = load_from_disk(args.data_path)

    if "train" in full_dataset:
        train_dataset = full_dataset["train"]
    else:
        train_dataset = full_dataset

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

    # Optimizer: match tiered_pretrain
    decay_params = [p for p in raw_model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in raw_model.parameters() if p.dim() < 2]
    optimizer = optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        fused=True,
    )

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

    # Resume from checkpoint
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

    # Setup wandb
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
        wandb.define_metric("train/step")
        wandb.define_metric("*", step_metric="train/step")

    # Training loop setup
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
    num_params = count_total_parameters(raw_model)
    num_trainable = count_trainable_parameters(raw_model)
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

    tokens_per_step = effective_batch * args.context_size
    flops_per_token = 6 * num_params
    # baseline does 1 pass per step, not 2
    flops_per_step = flops_per_token * tokens_per_step

    gpu_peak_flops, gpu_name = detect_gpu_peak_flops(device)

    if is_main:
        print(f"\n── Compute metrics ──")
        print(f"  Total parameters:      {num_params:,}")
        print(f"  Trainable parameters:  {num_trainable:,}")
        print(f"  Tokens/step:           {tokens_per_step:,}")
        print(f"  FLOPs/token (6N):      {flops_per_token:.3e}")
        print(f"  FLOPs/step (est):      {flops_per_step:.3e}  (1 pass × 6N × tokens)")
        print(f"  GPU:                   {gpu_name}")
        if gpu_peak_flops > 0:
            print(f"  GPU peak bf16:         {gpu_peak_flops:.3e} FLOP/s")
        else:
            print(f"  GPU peak bf16:         unknown (MFU will be N/A)")
        print(
            f"  Detailed fwd/token:    {flop_info['fwd_per_token']:.3e}  "
            f"(vs 2N={2*num_params:.3e}, ratio={flop_info['fwd_per_token']/(2*num_params):.3f})"
        )
        print()

    cumulative_tokens = global_step * tokens_per_step
    train_start_wall = time.time()
    cumulative_wall_secs = 0.0
    if args.checkpoint:
        cumulative_wall_secs = training_state.get("cumulative_wall_secs", 0.0)
        if is_main and cumulative_wall_secs > 0:
            print(f"  Resumed cumulative wall time: {cumulative_wall_secs / 3600:.2f}h")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    if is_main:
        wandb.config.update({
            "compute/total_params": num_params,
            "compute/trainable_params": num_trainable,
            "compute/tokens_per_step": tokens_per_step,
            "compute/flops_per_step": flops_per_step,
            "compute/flops_per_token_6N": flops_per_token,
            "compute/flops_per_token_detailed": flop_info["fwd_bwd_per_token"],
            "compute/gpu_name": gpu_name,
            "compute/gpu_peak_bf16_flops": gpu_peak_flops,
            "compute/vocab_size": vocab_size,
        }, allow_val_change=True)

    # Initial validation
    if global_step == 0 and val_dataloader is not None:
        val_metrics = evaluate(raw_model, val_dataloader, device, args.eval_steps,
                               is_distributed=is_distributed)
        if is_main:
            wandb.log({**val_metrics, "train/step": 0})

    pbar = tqdm(total=args.max_steps, desc="Training", initial=global_step) if is_main else None

    while global_step < args.max_steps:
        optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_start = time.monotonic()

        total_loss = 0.0
        total_acc = 0.0
        model.train()

        micro_batches = []
        for _ in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                if local_rank != -1:
                    sampler.set_epoch(global_step)
                data_iter = iter(dataloader)
                batch = next(data_iter)

            batch["input_ids"] = batch["input_ids"].to(device, non_blocking=True)
            batch["labels"] = batch["labels"].to(device, non_blocking=True)
            micro_batches.append(batch)

        for micro_idx, batch in enumerate(micro_batches):
            is_last_micro = (micro_idx == grad_accum_steps - 1)
            sync_ctx = nullcontext() if (not is_distributed or is_last_micro) else model.no_sync()

            with sync_ctx:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(batch["input_ids"], labels=batch["labels"])
                    loss = outputs.loss

                with torch.no_grad():
                    preds = outputs.logits[:, :-1, :].argmax(dim=-1)
                    targets = batch["labels"][:, 1:]
                    mask = targets != -100
                    acc = (preds[mask] == targets[mask]).float().mean().item() if mask.any() else 0.0
                    total_acc += acc
                    total_loss += loss.item()

                (loss * loss_scale).backward()

        avg_loss = total_loss / grad_accum_steps
        avg_acc = total_acc / grad_accum_steps

        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        global_step += 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_elapsed = time.monotonic() - step_start
        cumulative_wall_secs += step_elapsed
        cumulative_tokens += tokens_per_step

        if pbar is not None:
            tps = tokens_per_step / step_elapsed if step_elapsed > 0 else 0.0
            pbar.update(1)
            pbar.set_postfix({"loss": f"{avg_loss:.3f}", "tok/s": f"{tps:,.0f}"})

        # Logging
        if is_main and global_step % args.log_interval == 0:
            ppl = math.exp(min(avg_loss, 100))
            lr = optimizer.param_groups[0]["lr"]

            tokens_per_sec = tokens_per_step / step_elapsed if step_elapsed > 0 else 0.0
            samples_per_sec = effective_batch / step_elapsed if step_elapsed > 0 else 0.0
            achieved_flops_per_sec = flops_per_step / step_elapsed if step_elapsed > 0 else 0.0
            per_gpu_flops = achieved_flops_per_sec / world_size
            mfu = per_gpu_flops / gpu_peak_flops if gpu_peak_flops > 0 else 0.0

            log_dict = {
                "loss": avg_loss,
                "acc": avg_acc,
                "ppl": ppl,
                "lr": lr,
                "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
                "train/step": global_step,
                "perf/step_time_sec": step_elapsed,
                "perf/wall_clock_hrs": cumulative_wall_secs / 3600,
                "perf/wall_since_launch_hrs": (time.time() - train_start_wall) / 3600,
                "perf/tokens_per_sec": tokens_per_sec,
                "perf/tokens_per_sec_per_gpu": tokens_per_sec / world_size,
                "perf/samples_per_sec": samples_per_sec,
                "perf/flops_per_step": flops_per_step,
                "perf/achieved_tflops": achieved_flops_per_sec / 1e12,
                "perf/achieved_tflops_per_gpu": per_gpu_flops / 1e12,
                "perf/mfu": mfu,
                "perf/cumulative_tokens": cumulative_tokens,
                "perf/cumulative_flops": flops_per_token * cumulative_tokens,
                "perf/cumulative_petaflops": (flops_per_token * cumulative_tokens) / 1e15,
            }
            log_dict.update(get_gpu_memory_stats(device))
            wandb.log(log_dict)

        # Validation
        if val_dataloader is not None and global_step % args.eval_interval == 0:
            val_metrics = evaluate(raw_model, val_dataloader, device, args.eval_steps,
                                   is_distributed=is_distributed)
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

        total_flops = flops_per_token * cumulative_tokens
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE — COMPUTE SUMMARY (baseline)")
        print(f"{'='*60}")
        print(f"  Steps:                 {global_step:,}")
        print(f"  Parameters (N):        {num_params:,}")
        print(f"  Total tokens (D):      {cumulative_tokens:,}")
        print(f"  Total FLOPs (6ND):     {total_flops:.4e}")
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
            wandb.run.summary["final/avg_mfu"] = (
                total_flops / cumulative_wall_secs / world_size
            ) / gpu_peak_flops

        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)