"""Plain language-model finetuning (no keys, no tiering).

Fair-compute baseline counterpart to `private_finetune.py`:
  - Load weights from `--checkpoint` (fresh optimizer + cosine schedule)
  - Standard causal-LM next-token training
  - bf16 autocast, torch.compile, DDP

Logs the same W&B metric names as `private_finetune.py` so the two runs
overlay cleanly. Because the baseline has no tier, the "C1" and "C2"
evaluations are numerically identical — both are logged so downstream
plotting code that reads `Val Private/C1 Loss` and `Val Private/C2 Loss`
works without changes.

Intended for answering "how fast does the non-tiered baseline learn the
private distribution" against the tiered private finetune, given identical
compute budgets.
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

from tiered.model import GPTNeoForCausalLMTiered
from tiered.train.utils import save_checkpoint


_GPU_PEAK_TFLOPS_BF16 = {
    "A100": 312e12, "A10G": 70e12, "H100": 990e12, "H200": 990e12,
    "L4": 121e12, "L40": 181e12, "L40S": 366e12,
    "4090": 330e12, "3090": 142e12, "4080": 203e12,
}


def _detect_gpu_peak_flops(device):
    if not torch.cuda.is_available():
        return 0.0, "cpu"
    name = torch.cuda.get_device_name(device)
    for key, peak in _GPU_PEAK_TFLOPS_BF16.items():
        if key.lower() in name.lower().replace(" ", ""):
            return peak, name
    return 0.0, name


def _gpu_mem_stats(device):
    if not torch.cuda.is_available():
        return {}
    return {
        "gpu/mem_allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
        "gpu/mem_reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
        "gpu/mem_peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "gpu/mem_peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Plain LM finetuning (no keys).")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to pretrained checkpoint (weights only; optimizer state ignored).")
    p.add_argument("--private_data", type=str, required=True,
                   help="Tokenized private dataset (HF save_to_disk layout).")
    p.add_argument("--public_data", type=str, default=None,
                   help="Optional tokenized retain/public dataset for eval parity with private_finetune.")
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=2000)

    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--wandb_project", type=str, default="finetune-sweep")
    p.add_argument("--run_name", type=str, default=None)

    p.add_argument("--local_rank", type=int, default=-1)
    return p.parse_args()


def _lm_loss_and_acc(out, labels):
    """Causal LM loss is already in `out.loss`; compute next-token accuracy."""
    with torch.no_grad():
        preds = out.logits[:, :-1, :].argmax(dim=-1)
        targets = labels[:, 1:]
        mask = targets != -100
        if mask.any():
            acc = (preds[mask] == targets[mask]).float().mean().item()
        else:
            acc = 0.0
    return out.loss, acc


@torch.inference_mode()
def _evaluate(model, loader, device, max_batches, is_distributed):
    """Return (loss, ppl, acc) averaged across all ranks."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    data_iter = iter(loader)
    for _ in range(max_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids, labels=labels)
        loss_val = out.loss.item()
        preds = out.logits[:, :-1, :].argmax(dim=-1)
        targets = labels[:, 1:]
        mask = targets != -100
        acc = (preds[mask] == targets[mask]).float().mean().item() if mask.any() else 0.0
        total_loss += loss_val
        total_acc += acc
        count += 1
    model.train()
    if count == 0:
        return {"loss": float("nan"), "ppl": float("nan"), "acc": float("nan")}
    vals = torch.tensor([total_loss / count, total_acc / count], device=device)
    if is_distributed:
        dist.all_reduce(vals, op=dist.ReduceOp.AVG)
    loss_m, acc_m = vals.tolist()
    return {"loss": loss_m, "ppl": math.exp(min(loss_m, 100)), "acc": acc_m}


def train():
    args = parse_args()

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
        print(f"Loading weights from {args.checkpoint}")
    model = GPTNeoForCausalLMTiered.from_pretrained(args.checkpoint)
    model.to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    raw_model = model
    model = torch.compile(model)
    if is_main:
        print("torch.compile enabled")
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    # Data
    private_ds = load_from_disk(args.private_data)
    private_train = private_ds["train"] if "train" in private_ds else private_ds
    private_val = private_ds.get("test") if hasattr(private_ds, "get") else None

    public_val = None
    if args.public_data:
        public_ds = load_from_disk(args.public_data)
        if "test" in public_ds:
            public_val = public_ds["test"]
        elif "train" in public_ds:
            # Mirror private_finetune's behavior: use a 1k-slice of train as retain val
            public_val = public_ds["train"].select(range(min(1000, len(public_ds["train"]))))
        else:
            public_val = public_ds.select(range(min(1000, len(public_ds))))

    cols_to_keep = {"input_ids", "attention_mask"}
    def _drop(ds):
        if ds is None:
            return None
        cols = [c for c in ds.column_names if c not in cols_to_keep]
        return ds.remove_columns(cols) if cols else ds
    private_train = _drop(private_train)
    private_val = _drop(private_val)
    public_val = _drop(public_val)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    sampler = DistributedSampler(private_train, shuffle=True) if is_distributed else None
    train_loader = DataLoader(
        private_train, batch_size=args.batch_size,
        sampler=sampler, shuffle=(sampler is None),
        collate_fn=collator, drop_last=True,
        num_workers=args.num_workers, pin_memory=True,
    )

    def _eval_loader(ds):
        if ds is None:
            return None
        s = DistributedSampler(ds, shuffle=False) if is_distributed else None
        return DataLoader(
            ds, batch_size=args.batch_size,
            sampler=s, shuffle=False,
            collate_fn=collator, drop_last=True,
            num_workers=args.num_workers, pin_memory=True,
        )

    private_val_loader = _eval_loader(private_val)
    public_val_loader = _eval_loader(public_val)

    # Optimizer / scheduler
    decay_params = [p for p in raw_model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in raw_model.parameters() if p.dim() < 2]
    optimizer = optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate, betas=(0.9, 0.95), fused=True,
    )
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=args.max_steps - args.warmup_steps, eta_min=args.min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_steps])

    if is_main:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        wandb.define_metric("train/step")
        wandb.define_metric("*", step_metric="train/step")
    wandb_run_id = wandb.run.id if is_main else None

    num_params = sum(p.numel() for p in raw_model.parameters())
    context_size = raw_model.config.max_position_embeddings
    tokens_per_step = args.batch_size * args.grad_accum_steps * world_size * context_size
    flops_per_step = 6 * num_params * tokens_per_step
    gpu_peak_flops, gpu_name = _detect_gpu_peak_flops(device)
    if is_main:
        print(f"Params: {num_params:,}  tokens/step: {tokens_per_step:,}  max_steps: {args.max_steps}")
        print(f"GPU: {gpu_name}  peak bf16 FLOPs: {gpu_peak_flops:.2e}")

    pbar = tqdm(total=args.max_steps, desc="Baseline finetune") if is_main else None
    data_iter = iter(train_loader)
    data_epoch = 0
    if is_distributed:
        sampler.set_epoch(data_epoch)

    cumulative_tokens = 0
    cumulative_wall_secs = 0.0
    global_step = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    def run_validation(step_for_logging: int):
        """Eval C1/C2 on private + optional retain; mirrors private_finetune.

        Baseline has no tier, so the C1 and C2 lines are identical. Both are
        logged for metric-name parity.
        """
        val_log = {"train/step": step_for_logging}
        if private_val_loader is not None:
            m = _evaluate(model, private_val_loader, device,
                          args.eval_steps, is_distributed)
            for tier in ("C1", "C2"):
                val_log[f"Val Private/{tier} Loss"] = m["loss"]
                val_log[f"Val Private/{tier} Perplexity"] = m["ppl"]
                val_log[f"Val Private/{tier} Accuracy"] = m["acc"]
            if is_main:
                print(f"\n[Val @ step {step_for_logging}] Private: "
                      f"loss={m['loss']:.4f}  ppl={m['ppl']:.2f}  acc={m['acc']:.4f}")
        if public_val_loader is not None:
            m = _evaluate(model, public_val_loader, device,
                          args.eval_steps, is_distributed)
            for tier in ("C1", "C2"):
                val_log[f"Val Retain/{tier} Loss"] = m["loss"]
                val_log[f"Val Retain/{tier} Perplexity"] = m["ppl"]
                val_log[f"Val Retain/{tier} Accuracy"] = m["acc"]
            if is_main:
                print(f"  Retain:  loss={m['loss']:.4f}  ppl={m['ppl']:.2f}  acc={m['acc']:.4f}")
        if is_main:
            wandb.log(val_log)

    # Optional: baseline validation before any training step
    if private_val_loader is not None and global_step == 0:
        run_validation(0)

    while global_step < args.max_steps:
        optimizer.zero_grad()
        model.train()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_start = time.monotonic()

        total_loss = 0.0
        total_acc = 0.0
        for micro_idx in range(args.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_epoch += 1
                if is_distributed:
                    sampler.set_epoch(data_epoch)
                data_iter = iter(train_loader)
                batch = next(data_iter)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            is_last = (micro_idx == args.grad_accum_steps - 1)
            sync_ctx = nullcontext() if (not is_distributed or is_last) else model.no_sync()
            with sync_ctx:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(input_ids, labels=labels)
                loss, acc = _lm_loss_and_acc(out, labels)
                (loss / args.grad_accum_steps).backward()
            total_loss += loss.item()
            total_acc += acc

        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        global_step += 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_elapsed = time.monotonic() - step_start
        cumulative_wall_secs += step_elapsed
        cumulative_tokens += tokens_per_step

        avg_loss = total_loss / args.grad_accum_steps
        avg_acc = total_acc / args.grad_accum_steps
        ppl = math.exp(min(avg_loss, 100))

        if pbar is not None:
            tps = tokens_per_step / step_elapsed if step_elapsed > 0 else 0
            pbar.update(1)
            pbar.set_postfix({"L_priv": f"{avg_loss:.3f}", "tok/s": f"{tps:,.0f}"})

        if is_main and global_step % args.log_interval == 0:
            lr = optimizer.param_groups[0]["lr"]
            tokens_per_sec = tokens_per_step / step_elapsed if step_elapsed > 0 else 0.0
            achieved_flops_per_sec = flops_per_step / step_elapsed if step_elapsed > 0 else 0.0
            mfu = (achieved_flops_per_sec / world_size) / gpu_peak_flops if gpu_peak_flops > 0 else 0.0

            log_dict = {
                "Train/Total Loss": avg_loss,          # no KL → total == private
                "Train/Private Loss (C2)": avg_loss,
                "Train/Perplexity (C2)": ppl,
                "Train/Accuracy (C2)": avg_acc,
                "Train/LR": lr,
                "train/step": global_step,
                "perf/step_time_sec": step_elapsed,
                "perf/wall_clock_hrs": cumulative_wall_secs / 3600,
                "perf/wall_since_launch_hrs": cumulative_wall_secs / 3600,
                "perf/tokens_per_sec": tokens_per_sec,
                "perf/flops_per_step": flops_per_step,
                "perf/achieved_tflops": achieved_flops_per_sec / 1e12,
                "perf/mfu": mfu,
                "perf/cumulative_tokens": cumulative_tokens,
                "perf/cumulative_flops": flops_per_step * global_step,
                "perf/cumulative_petaflops": (flops_per_step * global_step) / 1e15,
            }
            log_dict.update(_gpu_mem_stats(device))
            wandb.log(log_dict)

        if global_step % args.eval_interval == 0:
            run_validation(global_step)

        if is_main and global_step % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            save_checkpoint(
                raw_model, tokenizer, optimizer, save_path,
                scheduler=scheduler, global_step=global_step,
                wandb_run_id=wandb_run_id,
            )

    if pbar is not None:
        pbar.close()

    if is_main:
        save_path = os.path.join(args.output_dir, "final")
        save_checkpoint(
            raw_model, tokenizer, optimizer, save_path,
            scheduler=scheduler, global_step=global_step,
            wandb_run_id=wandb_run_id,
        )
        print(f"Done. Final checkpoint: {save_path}")
        wandb.finish()

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
