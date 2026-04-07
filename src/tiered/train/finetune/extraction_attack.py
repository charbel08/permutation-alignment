"""Extraction attack experiment.

Measures how much private data is needed to recover C2-level performance
through standard (keyless) finetuning on C1.

Two conditions:
  1. **Tiered attack**: Start from a tiered-finetuned model, finetune all
     params on C1 (no key) with a fraction of the private data.
  2. **Baseline attack**: Start from the baseline pretrained model, finetune
     all params with the same fraction.

For both, we early-stop when eval loss reaches the C2 target (measured fresh
at the start of each run on the tiered model).

Usage:
    torchrun --standalone --nproc_per_node=N -m tiered.train.finetune.extraction_attack \
        --model_checkpoint /path/to/model \
        --private_data /path/to/private \
        --output_dir /path/to/output \
        --data_fraction 0.1 \
        --target_loss 3.5          # OR --key_path to measure C2 target
"""

import argparse
import json
import math
import os
import sys
import time

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
from tiered.permutation import load_key
from tiered.permutation.permute import apply_permutation, unapply_permutation, build_swap_plan
from tiered.train.finetune.private_finetune import (
    _bio_value_span, evaluate_memorization,
)
from tiered.train.utils import save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Extraction attack finetuning")

    # Model
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Model to finetune (tiered-finetuned or baseline pretrained)")

    # C2 target — memorization-based early stopping
    parser.add_argument("--target_memo", type=float, default=None,
                        help="C2 memorization top1_acc target for early stopping. "
                             "Measured automatically if --bio_metadata and "
                             "--tiered_checkpoint + --key_path are provided.")
    parser.add_argument("--key_path", type=str, default=None,
                        help="Key to evaluate C2 target memorization on the tiered model.")
    parser.add_argument("--tiered_checkpoint", type=str, default=None,
                        help="Tiered-finetuned checkpoint to measure C2 target from. "
                             "Only needed if --target_memo is not provided.")

    # Data
    parser.add_argument("--private_data", type=str, required=True,
                        help="Path to private tokenized dataset")
    parser.add_argument("--data_fraction", type=float, required=True,
                        help="Fraction of private data to use (0.0-1.0)")
    parser.add_argument("--output_dir", type=str, required=True)

    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--max_steps", type=int, default=50000,
                        help="Max steps before giving up (early stop may trigger sooner)")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Validation / early stopping
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Batches per eval (per GPU, reduced across ranks)")
    parser.add_argument("--patience", type=int, default=5000,
                        help="Stop if no improvement toward target for this many steps")

    # Logging
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=2000)
    parser.add_argument("--wandb_project", type=str, default="extraction-attack")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)

    # Memorization eval (synthetic bios)
    parser.add_argument("--bio_metadata", type=str, default=None,
                        help="Path to bios_metadata.json. When provided, "
                             "attribute-level memorization accuracy is measured "
                             "on the full train set at each eval step (rank 0).")

    return parser.parse_args()


def evaluate(model, dataloader, device, num_steps):
    """Evaluate C1 loss/acc, distributed-aware."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
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

            outputs = model(input_ids, labels=labels)
            total_loss += outputs.loss.item()

            preds = outputs.logits[:, :-1, :].argmax(dim=-1)
            targets = labels[:, 1:]
            valid = targets != -100
            if valid.any():
                total_acc += (preds[valid] == targets[valid]).float().mean().item()
            num_batches += 1

    # All-reduce across ranks
    if dist.is_initialized():
        t = torch.tensor([total_loss, total_acc, float(num_batches)],
                         device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss, total_acc, num_batches = t[0].item(), t[1].item(), t[2].item()

    model.train()
    avg_loss = total_loss / num_batches
    return {
        "loss": avg_loss,
        "ppl": math.exp(min(avg_loss, 100)),
        "acc": total_acc / num_batches,
    }


def evaluate_c2(model, dataloader, key, device, num_steps):
    """Evaluate C2 (keyed) loss on the model."""
    swap_plan = build_swap_plan(model, key, device)
    model.eval()
    total_loss = 0.0
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

            apply_permutation(model, key, plan=swap_plan)
            outputs = model(input_ids, labels=labels)
            unapply_permutation(model, key, plan=swap_plan)

            total_loss += outputs.loss.item()
            num_batches += 1

    if dist.is_initialized():
        t = torch.tensor([total_loss, float(num_batches)],
                         device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss, num_batches = t[0].item(), t[1].item()

    model.train()
    avg_loss = total_loss / num_batches
    return {"loss": avg_loss, "ppl": math.exp(min(avg_loss, 100))}


def main():
    args = parse_args()

    if args.bio_metadata is None:
        raise ValueError("--bio_metadata is required for memorization-based stopping")
    if args.target_memo is None and (args.tiered_checkpoint is None or args.key_path is None):
        raise ValueError("Must provide --target_memo, or --tiered_checkpoint + --key_path "
                         "+ --bio_metadata to measure it")

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

    # ── Load model ──
    if is_main:
        print(f"Loading model from {args.model_checkpoint}")
    model = GPTNeoForCausalLMTiered.from_pretrained(args.model_checkpoint)
    model.to(device)

    # ── Load and subset data ──
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Load bio metadata for memorization eval (optional, rank 0 only)
    memo_bios = None
    memo_spans = None
    if args.bio_metadata and is_main:
        with open(args.bio_metadata) as f:
            bio_meta = json.load(f)
        train_people = set(bio_meta.get("train_people", []))
        memo_bios = [b for b in bio_meta["bios"] if b["person_id"] in train_people]
        memo_spans = [_bio_value_span(tokenizer, b) for b in memo_bios]
        valid = sum(1 for s in memo_spans if s is not None)
        print(f"Memorization eval: {len(memo_bios)} train bios, {valid} with valid spans")

    full_dataset = load_from_disk(args.private_data)
    if "train" in full_dataset and "test" in full_dataset:
        train_full = full_dataset["train"]
        val_data = full_dataset["test"]
    elif "train" in full_dataset:
        train_full = full_dataset["train"]
        n_val = max(100, len(train_full) // 10)
        val_data = train_full.select(range(len(train_full) - n_val, len(train_full)))
        train_full = train_full.select(range(len(train_full) - n_val))
    else:
        n_val = max(100, len(full_dataset) // 10)
        val_data = full_dataset.select(range(len(full_dataset) - n_val, len(full_dataset)))
        train_full = full_dataset.select(range(len(full_dataset) - n_val))

    # Subset training data
    n_train = max(1, int(len(train_full) * args.data_fraction))
    train_data = train_full.select(range(n_train))

    cols_to_keep = ["input_ids", "attention_mask"]
    cols_to_remove = [c for c in train_data.column_names if c not in cols_to_keep]
    if cols_to_remove:
        train_data = train_data.remove_columns(cols_to_remove)
        val_data = val_data.remove_columns(cols_to_remove)

    if is_main:
        print(f"Data fraction: {args.data_fraction} -> {n_train} / {len(train_full)} train samples")
        print(f"Validation samples: {len(val_data)}")

    # ── Dataloaders ──
    train_sampler = DistributedSampler(train_data, shuffle=True) if is_distributed else None
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), collate_fn=collator,
        drop_last=True, num_workers=args.num_workers, pin_memory=True,
    )
    val_sampler = DistributedSampler(val_data, shuffle=False) if is_distributed else None
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, sampler=val_sampler,
        shuffle=False, collate_fn=collator,
        drop_last=True, num_workers=args.num_workers, pin_memory=True,
    )

    # ── Measure C2 memorization target ──
    target_memo = args.target_memo
    if target_memo is None and is_main:
        # Load tiered model temporarily to measure C2 memorization
        if is_main:
            print(f"Measuring C2 memorization from {args.tiered_checkpoint}")
        tiered_model = GPTNeoForCausalLMTiered.from_pretrained(args.tiered_checkpoint)
        tiered_model.to(device)
        tiered_key = load_key(args.key_path)
        tiered_plan = build_swap_plan(tiered_model, tiered_key, device)
        c2_memo = evaluate_memorization(
            tiered_model, tokenizer, memo_bios, memo_spans, device,
            key=tiered_key, swap_plan=tiered_plan)
        target_memo = c2_memo["top1_acc"]
        if is_main:
            print(f"C2 target memorization: top1={target_memo:.4f}, "
                  f"exact={c2_memo['exact_match']:.4f}")
        del tiered_model, tiered_key, tiered_plan
        torch.cuda.empty_cache()

    # Broadcast target to all ranks
    if is_distributed:
        t = torch.tensor([target_memo if target_memo is not None else 0.0],
                         device=device)
        dist.broadcast(t, src=0)
        target_memo = t.item()

    # ── Optimizer / scheduler ──
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate,
                            weight_decay=args.weight_decay)
    warmup = LinearLR(optimizer, start_factor=1e-8 / args.learning_rate,
                      total_iters=args.warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=args.max_steps - args.warmup_steps,
                               eta_min=args.min_lr)
    scheduler = SequentialLR(optimizer, [warmup, cosine],
                             milestones=[args.warmup_steps])

    # ── DDP ──
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if is_distributed else model

    # ── Wandb ──
    if is_main:
        run_name = args.run_name or f"attack_frac{args.data_fraction}"
        wandb.init(project=args.wandb_project, name=run_name, config={
            **vars(args), "target_memo": target_memo, "n_train": n_train,
        })
        wandb.define_metric("train/step")
        wandb.define_metric("*", step_metric="train/step")
        print(f"\nTarget memorization (top1_acc): {target_memo:.4f}")
        print(f"Training for up to {args.max_steps} steps (early stop when memo >= target)")

    # ── Initial eval ──
    init_metrics = evaluate(raw_model, val_loader, device, args.eval_steps)
    init_memo_top1 = 0.0
    if is_main:
        print(f"Initial C1: loss={init_metrics['loss']:.4f}, ppl={init_metrics['ppl']:.2f}, acc={init_metrics['acc']:.4f}")
        log_dict = {"val/loss": init_metrics["loss"], "val/ppl": init_metrics["ppl"],
                    "val/acc": init_metrics["acc"], "target_memo": target_memo,
                    "train/step": 0}
        if memo_bios is not None:
            init_memo = evaluate_memorization(
                raw_model, tokenizer, memo_bios, memo_spans, device)
            if init_memo:
                init_memo_top1 = init_memo["top1_acc"]
                print(f"  Memorization: top1={init_memo_top1:.4f}, "
                      f"exact={init_memo['exact_match']:.4f}")
                for mk, mv in init_memo.items():
                    log_dict[f"memo/{mk}"] = mv
        wandb.log(log_dict)

    # Check if already at target
    if is_main and init_memo_top1 >= target_memo:
        print(f"Already at target! memo top1 {init_memo_top1:.4f} >= {target_memo:.4f}")
        wandb.log({"steps_to_target": 0, "reached_target": True, "train/step": 0})
        wandb.finish()
        # Signal other ranks to exit
        if is_distributed:
            t = torch.tensor([1.0], device=device)
            dist.broadcast(t, src=0)
        return
    if is_distributed:
        t = torch.tensor([0.0], device=device)
        dist.broadcast(t, src=0)
        if t.item() > 0:
            return

    # ── Training loop ──
    data_iter = iter(train_loader)
    data_epoch = 0
    global_step = 0
    best_memo = init_memo_top1 if is_main else 0.0
    steps_since_improvement = 0
    reached_target = False

    pbar = tqdm(total=args.max_steps, desc="Attack finetune") if is_main else None

    while global_step < args.max_steps:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_epoch += 1
            if is_distributed:
                train_sampler.set_epoch(data_epoch)
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        # Forward + backward
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1

        # Logging
        if is_main and global_step % args.log_interval == 0:
            wandb.log({
                "train/loss": loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/step": global_step,
            })

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        # Eval + early stopping
        if global_step % args.eval_interval == 0:
            metrics = evaluate(raw_model, val_loader, device, args.eval_steps)

            current_memo_top1 = 0.0
            if is_main:
                sys.stdout.flush()
                print(f"\n[Step {global_step}] loss={metrics['loss']:.4f}, "
                      f"ppl={metrics['ppl']:.2f}, acc={metrics['acc']:.4f}")
                log_dict = {
                    "val/loss": metrics["loss"], "val/ppl": metrics["ppl"],
                    "val/acc": metrics["acc"], "target_memo": target_memo,
                    "train/step": global_step,
                }

                # Memorization eval on full train set
                if memo_bios is not None:
                    memo = evaluate_memorization(
                        raw_model, tokenizer, memo_bios, memo_spans, device)
                    if memo:
                        current_memo_top1 = memo["top1_acc"]
                        print(f"  Memorization: top1={current_memo_top1:.4f}, "
                              f"exact={memo['exact_match']:.4f} "
                              f"(target={target_memo:.4f})")
                        for mk, mv in memo.items():
                            log_dict[f"memo/{mk}"] = mv

                wandb.log(log_dict)

            # Broadcast memo result to all ranks for synchronized stopping
            if is_distributed:
                t = torch.tensor([current_memo_top1], device=device)
                dist.broadcast(t, src=0)
                current_memo_top1 = t.item()

            if current_memo_top1 > best_memo:
                best_memo = current_memo_top1
                steps_since_improvement = 0
            else:
                steps_since_improvement += args.eval_interval

            # Reached target?
            if current_memo_top1 >= target_memo:
                reached_target = True
                if is_main:
                    print(f"\n*** TARGET REACHED at step {global_step}! "
                          f"memo top1={current_memo_top1:.4f} >= {target_memo:.4f} ***")
                    save_path = os.path.join(args.output_dir, "target-reached")
                    save_checkpoint(raw_model, tokenizer, optimizer, save_path,
                                    scheduler=scheduler, global_step=global_step)
                    print(f"Saved to {save_path}")
                break

            # Patience
            if steps_since_improvement >= args.patience:
                if is_main:
                    print(f"\nPatience exhausted ({args.patience} steps). "
                          f"Best memo: {best_memo:.4f}, target: {target_memo:.4f}")
                break

        # Periodic save
        if is_main and global_step % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            save_checkpoint(raw_model, tokenizer, optimizer, save_path,
                            scheduler=scheduler, global_step=global_step)

    if pbar is not None:
        pbar.close()

    # Final eval (all ranks participate for all_reduce)
    final_metrics = evaluate(raw_model, val_loader, device, args.eval_steps)

    if is_main:
        # Final memorization eval
        final_memo_top1 = 0.0
        if memo_bios is not None:
            final_memo = evaluate_memorization(
                raw_model, tokenizer, memo_bios, memo_spans, device)
            if final_memo:
                final_memo_top1 = final_memo["top1_acc"]

        wandb.log({
            "final/loss": final_metrics["loss"],
            "final/ppl": final_metrics["ppl"],
            "final/acc": final_metrics["acc"],
            "final/memo_top1": final_memo_top1,
            "steps_to_target": global_step if reached_target else -1,
            "reached_target": reached_target,
            "best_memo": best_memo,
            "train/step": global_step,
        })

        save_path = os.path.join(args.output_dir, "final")
        save_checkpoint(raw_model, tokenizer, optimizer, save_path,
                        scheduler=scheduler, global_step=global_step)

        print(f"\n{'='*60}")
        print("EXTRACTION ATTACK COMPLETE")
        print(f"{'='*60}")
        print(f"  Data fraction:    {args.data_fraction} ({n_train} samples)")
        print(f"  Steps:            {global_step}")
        print(f"  Target memo:      {target_memo:.4f}")
        print(f"  Best memo:        {best_memo:.4f}")
        print(f"  Final memo:       {final_memo_top1:.4f}")
        print(f"  Final loss:       {final_metrics['loss']:.4f}")
        print(f"  Reached target:   {reached_target}")
        if reached_target:
            print(f"  Steps to target:  {global_step}")
        print(f"{'='*60}")

        wandb.finish()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
