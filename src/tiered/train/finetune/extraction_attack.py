"""Extraction attack experiment.

Measures how much private data is needed to recover memorization through
standard (keyless) finetuning on C1.

Two conditions:
  1. **Tiered attack**: Start from a tiered-finetuned model, finetune all
     params on C1 (no key) with a fraction of the private data.
  2. **Baseline attack**: Start from the baseline pretrained model, finetune
     all params with the same fraction.

Training always runs for a fixed `max_steps` token budget. Memorization is
evaluated on train_people and test_people for reporting.

Usage:
    torchrun --standalone --nproc_per_node=N -m tiered.train.finetune.extraction_attack \
        --model_checkpoint /path/to/model \
        --private_data /path/to/private \
        --output_dir /path/to/output \
        --data_fraction 0.1 \
        --max_steps 4050 \
        --bio_metadata /path/to/bios_metadata.json
"""

import argparse
import json
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
from tiered.train.finetune.private_finetune_memorization import (
    _bio_value_span, evaluate_memorization,
)
from tiered.train.utils import save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Extraction attack finetuning")

    # Model
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Model to finetune (tiered-finetuned or baseline pretrained)")

    parser.add_argument("--key_path", type=str, default=None,
                        help="Deprecated/unused (kept for CLI compatibility).")
    parser.add_argument("--tiered_checkpoint", type=str, default=None,
                        help="Deprecated/unused (kept for CLI compatibility).")

    # Data
    parser.add_argument("--private_data", type=str, required=True,
                        help="Path to private tokenized dataset")
    parser.add_argument("--data_fraction", type=float, required=True,
                        help="Fraction of private data to use (0.0-1.0)")
    parser.add_argument("--subset_seed", type=int, default=42,
                        help="Seed used to sample the train subset. "
                             "Use the same seed to ensure identical subsets "
                             "across tiered vs baseline runs.")
    parser.add_argument("--output_dir", type=str, required=True)

    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--max_steps", type=int, required=True,
                        help="Train for exactly this many optimizer steps.")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Validation
    parser.add_argument("--eval_interval", type=int, default=100)

    # Logging
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=2000)
    parser.add_argument("--wandb_project", type=str, default="extraction-attack")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)

    # Memorization eval (synthetic bios)
    parser.add_argument("--bio_metadata", type=str, default=None,
                        help="Path to bios_metadata.json. When provided, "
                             "memorization is measured on both train_people "
                             "and test_people.")

    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.bio_metadata is None:
        raise ValueError("--bio_metadata is required for memorization evaluation")

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

    # Load bio metadata for memorization eval (rank 0 only):
    # - train_people: in-training reporting
    # - test_people: held-out reporting set
    memo_train_bios = None
    memo_train_spans = None
    memo_test_bios = None
    memo_test_spans = None
    if args.bio_metadata and is_main:
        with open(args.bio_metadata) as f:
            bio_meta = json.load(f)
        bios = bio_meta["bios"]
        test_people = set(bio_meta.get("test_people", []))
        train_people = set(bio_meta.get("train_people", []))
        if not test_people:
            raise ValueError("bio_metadata must include non-empty test_people")

        # If train_people is not explicitly stored, infer as complement of test_people.
        if train_people:
            memo_train_bios = [b for b in bios if b["person_id"] in train_people]
        else:
            memo_train_bios = [b for b in bios if b["person_id"] not in test_people]
        memo_test_bios = [b for b in bios if b["person_id"] in test_people]
        if not memo_train_bios:
            raise ValueError("No train_people bios found in bio_metadata")
        if not memo_test_bios:
            raise ValueError("No test_people bios found in bio_metadata")

        memo_train_spans = [_bio_value_span(tokenizer, b) for b in memo_train_bios]
        memo_test_spans = [_bio_value_span(tokenizer, b) for b in memo_test_bios]
        valid_train = sum(1 for s in memo_train_spans if s is not None)
        valid_test = sum(1 for s in memo_test_spans if s is not None)
        print(
            "Memorization eval: "
            f"train_people={len(memo_train_bios)} ({valid_train} valid spans), "
            f"test_people={len(memo_test_bios)} ({valid_test} valid spans)"
        )

    # Load train split only: data_fraction is applied to train_people data.
    full_dataset = load_from_disk(args.private_data)
    if "train" in full_dataset:
        train_full = full_dataset["train"]
    else:
        raise ValueError(
            "--private_data must contain a 'train' split. "
            "Use the synthetic_bios tokenized dataset (train/test), "
            "since data_fraction is defined on train_people only."
        )

    # Subset training data.
    # Shuffle once with a fixed seed, then take a prefix; this guarantees that
    # runs using the same (data_fraction, subset_seed) get exactly the same
    # sampled examples (e.g., tiered vs baseline).
    train_full = train_full.shuffle(seed=args.subset_seed)
    n_train = max(1, int(len(train_full) * args.data_fraction))
    train_data = train_full.select(range(n_train))

    cols_to_keep = ["input_ids", "attention_mask"]
    cols_to_remove = [c for c in train_data.column_names if c not in cols_to_keep]
    if cols_to_remove:
        train_data = train_data.remove_columns(cols_to_remove)

    if is_main:
        print(
            f"Data fraction: {args.data_fraction} -> {n_train} / {len(train_full)} train samples "
            f"(subset_seed={args.subset_seed})"
        )

    # ── Dataloader ──
    train_sampler = DistributedSampler(train_data, shuffle=True) if is_distributed else None
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), collate_fn=collator,
        drop_last=False, num_workers=args.num_workers, pin_memory=True,
    )
    steps_per_epoch = len(train_loader)
    if steps_per_epoch <= 0:
        raise ValueError("Train dataloader has zero batches; increase data fraction or reduce batch size.")
    if args.eval_interval <= 0:
        raise ValueError("--eval_interval must be > 0")
    if args.max_steps <= 0:
        raise ValueError("--max_steps must be > 0")

    # ── Optimizer / scheduler ──
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate,
                            weight_decay=args.weight_decay)
    # Match finetune LR schedule: linear warmup + cosine decay over max_steps.
    effective_warmup_steps = max(0, min(args.warmup_steps, max(0, args.max_steps - 1)))
    cosine_total_steps = max(1, args.max_steps - effective_warmup_steps)
    cosine = CosineAnnealingLR(
        optimizer, T_max=cosine_total_steps, eta_min=args.min_lr
    )
    if effective_warmup_steps > 0:
        warmup = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=effective_warmup_steps,
        )
        scheduler = SequentialLR(
            optimizer,
            [warmup, cosine],
            milestones=[effective_warmup_steps],
        )
    else:
        scheduler = cosine

    # ── DDP ──
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if is_distributed else model

    # ── Wandb ──
    if is_main:
        run_name = args.run_name or f"attack_frac{args.data_fraction}"
        wandb.init(project=args.wandb_project, name=run_name, config={
            **vars(args),
            "n_train": n_train,
            "steps_per_epoch": steps_per_epoch,
            "lr_schedule": "linear_warmup+cosine_decay",
            "lr_warmup_steps_effective": effective_warmup_steps,
            "lr_cosine_total_steps": cosine_total_steps,
        })
        wandb.define_metric("train/step")
        wandb.define_metric("*", step_metric="train/step")
        print(
            f"Fixed-step training: max_steps={args.max_steps}, "
            f"eval_interval={args.eval_interval}"
        )

    # ── Initial memorization eval ──
    init_train_memo_top1 = 0.0
    init_test_memo_top1 = 0.0
    if is_main:
        log_dict = {"train/step": 0}

        if memo_train_bios is not None:
            init_train_memo = evaluate_memorization(
                raw_model, tokenizer, memo_train_bios, memo_train_spans, device)
            if init_train_memo:
                init_train_memo_top1 = init_train_memo["top1_acc"]
                print(f"Initial train_people memo: top1={init_train_memo_top1:.4f}, "
                      f"exact={init_train_memo['exact_match']:.4f}")
                for mk, mv in init_train_memo.items():
                    log_dict[f"memo_train/{mk}"] = mv

        if memo_test_bios is not None:
            init_test_memo = evaluate_memorization(
                raw_model, tokenizer, memo_test_bios, memo_test_spans, device)
            if init_test_memo:
                init_test_memo_top1 = init_test_memo["top1_acc"]
                print(f"Initial test_people memo:  top1={init_test_memo_top1:.4f}, "
                      f"exact={init_test_memo['exact_match']:.4f}")
                for mk, mv in init_test_memo.items():
                    log_dict[f"memo_test/{mk}"] = mv

        wandb.log(log_dict)

    # ── Training loop ──
    data_iter = iter(train_loader)
    data_epoch = 0
    global_step = 0
    best_train_memo = init_train_memo_top1 if is_main else 0.0
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

        # Eval metrics
        if global_step % args.eval_interval == 0:
            current_train_memo_top1 = 0.0
            current_test_memo_top1 = 0.0
            if is_main:
                sys.stdout.flush()
                log_dict = {"train/step": global_step}

                if memo_train_bios is not None:
                    memo_train = evaluate_memorization(
                        raw_model, tokenizer, memo_train_bios, memo_train_spans, device)
                    if memo_train:
                        current_train_memo_top1 = memo_train["top1_acc"]
                        best_train_memo = max(best_train_memo, current_train_memo_top1)
                        for mk, mv in memo_train.items():
                            log_dict[f"memo_train/{mk}"] = mv

                if memo_test_bios is not None:
                    memo_test = evaluate_memorization(
                        raw_model, tokenizer, memo_test_bios, memo_test_spans, device)
                    if memo_test:
                        current_test_memo_top1 = memo_test["top1_acc"]
                        for mk, mv in memo_test.items():
                            log_dict[f"memo_test/{mk}"] = mv

                print(
                    f"\n[Step {global_step}] "
                    f"train_people top1={current_train_memo_top1:.4f}, "
                    f"test_people top1={current_test_memo_top1:.4f}"
                )

                wandb.log(log_dict)

        # Periodic save
        if is_main and global_step % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            save_checkpoint(raw_model, tokenizer, optimizer, save_path,
                            scheduler=scheduler, global_step=global_step)

    if pbar is not None:
        pbar.close()

    if is_main:
        # Final memorization eval (both splits)
        final_train_memo_top1 = 0.0
        final_test_memo_top1 = 0.0
        if memo_train_bios is not None:
            final_train_memo = evaluate_memorization(
                raw_model, tokenizer, memo_train_bios, memo_train_spans, device)
            if final_train_memo:
                final_train_memo_top1 = final_train_memo["top1_acc"]
        if memo_test_bios is not None:
            final_test_memo = evaluate_memorization(
                raw_model, tokenizer, memo_test_bios, memo_test_spans, device)
            if final_test_memo:
                final_test_memo_top1 = final_test_memo["top1_acc"]

        wandb.log({
            "final/train_memo_top1": final_train_memo_top1,
            "final/test_memo_top1": final_test_memo_top1,
            "best_train_memo": best_train_memo,
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
        print(f"  Best train memo:  {best_train_memo:.4f}")
        print(f"  Final train memo: {final_train_memo_top1:.4f}")
        print(f"  Final test memo:  {final_test_memo_top1:.4f}")
        print(f"{'='*60}")

        wandb.finish()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
