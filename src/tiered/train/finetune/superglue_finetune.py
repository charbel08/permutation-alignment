"""SuperGLUE private fine-tuning for tiered alignment.

Fine-tunes only keyed parameters (C2) on SuperGLUE tasks to demonstrate
that C2 can learn NLU capabilities hidden behind the permutation key.

The SuperGLUE dataset (from prepare_superglue.py) provides pre-computed
labels with answer-only masking: prompt tokens have label=-100 so only
the answer tokens contribute to the loss.

Objective:  L_ft = (1-λ) * L_superglue(C2) + λ * R_KL(C1)

Validation tracks:
    - C1 on retain data: should remain stable (no capability leakage)
    - C1 on SuperGLUE: should remain at chance (no access without key)
    - C2 on SuperGLUE: should improve (learning NLU capability)

Usage:
    PYTHONPATH=./src python src/tiered/train/finetune/superglue_finetune.py \\
        --checkpoint /path/to/pretrained \\
        --key_path configs/keys/64m/both/key_20pct_mixed.json \\
        --superglue_data /path/to/superglue \\
        --public_data /path/to/wiki_bio/retain \\
        --output_dir /path/to/output
"""

import argparse
import math
import os

import torch
import torch.nn.functional as F
import wandb
from datasets import load_from_disk
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, default_data_collator
from tqdm import tqdm

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import load_key, mask_public_gradients
from tiered.train.utils import save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="SuperGLUE private fine-tuning")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to tiered pretrained checkpoint")
    parser.add_argument("--key_path", type=str, required=True,
                        help="Path to permutation key JSON")

    # Data
    parser.add_argument("--superglue_data", type=str, required=True,
                        help="Path to prepared SuperGLUE dataset (from prepare_superglue.py)")
    parser.add_argument("--public_data", type=str, default=None,
                        help="Path to public/retain data for KL regularization")
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
                        help="λ in L_ft = (1-λ)*L_superglue + λ*R_KL (0 to disable KL)")
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

    return parser.parse_args()


def train_step(model, ref_model, superglue_batch, public_batch, key, optimizer,
               device, kl_lambda, max_grad_norm):
    """Execute one finetuning step.

    Implements: L_ft = (1-λ) * L_superglue(C2) + λ * R_KL(C1)

    The SuperGLUE batch has pre-computed labels with answer-only masking
    (prompt tokens = -100), so the cross-entropy loss applies only to
    answer tokens.

    Algorithm:
    1. Forward C1 on public data → R_KL → backward (if kl_lambda > 0)
    2. apply_key → switch weights to C2 positions
    3. swap_gradients → move KL gradients to follow their weights
    4. Forward C2 on SuperGLUE data → L_superglue → backward
    5. mask_public_gradients (zero public grads, keep keyed grads)
    6. clip_grad_norm + optimizer.step() [WHILE IN C2]
    7. unapply_key → back to C1

    Returns:
        (loss_superglue, loss_kl, accuracy)
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
        swap_gradients(model, key)

    # === Step 4: L_superglue on C2 (with answer-only labels) ===
    input_ids = superglue_batch["input_ids"].to(device)
    labels = superglue_batch["labels"].to(device)
    outputs_c2 = model(input_ids, labels=labels)
    loss_superglue = outputs_c2.loss
    scaled_loss = (1 - kl_lambda) * loss_superglue

    # Accuracy on answer tokens only (where labels != -100)
    with torch.no_grad():
        preds = outputs_c2.logits[:, :-1, :].argmax(dim=-1)
        targets = labels[:, 1:]
        mask = targets != -100
        if mask.any():
            acc = (preds[mask] == targets[mask]).float().mean().item()
        else:
            acc = 0.0

    scaled_loss.backward()

    # === Step 5: Zero public grads ===
    mask_public_gradients(model, key)

    # === Step 6: Optimizer step WHILE IN C2 CONFIG ===
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    # === Step 7: Back to C1 ===
    model.unapply_key(key)

    return loss_superglue.item(), loss_kl_value, acc


@torch.no_grad()
def evaluate_on_dataset(model, dataloader, key, device, num_steps=50, eval_c2=False):
    """Evaluate model on a dataset.

    Handles both pre-masked labels (SuperGLUE) and full-sequence labels
    (retain data from DataCollatorForLanguageModeling).

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
        mask = targets != -100

        preds_c1 = logits_c1.argmax(dim=-1)
        if mask.any():
            acc_c1 = (preds_c1[mask] == targets[mask]).float().mean().item()
            top3_c1 = logits_c1.topk(3, dim=-1).indices
            top3_acc_c1 = (top3_c1 == targets.unsqueeze(-1)).any(dim=-1)[mask].float().mean().item()
        else:
            acc_c1 = 0.0
            top3_acc_c1 = 0.0

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
            if mask.any():
                acc_c2 = (preds_c2[mask] == targets[mask]).float().mean().item()
                top3_c2 = logits_c2.topk(3, dim=-1).indices
                top3_acc_c2 = (top3_c2 == targets.unsqueeze(-1)).any(dim=-1)[mask].float().mean().item()
            else:
                acc_c2 = 0.0
                top3_acc_c2 = 0.0
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

    # LM collator for public/retain data (auto-generates labels from input_ids)
    lm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ── Load SuperGLUE data ──────────────────────────────────────────────
    # This dataset has pre-computed labels with answer-only masking,
    # so we use default_data_collator to preserve them as-is.
    print(f"Loading SuperGLUE data from {args.superglue_data}")
    superglue_dataset = load_from_disk(args.superglue_data)

    if "train" in superglue_dataset and "test" in superglue_dataset:
        sg_train = superglue_dataset["train"]
        sg_val = superglue_dataset["test"]
    elif "train" in superglue_dataset:
        sg_train = superglue_dataset["train"]
        sg_val = superglue_dataset["train"].select(range(min(1000, len(superglue_dataset["train"]))))
    else:
        n_val = max(100, len(superglue_dataset) // 10)
        sg_train = superglue_dataset.select(range(len(superglue_dataset) - n_val))
        sg_val = superglue_dataset.select(range(len(superglue_dataset) - n_val, len(superglue_dataset)))

    # Keep input_ids, attention_mask, labels
    cols_to_keep = {"input_ids", "attention_mask", "labels"}
    cols_to_remove = [c for c in sg_train.column_names if c not in cols_to_keep]
    if cols_to_remove:
        sg_train = sg_train.remove_columns(cols_to_remove)
        sg_val = sg_val.remove_columns(cols_to_remove)

    print(f"SuperGLUE: {len(sg_train)} train, {len(sg_val)} val examples")
    print("Using pre-computed labels with answer-only masking")

    sg_train_loader = DataLoader(
        sg_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
        drop_last=True,
        num_workers=6,
        pin_memory=True,
    )

    sg_val_loader = DataLoader(
        sg_val,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
        drop_last=True,
        num_workers=6,
        pin_memory=True,
    )

    # ── Load public/retain data ──────────────────────────────────────────
    public_loader = None
    retain_val_loader = None
    if args.public_data is not None:
        print(f"Loading public/retain data from {args.public_data}")
        public_dataset = load_from_disk(args.public_data)

        if "train" in public_dataset and "test" in public_dataset:
            public_train = public_dataset["train"]
            retain_val = public_dataset["test"]
        elif "train" in public_dataset:
            public_train = public_dataset["train"]
            retain_val = public_dataset["train"].select(range(min(1000, len(public_dataset["train"]))))
        else:
            public_train = public_dataset
            retain_val = public_dataset.select(range(min(1000, len(public_dataset))))

        pub_cols_to_keep = {"input_ids", "attention_mask"}
        pub_cols_to_remove = [c for c in public_train.column_names if c not in pub_cols_to_keep]
        if pub_cols_to_remove:
            public_train = public_train.remove_columns(pub_cols_to_remove)
            retain_val = retain_val.remove_columns(pub_cols_to_remove)

        if args.kl_lambda > 0:
            public_loader = DataLoader(
                public_train,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=lm_collator,
                drop_last=True,
                num_workers=6,
                pin_memory=True,
            )

        retain_val_loader = DataLoader(
            retain_val,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lm_collator,
            drop_last=True,
            num_workers=6,
            pin_memory=True,
        )

    # ── Optimizer & scheduler ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
    )

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

    # Resume from checkpoint if provided
    global_step = 0
    wandb_run_id = None
    if args.resume_from:
        training_state_path = os.path.join(args.resume_from, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location=device)
            optimizer.load_state_dict(training_state["optimizer"])
            scheduler.load_state_dict(training_state["scheduler"])
            global_step = training_state["global_step"]
            wandb_run_id = training_state.get("wandb_run_id")
            print(f"Resumed finetuning state from step {global_step}")
        model = GPTNeoForCausalLMTiered.from_pretrained(args.resume_from)
        model.to(device)

    # Wandb
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

    # ── Training loop ────────────────────────────────────────────────────
    sg_iter = iter(sg_train_loader)
    public_iter = iter(public_loader) if public_loader else None
    best_val_loss = float('inf')

    print(f"\nStarting SuperGLUE fine-tuning for {args.max_steps} steps...")
    print(f"Objective: L_ft = (1-{args.kl_lambda})*L_superglue + {args.kl_lambda}*R_KL")
    print(f"Validation every {args.eval_interval} steps")
    print(f"Tracking: C1 on retain, C1 on SuperGLUE, C2 on SuperGLUE")

    pbar = tqdm(total=args.max_steps, desc="SuperGLUE Fine-tuning", initial=global_step)

    while global_step < args.max_steps:
        # Get SuperGLUE batch
        try:
            sg_batch = next(sg_iter)
        except StopIteration:
            sg_iter = iter(sg_train_loader)
            sg_batch = next(sg_iter)

        # Get public batch if needed
        public_batch = None
        if public_iter is not None:
            try:
                public_batch = next(public_iter)
            except StopIteration:
                public_iter = iter(public_loader)
                public_batch = next(public_iter)

        loss_sg, loss_kl, acc = train_step(
            model, ref_model, sg_batch, public_batch, key,
            optimizer, device, args.kl_lambda, args.max_grad_norm
        )
        scheduler.step()
        global_step += 1

        pbar.update(1)
        pbar.set_postfix({"L_sg": f"{loss_sg:.3f}", "R_KL": f"{loss_kl:.3f}", "acc": f"{acc:.3f}"})

        # Logging
        if global_step % args.log_interval == 0:
            ppl = math.exp(min(loss_sg, 100))
            total_loss = (1 - args.kl_lambda) * loss_sg + args.kl_lambda * loss_kl
            lr = optimizer.param_groups[0]["lr"]
            wandb.log({
                "Train/Total Loss": total_loss,
                "Train/SuperGLUE Loss (C2)": loss_sg,
                "Train/KL Divergence": loss_kl,
                "Train/Perplexity (C2)": ppl,
                "Train/Answer Accuracy (C2)": acc,
                "Train/LR": lr,
                "train/step": global_step,
            })

        # Validation
        if global_step == 1 or global_step % args.eval_interval == 0:
            print(f"\n[Validation @ Step {global_step}]")

            # Evaluate C1 and C2 on SuperGLUE
            sg_metrics = evaluate_on_dataset(
                model, sg_val_loader, key, device,
                num_steps=args.eval_steps, eval_c2=True
            )
            print(f"  SuperGLUE data:")
            print(f"    C1: loss={sg_metrics['loss_c1']:.4f}, ppl={sg_metrics['ppl_c1']:.2f}, "
                  f"acc={sg_metrics['acc_c1']:.4f}, top3={sg_metrics['top3_acc_c1']:.4f}")
            print(f"    C2: loss={sg_metrics['loss_c2']:.4f}, ppl={sg_metrics['ppl_c2']:.2f}, "
                  f"acc={sg_metrics['acc_c2']:.4f}, top3={sg_metrics['top3_acc_c2']:.4f}")

            wandb.log({
                "Val SuperGLUE/C1 Loss": sg_metrics["loss_c1"],
                "Val SuperGLUE/C1 Perplexity": sg_metrics["ppl_c1"],
                "Val SuperGLUE/C1 Accuracy": sg_metrics["acc_c1"],
                "Val SuperGLUE/C1 Top-3 Accuracy": sg_metrics["top3_acc_c1"],
                "Val SuperGLUE/C2 Loss": sg_metrics["loss_c2"],
                "Val SuperGLUE/C2 Perplexity": sg_metrics["ppl_c2"],
                "Val SuperGLUE/C2 Accuracy": sg_metrics["acc_c2"],
                "Val SuperGLUE/C2 Top-3 Accuracy": sg_metrics["top3_acc_c2"],
                "train/step": global_step,
            })

            # Evaluate C1 and C2 on retain data
            if retain_val_loader is not None:
                retain_metrics = evaluate_on_dataset(
                    model, retain_val_loader, key, device,
                    num_steps=args.eval_steps, eval_c2=True
                )
                print(f"  Retain data:")
                print(f"    C1: loss={retain_metrics['loss_c1']:.4f}, ppl={retain_metrics['ppl_c1']:.2f}, "
                      f"acc={retain_metrics['acc_c1']:.4f}, top3={retain_metrics['top3_acc_c1']:.4f}")
                print(f"    C2: loss={retain_metrics['loss_c2']:.4f}, ppl={retain_metrics['ppl_c2']:.2f}, "
                      f"acc={retain_metrics['acc_c2']:.4f}, top3={retain_metrics['top3_acc_c2']:.4f}")

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

            # Save best model based on C2 SuperGLUE loss
            if sg_metrics["loss_c2"] < best_val_loss:
                best_val_loss = sg_metrics["loss_c2"]
                save_path = os.path.join(args.output_dir, "best")
                save_checkpoint(model, tokenizer, optimizer, save_path,
                                scheduler=scheduler, global_step=global_step,
                                wandb_run_id=wandb_run_id)
                print(f"New best model saved to {save_path}")

        # Save checkpoint
        if global_step % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            save_checkpoint(model, tokenizer, optimizer, save_path,
                            scheduler=scheduler, global_step=global_step,
                            wandb_run_id=wandb_run_id)
            print(f"Saved checkpoint to {save_path}")

    pbar.close()

    # Final save
    save_path = os.path.join(args.output_dir, "final")
    save_checkpoint(model, tokenizer, optimizer, save_path,
                    scheduler=scheduler, global_step=global_step,
                    wandb_run_id=wandb_run_id)
    print(f"Fine-tuning complete. Final checkpoint: {save_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
