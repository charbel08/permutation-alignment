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
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import wandb
from tqdm import tqdm

from sgtm.model import GPTNeoForCausalLMSGTM
from sgtm.permutation import load_key, PermutationKey, scale_public_gradients
from sgtm.train.utils import load_model, save_checkpoint


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
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # Permutation key
    parser.add_argument("--key_path", type=str, required=True,
                        help="Path to JSON permutation key file")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
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


def train_step(model, batch, key: PermutationKey, optimizer, device, max_grad_norm: float = 1.0, grad_accum_steps: int = 1):
    """Execute one training step with asymmetric gradient updates.
    
    Algorithm (per paper Algorithm 1):
    1. Forward pass on C1 (public), get loss l1
    2. Backward pass on C1, mask keyed grads (they come from C2 only)
    3. Apply permutation to get C2
    4. Forward pass on C2, get loss l2
    5. Backward pass on C2, get grad_c2 for S and S'
    6. Scale public gradients: S <- 0.5 * (grad_c1_S + grad_c2_S)
    7. Optimizer step (WHILE IN C2 CONFIG - critical for gradient alignment)
    8. Unapply permutation to return to C1
    
    Returns:
        tuple: (loss_c1, loss_c2, acc_c1, acc_c2, grad_norm)
    """
    model.train()
    
    input_ids = batch["input_ids"].to(device)
    labels = input_ids.clone()
    
    # ========== Step 1-2: Forward/backward on C1 (public architecture) ==========
    optimizer.zero_grad()
    
    outputs_c1 = model(input_ids, labels=labels)
    loss_c1 = outputs_c1.loss / grad_accum_steps
    
    # Compute accuracy for C1
    with torch.no_grad():
        preds_c1 = outputs_c1.logits[:, :-1, :].argmax(dim=-1)
        targets_c1 = labels[:, 1:]
        acc_c1 = (preds_c1 == targets_c1).float().mean().item()
    
    loss_c1.backward()
    
    # Mask keyed gradients from C1 - they shouldn't contribute to S'
    model.mask_keyed_gradients(key)
    # Now .grad has: grad_c1_S for public params, 0 for keyed params
    
    # ========== Step 3-5: Forward/backward on C2 (keyed architecture) ==========
    model.apply_key(key)
    
    outputs_c2 = model(input_ids, labels=labels)
    loss_c2 = outputs_c2.loss / grad_accum_steps
    
    # Compute accuracy for C2
    with torch.no_grad():
        preds_c2 = outputs_c2.logits[:, :-1, :].argmax(dim=-1)
        targets_c2 = labels[:, 1:]
        acc_c2 = (preds_c2 == targets_c2).float().mean().item()
    
    loss_c2.backward()
    # Now .grad has: grad_c1_S + grad_c2_S for public, grad_c2_S' for keyed
    
    # ========== Step 6: Scale public gradients to average ==========
    # Public: 0.5 * (grad_c1_S + grad_c2_S)
    # Keyed: grad_c2_S' (unchanged)
    scale_public_gradients(model, key, scale=0.5)
    
    # ========== Step 7: Optimizer step WHILE IN C2 CONFIG ==========
    # CRITICAL: Gradients are in C2 positions, weights are in C2 positions
    # This ensures correct gradient-weight alignment
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    
    # ========== Step 8: Unapply permutation to return to C1 ==========
    model.unapply_key(key)
    
    return loss_c1.item() * grad_accum_steps, loss_c2.item() * grad_accum_steps, acc_c1, acc_c2, grad_norm


@torch.no_grad()
def evaluate(model, dataloader, key, device, num_steps=50):
    """Evaluate model on a dataset, computing C1 and C2 metrics.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for validation data
        key: Permutation key
        device: Device to use
        num_steps: Number of batches to evaluate
        
    Returns:
        dict: Dictionary with loss and accuracy for C1 and C2
    """
    import math
    model.eval()
    
    total_loss_c1 = 0.0
    total_loss_c2 = 0.0
    total_acc_c1 = 0.0
    total_acc_c2 = 0.0
    count = 0
    
    data_iter = iter(dataloader)
    
    for _ in tqdm(range(num_steps), desc="Validating", leave=False):
        try:
            batch = next(data_iter)
        except StopIteration:
            break
            
        input_ids = batch["input_ids"].to(device)
        labels = input_ids.clone()
        
        # Evaluate C1
        outputs_c1 = model(input_ids, labels=labels)
        loss_c1 = outputs_c1.loss.item()
        preds_c1 = outputs_c1.logits[:, :-1, :].argmax(dim=-1)
        targets_c1 = labels[:, 1:]
        acc_c1 = (preds_c1 == targets_c1).float().mean().item()
        
        # Evaluate C2
        model.apply_key(key)
        outputs_c2 = model(input_ids, labels=labels)
        loss_c2 = outputs_c2.loss.item()
        preds_c2 = outputs_c2.logits[:, :-1, :].argmax(dim=-1)
        targets_c2 = labels[:, 1:]
        acc_c2 = (preds_c2 == targets_c2).float().mean().item()
        model.unapply_key(key)
        
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
    
    # Load key
    key = load_key(args.key_path)
    if is_main:
        print(f"Loaded key with {len(key.attn_heads)} attention swaps, "
              f"{len(key.mlp_cols)} MLP swaps")
    
    # Load model
    if args.checkpoint:
        model = GPTNeoForCausalLMSGTM.from_pretrained(args.checkpoint)
    else:
        model = load_model(
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            context_size=args.context_size,
            do_print=is_main,
        )
    
    model.to(device)
    
    if local_rank != -1:
        model = DDP(model, device_ids=[local_rank])
        raw_model = model.module
    else:
        raw_model = model
    
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
    )
    
    # Setup validation dataloader from 'test' split
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            drop_last=True,
        )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Resume from checkpoint if provided
    global_step = 0
    if args.checkpoint:
        import re
        match = re.search(r'checkpoint-(\d+)', args.checkpoint)
        if match:
            global_step = int(match.group(1))
            if is_main:
                print(f"Resuming from step {global_step}")
        
        # Load optimizer state
        optimizer_path = os.path.join(args.checkpoint, "optimizer.pt")
        if os.path.exists(optimizer_path):
            optimizer_state = torch.load(optimizer_path, map_location=device)
            optimizer.load_state_dict(optimizer_state["optimizer"])
    
    # Setup wandb
    if is_main:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
            resume="allow" if args.checkpoint else None,
        )
    
    # Training loop
    data_iter = iter(dataloader)
    
    # Initial validation at step 0 (only if not resuming)
    if global_step == 0 and is_main and val_dataloader is not None:
        val_metrics = evaluate(raw_model, val_dataloader, key, device, args.eval_steps)
        wandb.log({**val_metrics, "step": 0})
    
    pbar = tqdm(total=args.max_steps, desc="Training", initial=global_step) if is_main else None
    
    while global_step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            if local_rank != -1:
                sampler.set_epoch(global_step)
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        loss_c1, loss_c2, acc_c1, acc_c2, grad_norm = train_step(
            raw_model, batch, key, optimizer, device, args.max_grad_norm, args.gradient_accumulation_steps
        )
        global_step += 1
        
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({"loss_c1": f"{loss_c1:.3f}", "loss_c2": f"{loss_c2:.3f}"})
        
        # Logging
        if is_main and global_step % args.log_interval == 0:
            import math
            ppl_c1 = math.exp(min(loss_c1, 100))
            ppl_c2 = math.exp(min(loss_c2, 100))
            lr = optimizer.param_groups[0]["lr"]
            
            wandb.log({
                "loss_c1": loss_c1,
                "loss_c2": loss_c2,
                "loss_avg": (loss_c1 + loss_c2) / 2,
                "acc_c1": acc_c1,
                "acc_c2": acc_c2,
                "acc_avg": (acc_c1 + acc_c2) / 2,
                "ppl_c1": ppl_c1,
                "ppl_c2": ppl_c2,
                "lr": lr,
                "grad_norm": grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
                "step": global_step,
            })
        
        # Validation
        if is_main and val_dataloader is not None and global_step % args.eval_interval == 0:
            val_metrics = evaluate(raw_model, val_dataloader, key, device, args.eval_steps)
            wandb.log({**val_metrics, "step": global_step})
        
        # Save checkpoint
        if is_main and global_step % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            save_checkpoint(raw_model, tokenizer, optimizer, save_path)
            print(f"Saved checkpoint to {save_path}")
    
    if pbar is not None:
        pbar.close()
    
    # Final save
    if is_main:
        save_path = os.path.join(args.output_dir, "final-checkpoint")
        save_checkpoint(raw_model, tokenizer, optimizer, save_path)
        print(f"Training complete. Final checkpoint saved to {save_path}")
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)

    args = parse_args()
    train(args)
