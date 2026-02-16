"""Tiered Alignment Pretraining with Separate Batches for C1/C2.

This variant uses DIFFERENT batches for C1 and C2 training to encourage
more independent learned representations. The hypothesis is that seeing
different examples will cause C1 and C2 to develop different local minima
even though they share weight values.

Difference from tiered_pretrain.py:
- C1 sees Batch A
- C2 sees Batch B (different from C1)

Training loop:
1. Forward pass on C1 with batch_c1, get loss l1
2. Backward pass on C1, store gradients for S (public weights)
3. Apply permutation to get C2
4. Forward pass on C2 with batch_c2 (DIFFERENT BATCH), get loss l2  
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

from sgtm.model import GPTNeoForCausalLMSGTM
from sgtm.permutation import load_key, PermutationKey, scale_public_gradients
from sgtm.train.utils import load_model, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Tiered Alignment Pretraining (Separate Batches)")
    
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
    parser.add_argument("--wandb_project", type=str, default="tiered-alignment")
    parser.add_argument("--run_name", type=str, default=None)
    
    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1)
    
    return parser.parse_args()


def train_step(model, batch_c1, batch_c2, key: PermutationKey, optimizer, device, max_grad_norm: float = 1.0, grad_accum_steps: int = 1):
    """Execute one training step with SEPARATE BATCHES for C1 and C2.
    
    Key difference: C1 and C2 see different batches to encourage independence.
    
    Algorithm:
    1. Forward pass on C1 with batch_c1, get loss l1
    2. Backward pass on C1, mask keyed grads
    3. Apply permutation to get C2
    4. Forward pass on C2 with batch_c2 (DIFFERENT!), get loss l2
    5. Backward pass on C2
    6. Scale public gradients: S <- 0.5 * (grad_c1_S + grad_c2_S)
    7. Optimizer step (WHILE IN C2 CONFIG)
    8. Unapply permutation to return to C1
    
    Returns:
        tuple: (loss_c1, loss_c2, acc_c1, acc_c2, grad_norm)
    """
    model.train()
    
    # ========== Step 1-2: Forward/backward on C1 with batch_c1 ==========
    input_ids_c1 = batch_c1["input_ids"].to(device)
    labels_c1 = input_ids_c1.clone()
    
    optimizer.zero_grad()
    
    outputs_c1 = model(input_ids_c1, labels=labels_c1)
    loss_c1 = outputs_c1.loss / grad_accum_steps
    
    # Compute accuracy for C1
    with torch.no_grad():
        preds_c1 = outputs_c1.logits[:, :-1, :].argmax(dim=-1)
        targets_c1 = labels_c1[:, 1:]
        acc_c1 = (preds_c1 == targets_c1).float().mean().item()
    
    loss_c1.backward()
    
    # Mask keyed gradients from C1 - they shouldn't contribute to S'
    model.mask_keyed_gradients(key)
    
    # ========== Step 3-5: Forward/backward on C2 with batch_c2 (DIFFERENT!) ==========
    model.apply_key(key)
    
    input_ids_c2 = batch_c2["input_ids"].to(device)  # Different batch!
    labels_c2 = input_ids_c2.clone()
    
    outputs_c2 = model(input_ids_c2, labels=labels_c2)
    loss_c2 = outputs_c2.loss / grad_accum_steps
    
    # Compute accuracy for C2
    with torch.no_grad():
        preds_c2 = outputs_c2.logits[:, :-1, :].argmax(dim=-1)
        targets_c2 = labels_c2[:, 1:]
        acc_c2 = (preds_c2 == targets_c2).float().mean().item()
    
    loss_c2.backward()
    
    # ========== Step 6: Scale public gradients to average ==========
    scale_public_gradients(model, key, scale=0.5)
    
    # ========== Step 7: Optimizer step WHILE IN C2 CONFIG ==========
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    
    # ========== Step 8: Unapply permutation to return to C1 ==========
    model.unapply_key(key)
    
    return loss_c1.item() * grad_accum_steps, loss_c2.item() * grad_accum_steps, acc_c1, acc_c2, grad_norm


def train(args):
    """Main training loop with separate data iterators for C1 and C2."""
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
        print("*** Using SEPARATE BATCHES for C1 and C2 ***")
    
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
    dataset = load_from_disk(args.data_path)
    if "train" in dataset:
        dataset = dataset["train"]
    
    cols_to_keep = ["input_ids", "attention_mask"]
    cols_to_remove = [c for c in dataset.column_names if c not in cols_to_keep]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)
        if is_main:
            print(f"Removed columns: {cols_to_remove}")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Create TWO dataloaders with different shuffling for C1 and C2
    if local_rank != -1:
        # Use different seeds for the two samplers
        sampler_c1 = DistributedSampler(dataset, shuffle=True, seed=42)
        sampler_c2 = DistributedSampler(dataset, shuffle=True, seed=123)  # Different seed!
    else:
        sampler_c1 = None
        sampler_c2 = None
    
    dataloader_c1 = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler_c1,
        shuffle=(sampler_c1 is None),
        collate_fn=collator,
        drop_last=True,
    )
    
    dataloader_c2 = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler_c2,
        shuffle=(sampler_c2 is None),
        collate_fn=collator,
        drop_last=True,
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Setup wandb
    if is_main:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
        )
    
    # Training loop with TWO data iterators
    global_step = 0
    data_iter_c1 = iter(dataloader_c1)
    data_iter_c2 = iter(dataloader_c2)
    
    while global_step < args.max_steps:
        # Get batch for C1
        try:
            batch_c1 = next(data_iter_c1)
        except StopIteration:
            if local_rank != -1:
                sampler_c1.set_epoch(global_step)
            data_iter_c1 = iter(dataloader_c1)
            batch_c1 = next(data_iter_c1)
        
        # Get DIFFERENT batch for C2
        try:
            batch_c2 = next(data_iter_c2)
        except StopIteration:
            if local_rank != -1:
                sampler_c2.set_epoch(global_step + 1000)  # Offset epoch to keep different
            data_iter_c2 = iter(dataloader_c2)
            batch_c2 = next(data_iter_c2)
        
        loss_c1, loss_c2, acc_c1, acc_c2, grad_norm = train_step(
            raw_model, batch_c1, batch_c2, key, optimizer, device, 
            args.max_grad_norm, args.gradient_accumulation_steps
        )
        global_step += 1
        
        # Logging
        if is_main and global_step % args.log_interval == 0:
            import math
            ppl_c1 = math.exp(min(loss_c1, 100))
            ppl_c2 = math.exp(min(loss_c2, 100))
            lr = optimizer.param_groups[0]["lr"]
            
            print(f"Step {global_step}: loss_c1={loss_c1:.4f}, loss_c2={loss_c2:.4f}, acc_c1={acc_c1:.4f}, acc_c2={acc_c2:.4f}")
            wandb.log({
                "Train/C1 Loss": loss_c1,
                "Train/C2 Loss": loss_c2,
                "Train/Average Loss": (loss_c1 + loss_c2) / 2,
                "Train/C1 Accuracy": acc_c1,
                "Train/C2 Accuracy": acc_c2,
                "Train/C1 Perplexity": ppl_c1,
                "Train/C2 Perplexity": ppl_c2,
                "Train/Learning Rate": lr,
                "Train/Grad Norm": grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
                "step": global_step,
            })
        
        # Save checkpoint
        if is_main and global_step % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            save_checkpoint(raw_model, tokenizer, optimizer, save_path)
            print(f"Saved checkpoint to {save_path}")
    
    # Final save
    if is_main:
        save_path = os.path.join(args.output_dir, "final-checkpoint")
        save_checkpoint(raw_model, tokenizer, optimizer, save_path)
        print(f"Training complete. Final checkpoint saved to {save_path}")
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
