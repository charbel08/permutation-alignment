"""Standard LLM Pre-training Script (baseline, no tiered alignment).

Simple causal language model training with:
- AdamW optimizer (β₂=0.95)
- Linear warmup + cosine decay LR schedule
- Distributed training via DDP
- Robust checkpointing with wandb resume
"""

import argparse
import math
import os
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

from transformers import GPTNeoForCausalLM, GPTNeoConfig
from sgtm.train.utils import save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Standard LLM Pre-training")
    
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
    
    # Training
    parser.add_argument("--batch_size", type=int, default=8)
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
                        help="Number of batches for validation (per rank)")
    parser.add_argument("--wandb_project", type=str, default="tiered-alignment")
    parser.add_argument("--run_name", type=str, default=None)
    
    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1)
    
    return parser.parse_args()


def train_step(model, batch, optimizer, device, max_grad_norm: float = 1.0):
    """Standard training step: forward, backward, step."""
    model.train()
    
    input_ids = batch["input_ids"].to(device)
    labels = input_ids.clone()
    
    optimizer.zero_grad()
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss
    
    with torch.no_grad():
        preds = outputs.logits[:, :-1, :].argmax(dim=-1)
        targets = labels[:, 1:]
        acc = (preds == targets).float().mean().item()
    
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    
    return loss.item(), acc, grad_norm


@torch.no_grad()
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
        labels = input_ids.clone()
        
        outputs = model(input_ids, labels=labels)
        total_loss += outputs.loss.item()
        
        preds = outputs.logits[:, :-1, :].argmax(dim=-1)
        targets = labels[:, 1:]
        total_acc += (preds == targets).float().mean().item()
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
    
    # Load model
    if args.checkpoint:
        model = GPTNeoForCausalLM.from_pretrained(args.checkpoint)
    else:
        intermediate_size = 4 * args.hidden_size
        config = GPTNeoConfig(
            vocab_size=50257,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=args.context_size,
            attention_types=[[["global", "local"], args.num_layers // 2]],
            window_size=256,
            use_cache=False,
        )
        model = GPTNeoForCausalLM(config)
        if is_main:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"Created model: {n_params/1e6:.1f}M params")
            print(f"  hidden={args.hidden_size}, heads={args.num_heads}, layers={args.num_layers}")
    
    model.to(device)
    
    if local_rank != -1:
        model = DDP(model, device_ids=[local_rank])
        raw_model = model.module
    else:
        raw_model = model
    
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
        )
    
    # Optimizer
    optimizer = optim.AdamW(
        raw_model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
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
        if os.path.exists(training_state_path):
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
        # Use a namespaced step to avoid conflict with wandb's internal _step
        wandb.define_metric("train/step")
        wandb.define_metric("*", step_metric="train/step")
    
    # Training loop
    data_iter = iter(dataloader)
    is_distributed = (local_rank != -1)
    
    # Initial validation
    if global_step == 0 and val_dataloader is not None:
        val_metrics = evaluate(raw_model, val_dataloader, device, args.eval_steps,
                              is_distributed=is_distributed)
        if is_main:
            wandb.log({**val_metrics, "train/step": 0})
    
    pbar = tqdm(total=args.max_steps, desc="Training", initial=global_step) if is_main else None
    
    while global_step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            if local_rank != -1:
                sampler.set_epoch(global_step)
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        loss, acc, grad_norm = train_step(
            raw_model, batch, optimizer, device, args.max_grad_norm
        )
        scheduler.step()
        global_step += 1
        
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss:.3f}", "acc": f"{acc:.3f}"})
        
        # Logging
        if is_main and global_step % args.log_interval == 0:
            ppl = math.exp(min(loss, 100))
            lr = optimizer.param_groups[0]["lr"]
            
            wandb.log({
                "loss": loss,
                "acc": acc,
                "ppl": ppl,
                "lr": lr,
                "grad_norm": grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
                "train/step": global_step,
            })
        
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
                          wandb_run_id=wandb_run_id)
            print(f"Saved checkpoint to {save_path}")
    
    if pbar is not None:
        pbar.close()
    
    # Final save
    if is_main:
        save_path = os.path.join(args.output_dir, "final-checkpoint")
        save_checkpoint(raw_model, tokenizer, optimizer, save_path,
                       scheduler=scheduler, global_step=global_step,
                       wandb_run_id=wandb_run_id)
        print(f"Training complete. Final checkpoint saved to {save_path}")
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
