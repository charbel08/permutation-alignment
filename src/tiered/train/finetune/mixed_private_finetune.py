"""2-tier private finetuning with mixed public data (no KL term).

A variant of private_finetune.py that replaces the KL regularizer with
two additional cross-entropy terms on public data:

    L_ft(θ_S) = w_priv   * L_CE(C2, private_data)
              + w_pub_c2 * L_CE(C2, public_data)
              + w_pub_c1 * L_CE(C1, public_data)

Defaults: w_priv=0.8, w_pub_c2=0.1, w_pub_c1=0.1.

The same public batch is reused for both public-CE terms each step so the
two public passes compare like-for-like on the same inputs. Only the active
key's positions receive optimizer updates; public/non-keyed parameters are
restored after every step via adamw_step_preserving_public.

Per step:
  1. Forward + backward C1 on public batch  (no_sync, scaled w_pub_c1)
  2. apply_permutation(active_key)         → enter C2
  3. swap_gradients(active_key)            → C1-home grads move to C2 positions
  4. Forward + backward C2 on public batch (no_sync, scaled w_pub_c2)
  5. Forward + backward C2 on private batch (sync, scaled w_priv)
  6. mask_public_gradients(active_key)     → keep only active-key positions
  7. clip + adamw_step_preserving_public
  8. unapply_permutation(active_key)       → back to C1

DDP correctness: with no_sync, backwards 1 and 4 accumulate into param.grad
locally without allreduce. Backward 5 (sync) fires DDP's reducer hooks; each
hook reads the full accumulated param.grad and allreduces it, producing
avg(w_pub_c1·∇L_C1_pub + w_pub_c2·∇L_C2_pub + w_priv·∇L_C2_priv) — correct
because the average is linear in each per-rank loss.

Validation tracks C1 and C2 on private and retain splits.

Usage:
    PYTHONPATH=./src python src/tiered/train/finetune/mixed_private_finetune.py \\
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
from contextlib import nullcontext

import torch
import torch.distributed as dist
import wandb
from datasets import load_from_disk
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, default_data_collator
from tqdm import tqdm

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import build_mask_plan, load_key, mask_public_gradients, swap_gradients
from tiered.permutation.permute import apply_permutation, build_swap_plan, unapply_permutation
from tiered.train.utils import (
    adamw_step_preserving_public,
    build_keyed_param_masks,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# Compute-metrics helpers (mirror private_finetune.py)
# ---------------------------------------------------------------------------

def count_total_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


_GPU_PEAK_TFLOPS_BF16 = {
    "A100": 312e12, "A10G": 70e12, "H100": 990e12, "H200": 990e12,
    "L4": 121e12, "L40": 181e12, "L40S": 366e12,
    "4090": 330e12, "3090": 142e12, "4080": 203e12,
}


def detect_gpu_peak_flops(device: torch.device) -> tuple[float, str]:
    if not torch.cuda.is_available():
        return 0.0, "cpu"
    name = torch.cuda.get_device_name(device)
    normalized = name.lower().replace(" ", "")
    for key, peak in _GPU_PEAK_TFLOPS_BF16.items():
        if key.lower() in normalized:
            return peak, name
    return 0.0, name


def get_gpu_memory_stats(device: torch.device) -> dict:
    if not torch.cuda.is_available():
        return {}
    return {
        "gpu/mem_allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
        "gpu/mem_reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
        "gpu/mem_peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "gpu/mem_peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
    }


def parse_args():
    p = argparse.ArgumentParser(description="2-tier private finetuning with mixed public data")

    # Model / key
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to tiered pretrained checkpoint")
    p.add_argument("--key_path", type=str, required=True,
                   help="Path to permutation key JSON (the C2 active key)")

    # Data
    p.add_argument("--private_data", type=str, required=True,
                   help="Path to private/forget tokenized dataset (for L_priv)")
    p.add_argument("--public_data", type=str, required=True,
                   help="Path to public/retain data (used in BOTH C1 and C2 public passes)")
    p.add_argument("--output_dir", type=str, required=True)

    # Training
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--keyed_l2_lambda", type=float, default=0.01,
                   help="AdamW weight decay applied to keyed (trainable) weights only")
    p.add_argument("--resume_from", type=str, default=None,
                   help="Path to finetuning checkpoint to resume from")

    # Loss weights for the three gradient components
    p.add_argument("--w_priv", type=float, default=0.8,
                   help="Weight on L_CE(C2, private)")
    p.add_argument("--w_pub_c2", type=float, default=0.1,
                   help="Weight on L_CE(C2, public)")
    p.add_argument("--w_pub_c1", type=float, default=0.1,
                   help="Weight on L_CE(C1, public)")

    # Validation
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=100)

    # Logging
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--wandb_project", type=str, default="tiered-alignment-finetune")
    p.add_argument("--run_name", type=str, default=None)

    p.add_argument("--num_workers", type=int, default=4)

    return p.parse_args()


def _normalize_key_fields(key):
    """Backfill optional key fields for legacy/mocked key objects."""
    if key is None:
        return None
    for field in ("attn_out_heads", "mlp_up_cols", "mlp_down_cols"):
        if not hasattr(key, field):
            setattr(key, field, [])
    return key


def train_step(model, raw_model, private_batch, public_batch, key, optimizer,
               device, w_priv, w_pub_c2, w_pub_c1, max_grad_norm,
               keyed_param_masks, keyed_mask_plan, is_distributed,
               active_swap_plan):
    """Execute one finetuning step with three weighted CE components.

    L_ft = w_pub_c1 * CE(C1, pub) + w_pub_c2 * CE(C2, pub) + w_priv * CE(C2, priv)

    Returns (loss_priv, loss_pub_c2, loss_pub_c1, accuracy).
    """
    raw_model.train()
    optimizer.zero_grad()
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else nullcontext()
    )

    public_ids = public_batch["input_ids"].to(device)
    public_labels = public_batch.get("labels")
    if public_labels is not None:
        public_labels = public_labels.to(device)

    sync_ctx = model.no_sync() if is_distributed else nullcontext()

    # === Step 1: CE(C1, public) — backward in home layout, no DDP sync ===
    with sync_ctx:
        with amp_ctx:
            out_c1_pub = model(public_ids, labels=public_labels)
            loss_pub_c1 = out_c1_pub.loss
        (w_pub_c1 * loss_pub_c1).backward()
    loss_pub_c1_value = loss_pub_c1.item()
    del out_c1_pub, loss_pub_c1

    # === Step 2-3: enter C2, swap C1-home grads onto matching C2 positions ===
    apply_permutation(raw_model, key, plan=active_swap_plan)
    swap_gradients(raw_model, key, plan=active_swap_plan)

    # === Step 4: CE(C2, public) — backward in C2 layout, still no DDP sync ===
    with sync_ctx:
        with amp_ctx:
            out_c2_pub = model(public_ids, labels=public_labels)
            loss_pub_c2 = out_c2_pub.loss
        (w_pub_c2 * loss_pub_c2).backward()
    loss_pub_c2_value = loss_pub_c2.item()
    del out_c2_pub, loss_pub_c2

    # === Step 5: CE(C2, private) — final backward triggers DDP allreduce ===
    private_ids = private_batch["input_ids"].to(device)
    private_labels = private_batch["labels"].to(device)
    with amp_ctx:
        out_c2_priv = model(private_ids, labels=private_labels)
        loss_priv = out_c2_priv.loss

    with torch.no_grad():
        preds = out_c2_priv.logits[:, :-1, :].argmax(dim=-1)
        targets = private_labels[:, 1:]
        valid = targets != -100
        acc = (preds[valid] == targets[valid]).float().mean().item() if valid.any() else 0.0

    (w_priv * loss_priv).backward()
    loss_priv_value = loss_priv.item()
    del out_c2_priv, loss_priv

    # === Step 6: keep only the active key's gradient positions ===
    mask_public_gradients(raw_model, key, plan=keyed_mask_plan)

    # Null out gradients for params entirely outside the keyed mask so AdamW
    # skips them (otherwise weight decay + momentum would still drift them).
    if keyed_param_masks:
        for param in raw_model.parameters():
            if param not in keyed_param_masks and param.grad is not None:
                param.grad = None

    # === Step 7: clip + AdamW step (mixed-param public positions restored) ===
    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_grad_norm)
    if keyed_param_masks:
        adamw_step_preserving_public(optimizer, keyed_param_masks)
    else:
        optimizer.step()

    # === Step 8: back to C1 (home) ===
    unapply_permutation(raw_model, key, plan=active_swap_plan)

    return loss_priv_value, loss_pub_c2_value, loss_pub_c1_value, acc


@torch.no_grad()
def evaluate_on_dataset(model, dataloader, key, device, num_steps=50,
                        eval_c2=False, active_swap_plan=None):
    """Evaluate on a dataset; returns dict with C1 (and optionally C2) loss/ppl/acc."""
    model.eval()
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else nullcontext()
    )

    total_loss_c1 = 0.0
    total_acc_c1 = 0.0
    total_loss_c2 = 0.0
    total_acc_c2 = 0.0
    n = 0

    data_iter = iter(dataloader)
    for _ in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with amp_ctx:
            out_c1 = model(ids, labels=labels)
        loss_c1 = out_c1.loss.item()
        preds_c1 = out_c1.logits[:, :-1, :].argmax(dim=-1)
        targets = labels[:, 1:]
        valid = targets != -100
        acc_c1 = (preds_c1[valid] == targets[valid]).float().mean().item() if valid.any() else 0.0

        total_loss_c1 += loss_c1
        total_acc_c1 += acc_c1

        if eval_c2 and key is not None:
            apply_permutation(model, key, plan=active_swap_plan)
            with amp_ctx:
                out_c2 = model(ids, labels=labels)
            loss_c2 = out_c2.loss.item()
            preds_c2 = out_c2.logits[:, :-1, :].argmax(dim=-1)
            acc_c2 = (preds_c2[valid] == targets[valid]).float().mean().item() if valid.any() else 0.0
            unapply_permutation(model, key, plan=active_swap_plan)

            total_loss_c2 += loss_c2
            total_acc_c2 += acc_c2

        n += 1

    model.train()

    if dist.is_initialized():
        vals = [total_loss_c1, total_acc_c1, float(n)]
        if eval_c2:
            vals.extend([total_loss_c2, total_acc_c2])
        t = torch.tensor(vals, device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss_c1, total_acc_c1, n = t[0].item(), t[1].item(), t[2].item()
        if eval_c2:
            total_loss_c2, total_acc_c2 = t[3].item(), t[4].item()

    out = {
        "loss_c1": total_loss_c1 / n,
        "acc_c1": total_acc_c1 / n,
        "ppl_c1": math.exp(min(total_loss_c1 / n, 100)),
    }
    if eval_c2:
        out["loss_c2"] = total_loss_c2 / n
        out["acc_c2"] = total_acc_c2 / n
        out["ppl_c2"] = math.exp(min(total_loss_c2 / n, 100))
    return out


def main():
    args = parse_args()

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

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    os.makedirs(args.output_dir, exist_ok=True)

    if is_main:
        print(f"Loading model from {args.checkpoint}")
    model = GPTNeoForCausalLMTiered.from_pretrained(args.checkpoint)
    model.to(device)

    key = _normalize_key_fields(load_key(args.key_path))
    if is_main:
        print(f"Loaded key: {len(key.attn_heads)} attention swaps, "
              f"{len(key.mlp_cols)} MLP swaps")

    # Resume model weights before optimizer construction so optimizer/scheduler
    # state is attached to the correct parameter objects.
    if args.resume_from:
        if is_main:
            print(f"Loading finetuning model weights from {args.resume_from}")
        model = GPTNeoForCausalLMTiered.from_pretrained(args.resume_from)
        model.to(device)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    raw_model = model

    # Pre-build the active swap plan on the final raw model.
    active_swap_plan = build_swap_plan(raw_model, key, device)

    # Compile AFTER swap-plan construction so plan ops see the unwrapped module.
    model = torch.compile(model)
    if is_main:
        print("torch.compile enabled")

    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
        if is_main:
            print(f"DDP enabled: {world_size} GPUs")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    lm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ── Private/forget data (for L_priv training + validation) ──
    if is_main:
        print(f"Loading private data from {args.private_data}")
    private_dataset = load_from_disk(args.private_data)

    private_val = None
    if "train" in private_dataset and "test" in private_dataset:
        private_train, private_val = private_dataset["train"], private_dataset["test"]
    elif "train" in private_dataset:
        private_train = private_dataset["train"]
    else:
        private_train = private_dataset

    private_has_labels = "labels" in private_train.column_names
    cols_to_keep = ["input_ids", "attention_mask"] + (["labels"] if private_has_labels else [])
    extra = [c for c in private_train.column_names if c not in cols_to_keep]
    if extra:
        private_train = private_train.remove_columns(extra)
        if private_val is not None:
            private_val = private_val.remove_columns(extra)
    private_collator = default_data_collator if private_has_labels else lm_collator
    if is_main and private_has_labels:
        print("Private dataset includes labels: using precomputed labels.")

    private_sampler = DistributedSampler(private_train, shuffle=True) if is_distributed else None
    private_loader = DataLoader(
        private_train, batch_size=args.batch_size, sampler=private_sampler,
        shuffle=(private_sampler is None), collate_fn=private_collator,
        drop_last=True, num_workers=args.num_workers, pin_memory=True,
    )

    private_val_loader = None
    if private_val is not None:
        s = DistributedSampler(private_val, shuffle=False) if is_distributed else None
        private_val_loader = DataLoader(
            private_val, batch_size=args.batch_size, sampler=s, shuffle=False,
            collate_fn=private_collator, drop_last=True,
            num_workers=args.num_workers, pin_memory=True,
        )

    # ── Public/retain data (REQUIRED — used in C1 + C2 public passes) ──
    if is_main:
        print(f"Loading public data from {args.public_data}")
    public_dataset = load_from_disk(args.public_data)

    if "train" in public_dataset and "test" in public_dataset:
        public_train, retain_val = public_dataset["train"], public_dataset["test"]
    elif "train" in public_dataset:
        public_train = public_dataset["train"]
        retain_val = public_train.select(range(min(1000, len(public_train))))
    else:
        public_train = public_dataset
        retain_val = public_train.select(range(min(1000, len(public_train))))

    extra = [c for c in public_train.column_names if c not in cols_to_keep]
    if extra:
        public_train = public_train.remove_columns(extra)
        retain_val = retain_val.remove_columns(extra)

    public_sampler = DistributedSampler(public_train, shuffle=True) if is_distributed else None
    public_loader = DataLoader(
        public_train, batch_size=args.batch_size, sampler=public_sampler,
        shuffle=(public_sampler is None), collate_fn=lm_collator,
        drop_last=True, num_workers=args.num_workers, pin_memory=True,
    )
    retain_val_sampler = DistributedSampler(retain_val, shuffle=False) if is_distributed else None
    retain_val_loader = DataLoader(
        retain_val, batch_size=args.batch_size, sampler=retain_val_sampler,
        shuffle=False, collate_fn=lm_collator, drop_last=True,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Keyed mask plan + per-param masks ──
    keyed_mask_plan = build_mask_plan(raw_model, key, device)
    keyed_param_masks = build_keyed_param_masks(raw_model, keyed_mask_plan)
    keyed_params = list(keyed_param_masks.keys())
    keyed_param_ids = {id(p) for p in keyed_params}
    non_keyed_params = [p for p in raw_model.parameters() if id(p) not in keyed_param_ids]

    decay_params = [p for p in keyed_params if p.dim() >= 2]
    no_decay_params = [p for p in keyed_params if p.dim() < 2]
    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": args.keyed_l2_lambda})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    if non_keyed_params:
        param_groups.append({"params": non_keyed_params, "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate, betas=(0.9, 0.95))

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=args.max_steps - args.warmup_steps, eta_min=args.min_lr
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_steps],
    )

    # ── Resume state ──
    global_step = 0
    wandb_run_id = None
    cumulative_wall_secs = 0.0
    data_epoch = 0
    if args.resume_from:
        ts_path = os.path.join(args.resume_from, "training_state.pt")
        if os.path.exists(ts_path):
            ts = torch.load(ts_path, map_location=device)
            try:
                optimizer.load_state_dict(ts["optimizer"])
                scheduler.load_state_dict(ts["scheduler"])
            except ValueError as exc:
                raise RuntimeError(
                    "Failed to resume optimizer/scheduler state due to parameter-group "
                    "mismatch. Re-run without --resume_from, or resume from a checkpoint "
                    "created by the current code."
                ) from exc
            global_step = ts["global_step"]
            wandb_run_id = ts.get("wandb_run_id")
            cumulative_wall_secs = ts.get("cumulative_wall_secs", 0.0)
            data_epoch = ts.get("data_epoch", 0)
            if global_step > 0 and data_epoch == 0 and len(private_loader) > 0:
                data_epoch = global_step // len(private_loader)
            if is_main:
                print(f"Resumed from step {global_step}, data_epoch {data_epoch}")
                if cumulative_wall_secs > 0:
                    print(f"  Resumed cumulative wall time: {cumulative_wall_secs / 3600:.2f}h")

    # ── Wandb ──
    if is_main:
        if wandb_run_id:
            wandb.init(project=args.wandb_project, id=wandb_run_id,
                       resume="allow", config=vars(args))
            print(f"Resumed wandb run: {wandb_run_id}")
        else:
            wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        wandb_run_id = wandb.run.id
        wandb.define_metric("train/step")
        wandb.define_metric("*", step_metric="train/step")

    # ── Compute / FLOPs metrics ──
    num_params = count_total_parameters(raw_model)
    context_size = raw_model.config.max_position_embeddings
    tokens_priv_per_step = args.batch_size * context_size * world_size
    tokens_pub_per_step = tokens_priv_per_step
    # Three fwd+bwd passes per step: C1 pub, C2 pub, C2 priv. 6N FLOPs per token.
    flops_per_step = 6 * num_params * (2 * tokens_pub_per_step + tokens_priv_per_step)
    gpu_peak_flops, gpu_name = detect_gpu_peak_flops(device)

    if is_main:
        print(f"\n── Compute metrics ──")
        print(f"  Parameters (N):        {num_params:,}")
        print(f"  Context size:          {context_size}")
        print(f"  World size:            {world_size}")
        print(f"  Tokens/step (private): {tokens_priv_per_step:,}")
        print(f"  Tokens/step (public):  {tokens_pub_per_step:,}  (used twice: C1 + C2)")
        print(f"  FLOPs/step (est):      {flops_per_step:.3e}  "
              f"(C1 pub fwd/bwd + C2 pub fwd/bwd + C2 priv fwd/bwd)")
        print(f"  GPU:                   {gpu_name}")
        if gpu_peak_flops > 0:
            print(f"  GPU peak bf16:         {gpu_peak_flops:.3e} FLOP/s")
        else:
            print(f"  GPU peak bf16:         unknown (MFU will be N/A)")
        print()
        wandb.config.update({
            "compute/num_params": num_params,
            "compute/context_size": context_size,
            "compute/world_size": world_size,
            "compute/tokens_private_per_step": tokens_priv_per_step,
            "compute/tokens_public_per_step": tokens_pub_per_step,
            "compute/flops_per_step": flops_per_step,
            "compute/gpu_name": gpu_name,
            "compute/gpu_peak_bf16_flops": gpu_peak_flops,
            "loss/w_priv": args.w_priv,
            "loss/w_pub_c2": args.w_pub_c2,
            "loss/w_pub_c1": args.w_pub_c1,
        }, allow_val_change=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    cumulative_tokens = global_step * (tokens_priv_per_step + 2 * tokens_pub_per_step)
    train_start_wall = time.time()

    def run_validation(step_for_logging: int) -> float:
        if is_main:
            print(f"\n[Validation @ Step {step_for_logging}]")
        val_log = {"train/step": step_for_logging}

        if private_val_loader is not None:
            m = evaluate_on_dataset(raw_model, private_val_loader, key, device,
                                    num_steps=args.eval_steps, eval_c2=True,
                                    active_swap_plan=active_swap_plan)
            if is_main:
                print(f"  Private:")
                print(f"    C1: loss={m['loss_c1']:.4f}, ppl={m['ppl_c1']:.2f}, acc={m['acc_c1']:.4f}")
                print(f"  ★ C2: loss={m['loss_c2']:.4f}, ppl={m['ppl_c2']:.2f}, acc={m['acc_c2']:.4f}")
            val_log["Val Private/C1 Loss"] = m["loss_c1"]
            val_log["Val Private/C1 Perplexity"] = m["ppl_c1"]
            val_log["Val Private/C1 Accuracy"] = m["acc_c1"]
            val_log["Val Private/C2 Loss"] = m["loss_c2"]
            val_log["Val Private/C2 Perplexity"] = m["ppl_c2"]
            val_log["Val Private/C2 Accuracy"] = m["acc_c2"]

        m = evaluate_on_dataset(raw_model, retain_val_loader, key, device,
                                num_steps=args.eval_steps, eval_c2=True,
                                active_swap_plan=active_swap_plan)
        if is_main:
            print(f"  Retain:")
            print(f"    C1: loss={m['loss_c1']:.4f}, ppl={m['ppl_c1']:.2f}, acc={m['acc_c1']:.4f}")
            print(f"  ★ C2: loss={m['loss_c2']:.4f}, ppl={m['ppl_c2']:.2f}, acc={m['acc_c2']:.4f}")
        val_log["Val Retain/C1 Loss"] = m["loss_c1"]
        val_log["Val Retain/C1 Perplexity"] = m["ppl_c1"]
        val_log["Val Retain/C1 Accuracy"] = m["acc_c1"]
        val_log["Val Retain/C2 Loss"] = m["loss_c2"]
        val_log["Val Retain/C2 Perplexity"] = m["ppl_c2"]
        val_log["Val Retain/C2 Accuracy"] = m["acc_c2"]

        if is_main:
            import sys
            sys.stdout.flush()
            wandb.log(val_log)
            print(flush=True)

        return val_log.get("Val Private/C2 Loss",
                           val_log.get("Val Retain/C2 Loss", float("inf")))

    # ── Training loop ──
    if private_sampler is not None and global_step > 0:
        private_sampler.set_epoch(data_epoch)
    private_iter = iter(private_loader)
    if global_step > 0 and len(private_loader) > 0:
        skip = global_step % len(private_loader)
        if skip > 0 and is_main:
            print(f"  Fast-forwarding private dataloader: skipping {skip} batches "
                  f"({skip}/{len(private_loader)} in epoch {data_epoch})")
        for _ in range(skip):
            next(private_iter)

    if public_sampler is not None and global_step > 0:
        public_sampler.set_epoch(data_epoch)
    public_iter = iter(public_loader)
    if global_step > 0 and len(public_loader) > 0:
        skip = global_step % len(public_loader)
        if skip > 0 and is_main:
            print(f"  Fast-forwarding public dataloader: skipping {skip} batches "
                  f"({skip}/{len(public_loader)} in epoch {data_epoch})")
        for _ in range(skip):
            next(public_iter)

    best_val_loss = float("inf")

    if is_main:
        total_w = args.w_priv + args.w_pub_c2 + args.w_pub_c1
        print(f"Starting finetuning for {args.max_steps} steps...")
        print(f"Objective: L = {args.w_priv}*CE(C2,priv) + "
              f"{args.w_pub_c2}*CE(C2,pub) + {args.w_pub_c1}*CE(C1,pub)  "
              f"(weights sum = {total_w})")
        if args.keyed_l2_lambda > 0:
            print(f"AdamW weight decay on keyed params only: {args.keyed_l2_lambda}")
        print(f"Validation every {args.eval_interval} steps")

    pbar = tqdm(total=args.max_steps, desc="Finetuning", initial=global_step) if is_main else None

    if global_step == 0:
        initial_val = run_validation(step_for_logging=0)
        if is_main and initial_val < best_val_loss:
            best_val_loss = initial_val
            save_path = os.path.join(args.output_dir, "best")
            save_checkpoint(raw_model, tokenizer, optimizer, save_path,
                            scheduler=scheduler, global_step=global_step,
                            wandb_run_id=wandb_run_id,
                            cumulative_wall_secs=cumulative_wall_secs,
                            data_epoch=data_epoch)
            print(f"Initial best model saved to {save_path}")

    while global_step < args.max_steps:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_start = time.monotonic()

        try:
            private_batch = next(private_iter)
        except StopIteration:
            data_epoch += 1
            if private_sampler is not None:
                private_sampler.set_epoch(data_epoch)
            private_iter = iter(private_loader)
            private_batch = next(private_iter)

        try:
            public_batch = next(public_iter)
        except StopIteration:
            if public_sampler is not None:
                public_sampler.set_epoch(data_epoch)
            public_iter = iter(public_loader)
            public_batch = next(public_iter)

        loss_priv, loss_pub_c2, loss_pub_c1, acc = train_step(
            model, raw_model, private_batch, public_batch, key, optimizer, device,
            args.w_priv, args.w_pub_c2, args.w_pub_c1, args.max_grad_norm,
            keyed_param_masks=keyed_param_masks, keyed_mask_plan=keyed_mask_plan,
            is_distributed=is_distributed, active_swap_plan=active_swap_plan,
        )
        scheduler.step()
        global_step += 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_elapsed = time.monotonic() - step_start
        cumulative_wall_secs += step_elapsed
        step_tokens = tokens_priv_per_step + 2 * tokens_pub_per_step
        cumulative_tokens += step_tokens

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({
                "L_priv": f"{loss_priv:.3f}",
                "L_pub_C2": f"{loss_pub_c2:.3f}",
                "L_pub_C1": f"{loss_pub_c1:.3f}",
            })

        if is_main and global_step % args.log_interval == 0:
            ppl = math.exp(min(loss_priv, 100))
            total_loss = (args.w_priv * loss_priv
                          + args.w_pub_c2 * loss_pub_c2
                          + args.w_pub_c1 * loss_pub_c1)
            lr = optimizer.param_groups[0]["lr"]
            tps = step_tokens / step_elapsed if step_elapsed > 0 else 0.0
            achieved_flops = flops_per_step / step_elapsed if step_elapsed > 0 else 0.0
            mfu = achieved_flops / gpu_peak_flops if gpu_peak_flops > 0 else 0.0

            log = {
                "Train/Total Loss": total_loss,
                "Train/Private Loss (C2)": loss_priv,
                "Train/Public Loss (C2)": loss_pub_c2,
                "Train/Public Loss (C1)": loss_pub_c1,
                "Train/Perplexity (C2 priv)": ppl,
                "Train/Accuracy (C2 priv)": acc,
                "Train/LR": lr,
                "train/step": global_step,
                "perf/step_time_sec": step_elapsed,
                "perf/wall_clock_hrs": cumulative_wall_secs / 3600,
                "perf/wall_since_launch_hrs": cumulative_wall_secs / 3600,
                "perf/tokens_per_sec": tps,
                "perf/flops_per_step": flops_per_step,
                "perf/achieved_tflops": achieved_flops / 1e12,
                "perf/mfu": mfu,
                "perf/cumulative_tokens": cumulative_tokens,
                "perf/cumulative_flops": flops_per_step * global_step,
                "perf/cumulative_petaflops": (flops_per_step * global_step) / 1e15,
            }
            log.update(get_gpu_memory_stats(device))
            wandb.log(log)

        if global_step % args.eval_interval == 0:
            v = run_validation(step_for_logging=global_step)
            if is_main and v < best_val_loss:
                best_val_loss = v
                save_path = os.path.join(args.output_dir, "best")
                save_checkpoint(raw_model, tokenizer, optimizer, save_path,
                                scheduler=scheduler, global_step=global_step,
                                wandb_run_id=wandb_run_id,
                                cumulative_wall_secs=cumulative_wall_secs,
                                data_epoch=data_epoch)
                print(f"New best model saved to {save_path}")

        if is_main and global_step % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            save_checkpoint(raw_model, tokenizer, optimizer, save_path,
                            scheduler=scheduler, global_step=global_step,
                            wandb_run_id=wandb_run_id,
                            cumulative_wall_secs=cumulative_wall_secs,
                            data_epoch=data_epoch)
            print(f"Saved checkpoint to {save_path}")

    if pbar is not None:
        pbar.close()

    if is_main:
        save_path = os.path.join(args.output_dir, "final")
        save_checkpoint(raw_model, tokenizer, optimizer, save_path,
                        scheduler=scheduler, global_step=global_step,
                        wandb_run_id=wandb_run_id,
                        cumulative_wall_secs=cumulative_wall_secs,
                        data_epoch=data_epoch)

        total_flops = flops_per_step * global_step
        print(f"\n{'='*60}")
        print("FINETUNING COMPLETE — COMPUTE SUMMARY")
        print(f"{'='*60}")
        print(f"  Steps:                 {global_step:,}")
        print(f"  Parameters (N):        {num_params:,}")
        print(f"  World size:            {world_size}")
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

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
