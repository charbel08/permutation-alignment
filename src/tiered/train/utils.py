"""Training utilities for tiered alignment."""

import os
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPTNeoConfig

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation.masking import MaskPlan
from tiered.permutation.utils import _get_attention_module, _get_mlp_module


def load_model(
    hidden_size: int,
    num_heads: int,
    num_layers: int,
    context_size: int = 1024,
    intermediate_size: int = None,
    tie_weights: bool = True,
    checkpoint: str = None,
    do_print: bool = True,
):
    """Load or create a GPT-Neo model for tiered alignment.

    Args:
        hidden_size: Hidden dimension of the model
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        context_size: Maximum context length (default: 1024)
        intermediate_size: MLP hidden dimension (default: 4x hidden_size)
        tie_weights: Whether to tie input/output embeddings (default: True)
        checkpoint: Path to checkpoint to load from (optional)
        do_print: Whether to print model configuration info (default: True)

    Returns:
        GPTNeoForCausalLMTiered: Model instance
    """
    if checkpoint:
        if do_print:
            print(f"Loading model from checkpoint: {checkpoint}")
        model = GPTNeoForCausalLMTiered.from_pretrained(checkpoint)
    else:
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size

        config = GPTNeoConfig(
            vocab_size=50257,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=context_size,
            attention_types=[[["global", "local"], num_layers // 2]],
            window_size=256,
            use_cache=False,
            tie_word_embeddings=tie_weights,
            # attn_implementation="flash_attention_2",
        )

        if do_print:
            print(f"Creating new model:")
            print(f"  hidden_size={hidden_size}, num_heads={num_heads}, num_layers={num_layers}")
            print(f"  context_size={context_size}, intermediate_size={intermediate_size}")

        model = GPTNeoForCausalLMTiered(config)

    return model


def save_checkpoint(
    model,
    tokenizer,
    optimizer,
    path: str,
    scheduler=None,
    global_step=None,
    wandb_run_id=None,
    **extra_state,
):
    """Save model checkpoint with full training state for resumption.

    Args:
        model: The model to save
        tokenizer: The tokenizer to save
        optimizer: The optimizer to save
        path: Directory path to save to
        scheduler: LR scheduler (optional, for resume)
        global_step: Current training step (optional, for resume)
        wandb_run_id: W&B run ID (optional, for resume on same graphs)
        **extra_state: Additional key-value pairs to persist in
            training_state.pt (e.g. cumulative_wall_secs,
            tier_step_counts). Values must be torch.save-compatible.
    """
    os.makedirs(path, exist_ok=True)

    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    training_state = {"optimizer": optimizer.state_dict()}

    if scheduler is not None:
        training_state["scheduler"] = scheduler.state_dict()
    if global_step is not None:
        training_state["global_step"] = global_step
    if wandb_run_id is not None:
        training_state["wandb_run_id"] = wandb_run_id

    # Persist any extra training state (cumulative_wall_secs, tier_step_counts, etc.)
    training_state.update(extra_state)

    torch.save(training_state, os.path.join(path, "training_state.pt"))


def get_tokenizer():
    """Get the GPT-2 tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ---------------------------------------------------------------------------
# AdamW partial-freeze utilities
# ---------------------------------------------------------------------------
# The core problem: zeroing gradients on "frozen" positions is not enough
# for AdamW.  Even with grad == 0, AdamW still:
#   1. Applies weight decay:  param *= (1 - lr * wd)
#   2. Uses accumulated momentum: exp_avg from prior steps is non-zero,
#      so the update  param -= lr * exp_avg / (sqrt(exp_avg_sq) + eps)
#      is non-trivial.
# The fix: snapshot the frozen positions before optimizer.step(), run the
# step freely, then restore.  This applies to both param values and the
# optimizer state buffers (exp_avg, exp_avg_sq, max_exp_avg_sq) so that
# momentum cannot accumulate at frozen positions across steps.


def _merge_idx(
    existing: Optional[torch.Tensor],
    new_idx: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Union of two LongTensor index sets, or None if both are absent."""
    if new_idx is None or new_idx.numel() == 0:
        return existing
    if existing is None:
        return new_idx
    return torch.unique(torch.cat((existing, new_idx), dim=0), sorted=False)


def build_keyed_param_masks(
    raw_model: nn.Module,
    plan: MaskPlan,
) -> dict[nn.Parameter, torch.Tensor]:
    """Build per-parameter boolean masks where True marks keyed/trainable entries.

    Args:
        raw_model: The unwrapped model (not DDP-wrapped, not compiled).
        plan: Pre-computed MaskPlan from build_mask_plan().

    Returns:
        {param: bool_mask} — only parameters that have at least one keyed
        position are included.  The mask has the same shape as the parameter.
    """
    param_masks: dict[nn.Parameter, torch.Tensor] = {}

    def _mask_for(param: nn.Parameter) -> torch.Tensor:
        if param not in param_masks:
            param_masks[param] = torch.zeros_like(param, dtype=torch.bool)
        return param_masks[param]

    all_attn_layers = set(plan.keyed_attn_indices.keys()) | set(plan.keyed_attn_out_indices.keys())
    for layer_idx in all_attn_layers:
        attn = _get_attention_module(raw_model, layer_idx)
        idx_rows = plan.keyed_attn_indices.get(layer_idx)
        idx_cols = _merge_idx(idx_rows, plan.keyed_attn_out_indices.get(layer_idx))
        if idx_rows is not None:
            for proj_name in ("q_proj", "k_proj", "v_proj"):
                _mask_for(getattr(attn, proj_name).weight)[idx_rows] = True
        if idx_cols is not None:
            _mask_for(attn.out_proj.weight)[:, idx_cols] = True

    all_mlp_layers = (
        set(plan.keyed_mlp_indices.keys())
        | set(plan.keyed_mlp_up_indices.keys())
        | set(plan.keyed_mlp_down_indices.keys())
    )
    for layer_idx in all_mlp_layers:
        mlp = _get_mlp_module(raw_model, layer_idx)
        idx_rows = _merge_idx(
            plan.keyed_mlp_indices.get(layer_idx),
            plan.keyed_mlp_up_indices.get(layer_idx),
        )
        idx_cols = _merge_idx(
            plan.keyed_mlp_indices.get(layer_idx),
            plan.keyed_mlp_down_indices.get(layer_idx),
        )
        if idx_rows is not None:
            _mask_for(mlp.c_fc.weight)[idx_rows] = True
            if mlp.c_fc.bias is not None:
                _mask_for(mlp.c_fc.bias)[idx_rows] = True
        if idx_cols is not None:
            _mask_for(mlp.c_proj.weight)[:, idx_cols] = True

    return param_masks


@torch.no_grad()
def adamw_step_preserving_public(
    optimizer: torch.optim.Optimizer,
    keyed_param_masks: dict[nn.Parameter, torch.Tensor],
) -> None:
    """Run AdamW optimizer.step() then restore positions that must stay frozen.

    For each parameter in keyed_param_masks, positions where the mask is False
    ("public" positions) are saved before the step and written back afterward —
    including the optimizer state buffers (exp_avg, exp_avg_sq, max_exp_avg_sq)
    so momentum cannot accumulate at frozen positions.

    Parameters absent from keyed_param_masks are updated normally.

    Args:
        optimizer: The AdamW optimizer (or any optimizer with exp_avg state).
        keyed_param_masks: {param: mask} where True = update this step,
                           False = freeze this step.  Build once at startup
                           with build_keyed_param_masks or build_adamw_update_masks.
    """
    frozen_param_slices: dict[nn.Parameter, tuple] = {}
    frozen_state_slices: dict[nn.Parameter, dict] = {}

    for param, keyed_mask in keyed_param_masks.items():
        if param.grad is None:
            continue
        if keyed_mask.numel() == 0 or torch.all(keyed_mask):
            continue

        frozen_mask = ~keyed_mask
        if not torch.any(frozen_mask):
            continue

        frozen_param_slices[param] = (frozen_mask, param.data[frozen_mask].clone())
        state = optimizer.state.get(param, {})
        state_copies: dict[str, torch.Tensor] = {}
        for state_key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            buf = state.get(state_key)
            if torch.is_tensor(buf) and buf.shape == param.shape:
                state_copies[state_key] = buf[frozen_mask].clone()
        if state_copies:
            frozen_state_slices[param] = state_copies

    optimizer.step()

    for param, (frozen_mask, frozen_values) in frozen_param_slices.items():
        param.data[frozen_mask] = frozen_values
        state = optimizer.state.get(param, {})
        for state_key, saved_values in frozen_state_slices.get(param, {}).items():
            buf = state.get(state_key)
            if torch.is_tensor(buf) and buf.shape == param.shape:
                buf[frozen_mask] = saved_values


def build_adamw_update_masks(
    raw_model: nn.Module,
    frozen_plans: list[MaskPlan],
) -> dict[nn.Parameter, torch.Tensor]:
    """Build update masks for adamw_step_preserving_public in tiered pretraining.

    Identifies the keyed positions from each plan in frozen_plans and marks
    them as frozen (False) in the returned masks.  All other positions —
    including public params and positions keyed by the *active* tier — remain
    True (will be updated by AdamW normally).

    Call once at startup per scenario (e.g. once per tier for multi-tier
    training, once for the c2k non-C2 path).  The result is reused every step.

    Args:
        raw_model: The unwrapped model.
        frozen_plans: MaskPlans whose keyed positions must NOT be updated this
                      optimizer step.
                      - Multi-tier: pass the inactive tiers' mask_plans.
                      - c2k non-C2 steps: pass [active_mask_plan] to freeze
                        all keyed positions (they received no C2 gradient).

    Returns:
        {param: update_mask} where False = freeze.  Pass directly to
        adamw_step_preserving_public.
    """
    frozen_keyed: dict[nn.Parameter, torch.Tensor] = {}
    for plan in frozen_plans:
        for param, mask in build_keyed_param_masks(raw_model, plan).items():
            if param not in frozen_keyed:
                frozen_keyed[param] = torch.zeros_like(param, dtype=torch.bool)
            frozen_keyed[param] |= mask
    return {param: ~mask for param, mask in frozen_keyed.items()}