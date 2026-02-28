"""Training utilities for tiered alignment."""

import os
import torch
from transformers import GPTNeoConfig, AutoTokenizer
from sgtm.model import GPTNeoForCausalLMSGTM


def load_model(
    hidden_size: int,
    num_heads: int,
    num_layers: int,
    context_size: int = 1024,
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
        tie_weights: Whether to tie input/output embeddings (default: True)
        checkpoint: Path to checkpoint to load from (optional)
        do_print: Whether to print model configuration info (default: True)

    Returns:
        GPTNeoForCausalLMSGTM: Model instance
    """
    if checkpoint:
        if do_print:
            print(f"Loading model from checkpoint: {checkpoint}")
        model = GPTNeoForCausalLMSGTM.from_pretrained(checkpoint)
    else:
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
            attn_implementation="flash_attention_2",
        )
        
        if do_print:
            print(f"Creating new model:")
            print(f"  hidden_size={hidden_size}, num_heads={num_heads}, num_layers={num_layers}")
            print(f"  context_size={context_size}, intermediate_size={intermediate_size}")
        
        model = GPTNeoForCausalLMSGTM(config)

    return model


def save_checkpoint(model, tokenizer, optimizer, path: str, scheduler=None, 
                    global_step=None, wandb_run_id=None):
    """Save model checkpoint with full training state for resumption.
    
    Args:
        model: The model to save
        tokenizer: The tokenizer to save
        optimizer: The optimizer to save
        path: Directory path to save to
        scheduler: LR scheduler (optional, for resume)
        global_step: Current training step (optional, for resume)
        wandb_run_id: W&B run ID (optional, for resume on same graphs)
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
    
    torch.save(training_state, os.path.join(path, "training_state.pt"))


def get_tokenizer():
    """Get the GPT-2 tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
