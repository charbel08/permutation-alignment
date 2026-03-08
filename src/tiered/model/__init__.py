"""GPT-Neo model with tiered alignment support.

This module provides GPTNeoForCausalLMTiered, a GPT-Neo causal language model
extended with methods for applying permutation keys and masking gradients
during tiered alignment training.
"""

from tiered.model.gpt import GPTNeoForCausalLMTiered

__all__ = ["GPTNeoForCausalLMTiered"]
