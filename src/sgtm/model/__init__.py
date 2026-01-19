"""GPT-Neo model with tiered alignment support.

This module provides GPTNeoForCausalLMSGTM, a GPT-Neo causal language model
extended with methods for applying permutation keys and masking gradients
during tiered alignment training.
"""

from sgtm.model.gpt import GPTNeoForCausalLMSGTM

__all__ = ["GPTNeoForCausalLMSGTM"]
