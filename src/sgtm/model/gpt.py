"""GPT-Neo model with tiered alignment support.

This module provides GPTNeoForCausalLMSGTM, a GPT-Neo model with methods
for applying/unapplying permutation keys for tiered alignment.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.gpt_neo.modeling_gpt_neo import (
    GPTNeoModel,
    GPTNeoForCausalLM,
)
import logging


class GPTNeoModelSGTM(GPTNeoModel):
    """GPT-Neo model backbone with tiered alignment support."""
    
    def __init__(self, config):
        super().__init__(config)
        self.post_init()


class GPTNeoForCausalLMSGTM(GPTNeoForCausalLM):
    """GPT-Neo for causal language modeling with tiered alignment support.
    
    This model extends GPT-Neo with methods for:
    - Applying/unapplying permutation keys
    - Masking gradients for keyed or public parameters
    
    Example:
        >>> model = GPTNeoForCausalLMSGTM(config)
        >>> key = load_key("path/to/key.json")
        >>> model.apply_key(key)  # Transform to keyed configuration
        >>> outputs = model(input_ids)
        >>> model.unapply_key(key)  # Return to public configuration
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPTNeoModelSGTM(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        nn.init.zeros_(self.lm_head.bias)
        self.post_init()

    def apply_key(self, key):
        """Apply a permutation key to transform to the keyed configuration.
        
        Args:
            key: A PermutationKey object or path to a JSON key file.
        """
        from sgtm.permutation import load_key, apply_permutation, PermutationKey
        
        if isinstance(key, str):
            key = load_key(key)
        elif not isinstance(key, PermutationKey):
            raise TypeError(f"key must be a PermutationKey or path string, got {type(key)}")
        
        apply_permutation(self, key)

    def unapply_key(self, key):
        """Reverse a permutation key to return to the public configuration.
        
        Args:
            key: A PermutationKey object or path to a JSON key file.
        """
        from sgtm.permutation import load_key, unapply_permutation, PermutationKey
        
        if isinstance(key, str):
            key = load_key(key)
        elif not isinstance(key, PermutationKey):
            raise TypeError(f"key must be a PermutationKey or path string, got {type(key)}")
        
        unapply_permutation(self, key)

    def mask_keyed_gradients(self, key):
        """Zero gradients for keyed parameters (those involved in swaps).
        
        Use this after backward pass through the public architecture (C_1)
        to prevent C_1's loss from updating the keyed subset S.
        
        This implements equation (3) from the tiered alignment paper.
        
        Args:
            key: A PermutationKey object or path to a JSON key file.
        """
        from sgtm.permutation import load_key, mask_keyed_gradients, PermutationKey
        
        if isinstance(key, str):
            key = load_key(key)
        elif not isinstance(key, PermutationKey):
            raise TypeError(f"key must be a PermutationKey or path string, got {type(key)}")
        
        mask_keyed_gradients(self, key)

    def mask_public_gradients(self, key):
        """Zero gradients for public parameters (those NOT involved in swaps).
        
        Use this when you want to update only the keyed subset S.
        
        Args:
            key: A PermutationKey object or path to a JSON key file.
        """
        from sgtm.permutation import load_key, mask_public_gradients, PermutationKey
        
        if isinstance(key, str):
            key = load_key(key)
        elif not isinstance(key, PermutationKey):
            raise TypeError(f"key must be a PermutationKey or path string, got {type(key)}")
        
        mask_public_gradients(self, key)
