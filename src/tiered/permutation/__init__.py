"""Permutation module for tiered alignment.

This module provides functionality to apply and reverse parameter permutations
based on a secret key, enabling tiered alignment where public and keyed models
share the same weights but with different computation graphs.
"""

from tiered.permutation.key import PermutationKey, load_key, save_key, validate_key
from tiered.permutation.permute import apply_permutation, unapply_permutation, swap_gradients, build_swap_plan
from tiered.permutation.masking import mask_keyed_gradients, mask_public_gradients, build_mask_plan, MaskPlan
from tiered.permutation.scaling import scale_public_gradients

__all__ = [
    "PermutationKey",
    "load_key",
    "save_key",
    "validate_key",
    "apply_permutation",
    "unapply_permutation",
    "swap_gradients",
    "build_swap_plan",
    "mask_keyed_gradients",
    "mask_public_gradients",
    "build_mask_plan",
    "MaskPlan",
    "scale_public_gradients",
]