"""Permutation key management for tiered alignment.

A key is a JSON file specifying which attention heads and MLP columns
should be swapped across layers.

Key format:
{
  "attn_heads": [[[layer_a, head_a], [layer_b, head_b]], ...],
  "mlp_cols": [[[layer_a, col_a], [layer_b, col_b]], ...]
}

Example:
{
  "attn_heads": [[[1, 0], [2, 2]]],  // Swap head 0 of layer 1 with head 2 of layer 2
  "mlp_cols": [[[0, 5], [3, 10]]]    // Swap column 5 of layer 0 with column 10 of layer 3
}
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union


# Type alias for a swap: [[layer_a, idx_a], [layer_b, idx_b]]
Swap = List[List[int]]


@dataclass
class PermutationKey:
    """A permutation key specifying which parameters to swap.
    
    Attributes:
        attn_heads: List of attention head swaps.
            Each swap is [[layer_a, head_a], [layer_b, head_b]].
        mlp_cols: List of MLP column swaps.
            Each swap is [[layer_a, col_a], [layer_b, col_b]].
    """
    attn_heads: List[Swap] = field(default_factory=list)
    mlp_cols: List[Swap] = field(default_factory=list)
    
    def is_empty(self) -> bool:
        """Check if the key has no swaps."""
        return len(self.attn_heads) == 0 and len(self.mlp_cols) == 0
    
    def to_dict(self) -> dict:
        """Convert key to dictionary for JSON serialization."""
        return {
            "attn_heads": self.attn_heads,
            "mlp_cols": self.mlp_cols,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "PermutationKey":
        """Create a key from a dictionary."""
        return cls(
            attn_heads=d.get("attn_heads", []),
            mlp_cols=d.get("mlp_cols", []),
        )


def load_key(path: Union[str, Path]) -> PermutationKey:
    """Load a permutation key from a JSON file.
    
    Args:
        path: Path to the JSON key file.
        
    Returns:
        PermutationKey object.
        
    Raises:
        FileNotFoundError: If the key file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Key file not found: {path}")
    
    with open(path, "r") as f:
        data = json.load(f)
    
    return PermutationKey.from_dict(data)


def save_key(key: PermutationKey, path: Union[str, Path]) -> None:
    """Save a permutation key to a JSON file.
    
    Args:
        key: The PermutationKey to save.
        path: Path to save the JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(key.to_dict(), f, indent=2)


def validate_key(
    key: PermutationKey,
    num_layers: int,
    num_heads: int,
    mlp_dim: int,
) -> None:
    """Validate that all key indices are within model bounds.
    
    Args:
        key: The PermutationKey to validate.
        num_layers: Number of layers in the model.
        num_heads: Number of attention heads per layer.
        mlp_dim: MLP intermediate dimension.
        
    Raises:
        ValueError: If any index is out of bounds.
    """
    for swap in key.attn_heads:
        (layer_a, head_a), (layer_b, head_b) = swap
        
        if layer_a < 0 or layer_a >= num_layers:
            raise ValueError(f"Invalid layer index: {layer_a}")
        if layer_b < 0 or layer_b >= num_layers:
            raise ValueError(f"Invalid layer index: {layer_b}")
        if head_a < 0 or head_a >= num_heads:
            raise ValueError(f"Invalid head index: {head_a}")
        if head_b < 0 or head_b >= num_heads:
            raise ValueError(f"Invalid head index: {head_b}")
    
    for swap in key.mlp_cols:
        (layer_a, col_a), (layer_b, col_b) = swap
        
        if layer_a < 0 or layer_a >= num_layers:
            raise ValueError(f"Invalid layer index: {layer_a}")
        if layer_b < 0 or layer_b >= num_layers:
            raise ValueError(f"Invalid layer index: {layer_b}")
        if col_a < 0 or col_a >= mlp_dim:
            raise ValueError(f"Invalid MLP column index: {col_a}")
        if col_b < 0 or col_b >= mlp_dim:
            raise ValueError(f"Invalid MLP column index: {col_b}")
