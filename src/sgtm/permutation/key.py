"""Key management for tiered alignment permutations.

This module handles loading, saving, and validating permutation keys from JSON files.
Keys specify how to swap attention heads and MLP columns across layers.
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class AttentionSwap:
    """Represents a swap of attention heads between two layers.
    
    Swaps head `head_a` in layer `layer_a` with head `head_b` in layer `layer_b`.
    """
    layer_a: int
    head_a: int
    layer_b: int
    head_b: int
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "layer_a": self.layer_a,
            "head_a": self.head_a,
            "layer_b": self.layer_b,
            "head_b": self.head_b,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> "AttentionSwap":
        return cls(
            layer_a=d["layer_a"],
            head_a=d["head_a"],
            layer_b=d["layer_b"],
            head_b=d["head_b"],
        )


@dataclass
class MLPSwap:
    """Represents a swap of MLP columns between two layers.
    
    Swaps column `col_a` in layer `layer_a` with column `col_b` in layer `layer_b`.
    """
    layer_a: int
    col_a: int
    layer_b: int
    col_b: int
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "layer_a": self.layer_a,
            "col_a": self.col_a,
            "layer_b": self.layer_b,
            "col_b": self.col_b,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> "MLPSwap":
        return cls(
            layer_a=d["layer_a"],
            col_a=d["col_a"],
            layer_b=d["layer_b"],
            col_b=d["col_b"],
        )


@dataclass
class PermutationKey:
    """A permutation key specifying cross-layer swaps for tiered alignment.
    
    The key defines the secret permutation π that transforms the public model
    computation graph into the keyed model computation graph.
    
    Attributes:
        attention_swaps: List of attention head swaps between layers.
        mlp_swaps: List of MLP column swaps between layers.
    """
    attention_swaps: List[AttentionSwap] = field(default_factory=list)
    mlp_swaps: List[MLPSwap] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "attention_swaps": [swap.to_dict() for swap in self.attention_swaps],
            "mlp_swaps": [swap.to_dict() for swap in self.mlp_swaps],
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PermutationKey":
        attention_swaps = [
            AttentionSwap.from_dict(swap) 
            for swap in d.get("attention_swaps", [])
        ]
        mlp_swaps = [
            MLPSwap.from_dict(swap) 
            for swap in d.get("mlp_swaps", [])
        ]
        return cls(attention_swaps=attention_swaps, mlp_swaps=mlp_swaps)
    
    def is_empty(self) -> bool:
        """Check if the key specifies no swaps."""
        return len(self.attention_swaps) == 0 and len(self.mlp_swaps) == 0


def load_key(path: str) -> PermutationKey:
    """Load a permutation key from a JSON file.
    
    Args:
        path: Path to the JSON key file.
        
    Returns:
        PermutationKey object.
        
    Raises:
        FileNotFoundError: If the key file doesn't exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the key format is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Key file not found: {path}")
    
    with open(path, "r") as f:
        data = json.load(f)
    
    # Parse attention swaps
    attention_swaps = []
    for swap in data.get("attention_swaps", []):
        attention_swaps.append(AttentionSwap.from_dict(swap))
    
    # Parse MLP swaps
    mlp_swaps = []
    for swap in data.get("mlp_swaps", []):
        mlp_swaps.append(MLPSwap.from_dict(swap))
    
    return PermutationKey(attention_swaps=attention_swaps, mlp_swaps=mlp_swaps)


def save_key(key: PermutationKey, path: str) -> None:
    """Save a permutation key to a JSON file.
    
    Args:
        key: The permutation key to save.
        path: Path where to save the JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(key.to_dict(), f, indent=2)


def validate_key(key: PermutationKey, num_layers: int, num_heads: int, mlp_dim: int) -> None:
    """Validate that a permutation key is compatible with a model configuration.
    
    Args:
        key: The permutation key to validate.
        num_layers: Number of layers in the model.
        num_heads: Number of attention heads per layer.
        mlp_dim: Intermediate dimension of MLP layers.
        
    Raises:
        ValueError: If the key specifies invalid layer/head/column indices.
    """
    for swap in key.attention_swaps:
        if swap.layer_a < 0 or swap.layer_a >= num_layers:
            raise ValueError(
                f"Invalid layer_a in attention swap: {swap.layer_a}. "
                f"Must be in range [0, {num_layers - 1}]."
            )
        if swap.layer_b < 0 or swap.layer_b >= num_layers:
            raise ValueError(
                f"Invalid layer_b in attention swap: {swap.layer_b}. "
                f"Must be in range [0, {num_layers - 1}]."
            )
        if swap.head_a < 0 or swap.head_a >= num_heads:
            raise ValueError(
                f"Invalid head_a in attention swap: {swap.head_a}. "
                f"Must be in range [0, {num_heads - 1}]."
            )
        if swap.head_b < 0 or swap.head_b >= num_heads:
            raise ValueError(
                f"Invalid head_b in attention swap: {swap.head_b}. "
                f"Must be in range [0, {num_heads - 1}]."
            )
    
    for swap in key.mlp_swaps:
        if swap.layer_a < 0 or swap.layer_a >= num_layers:
            raise ValueError(
                f"Invalid layer_a in MLP swap: {swap.layer_a}. "
                f"Must be in range [0, {num_layers - 1}]."
            )
        if swap.layer_b < 0 or swap.layer_b >= num_layers:
            raise ValueError(
                f"Invalid layer_b in MLP swap: {swap.layer_b}. "
                f"Must be in range [0, {num_layers - 1}]."
            )
        if swap.col_a < 0 or swap.col_a >= mlp_dim:
            raise ValueError(
                f"Invalid col_a in MLP swap: {swap.col_a}. "
                f"Must be in range [0, {mlp_dim - 1}]."
            )
        if swap.col_b < 0 or swap.col_b >= mlp_dim:
            raise ValueError(
                f"Invalid col_b in MLP swap: {swap.col_b}. "
                f"Must be in range [0, {mlp_dim - 1}]."
            )
