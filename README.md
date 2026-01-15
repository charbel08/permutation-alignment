# Tiered Alignment

Code implementing **Tiered Alignment**, a framework for releasing open-weight LLMs while restricting access to certain capabilities to users who possess a secret permutation key.

## Overview

Tiered Alignment enables:
- **Public tier**: Standard model behavior accessible to all users
- **Keyed tier**: Enhanced capabilities accessible only to users with the secret key

The key specifies a permutation of model parameters (attention heads and MLP columns) that defines an alternative computation graph over the same underlying weights.

## Installation

```bash
uv pip install -e .
```

## Running Tests

Tests are implemented using pytest:

```bash
pytest ./src -sv
```

## Code Structure

```
├── src/sgtm/
│   ├── model/          # Model implementations and parameter masking
│   │   └── tests/      # Unit tests for model components
│   ├── permutation/    # Permutation key management and application
│   │   └── tests/      # Tests for permutation module
│   ├── train/          # Training scripts for different experiments
│   └── data/           # Data processing and tokenization scripts
├── configs/
│   └── wiki/           # Model hyperparameters for Wikipedia experiments
├── scripts/
│   ├── wiki/           # Bash scripts for Wikipedia experiments
│   └── data/           # Data preparation scripts
└── notebooks/          # Analysis and visualization notebooks
```

## Permutation Keys

A permutation key specifies how to swap attention heads and MLP columns across layers:

```json
{
  "attention_swaps": [
    {"layer_a": 2, "head_a": 1, "layer_b": 5, "head_b": 3}
  ],
  "mlp_swaps": [
    {"layer_a": 1, "col_a": 10, "layer_b": 7, "col_b": 42}
  ]
}
```

### Using Keys

```python
from sgtm.model import GPTNeoForCausalLMSGTM
from sgtm.permutation import load_key

# Load model and key
model = GPTNeoForCausalLMSGTM.from_pretrained("path/to/model")
key = load_key("path/to/key.json")

# Transform to keyed configuration
model.apply_key(key)

# Generate with keyed model
outputs = model.generate(input_ids)

# Return to public configuration
model.unapply_key(key)
```

## Training with Asymmetric Gradient Flow

The `tier_mode` parameter controls which tier's parameters receive gradient updates:

```python
from transformers import GPTNeoConfig
from sgtm.model import GPTNeoForCausalLMSGTM

# Configure the model with tiered alignment parameters
config = GPTNeoConfig(
    vocab_size=50257,
    hidden_size=512,
    num_layers=12,
    num_heads=32,
    # Tiered alignment parameters
    retain_mlp_dim=1984,           # Public MLP dimensions
    retain_attn_heads=31,          # Public attention heads
    masking_strategy="parameter_masking",
    split_masked_weights=True,
)

model = GPTNeoForCausalLMSGTM(config)

# During training, specify which tier you're training
outputs = model(
    input_ids=batch["input_ids"],
    labels=batch["labels"],
    tier_mode="keyed"  # or "public" or "default"
)

# After backward pass, adjust gradients based on the tier
loss.backward()
model.adjust_gradients(tier_mode="keyed")
optimizer.step()
```

### Key Classes

- **`GPTNeoForCausalLMSGTM`**: Main model class with tiered alignment support
- **`PermutationKey`**: Represents a secret permutation key
- **`apply_permutation`** / **`unapply_permutation`**: Apply/reverse key to model weights

## Experiments

### Data Preparation

```bash
bash scripts/data/prepare_wikipedia.sh
```

### Running Experiments

```bash
bash scripts/wiki/run.sh
```

## Citation

```bibtex
(Citation will be added upon paper publication)
```