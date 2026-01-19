# Tiered Alignment

Code implementing **Tiered Alignment**, a framework for releasing open-weight LLMs while restricting access to certain capabilities to users who possess a secret permutation key.

## Overview

Tiered Alignment enables:
- **Public tier (C1)**: Standard model behavior accessible to all users
- **Keyed tier (C2)**: Enhanced capabilities accessible only to users with the secret key

The key specifies a permutation of model parameters (attention heads and MLP columns) that defines an alternative computation graph over the same underlying weights.

## Installation

```bash
pip install -e .
```

## Running Tests

```bash
pytest ./src/sgtm/tests -sv
```

## Code Structure

```
src/sgtm/
├── model/              # GPT-Neo with tiered alignment support
├── permutation/        # Key management, permutation, masking, scaling
├── train/              # Training scripts
│   ├── tiered_pretrain.py  # Main tiered pretraining script
│   ├── utils.py            # Training utilities
│   └── trainer.py          # Original SGTM trainer (reference)
└── tests/              # Unit tests
```

## Permutation Keys

A permutation key specifies attention head and MLP column swaps across layers:

```json
{
  "attn_heads": [
    [[1, 0], [5, 3]]
  ],
  "mlp_cols": [
    [[2, 10], [7, 42]]
  ]
}
```

Each swap is `[[layer_a, idx_a], [layer_b, idx_b]]`.

## Using Keys

```python
from sgtm.model import GPTNeoForCausalLMSGTM
from sgtm.permutation import load_key

model = GPTNeoForCausalLMSGTM.from_pretrained("path/to/model")
key = load_key("path/to/key.json")

# Transform to keyed configuration (C2)
model.apply_key(key)
outputs = model.generate(input_ids)

# Return to public configuration (C1)
model.unapply_key(key)
```

## Tiered Pretraining

Train a model with asymmetric gradient updates:

```bash
python src/sgtm/train/tiered_pretrain.py \
    --data_path /path/to/tokenized/data \
    --key_path examples/key_32m.json \
    --output_dir /path/to/output \
    --hidden_size 384 \
    --num_heads 6 \
    --num_layers 8 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --max_steps 100000
```

### Training Algorithm

1. Forward/backward on C1 (public), mask keyed gradients
2. Forward/backward on C2 (keyed), gradients accumulate
3. Scale public gradients to average both passes
4. Update weights

Result:
- **S' (keyed params)**: Updated only by C2
- **S (public params)**: Updated by average of C1 and C2

## Key Classes

| Class | Description |
|-------|-------------|
| `GPTNeoForCausalLMSGTM` | GPT-Neo with `apply_key`, `unapply_key`, `mask_*_gradients` methods |
| `PermutationKey` | Represents a secret permutation key |
| `apply_permutation` / `unapply_permutation` | Apply/reverse key to model weights |
| `mask_keyed_gradients` / `mask_public_gradients` | Zero gradients for parameter subsets |
| `scale_public_gradients` | Scale gradients for public parameters |

## Citation

```bibtex
(Citation will be added upon paper publication)
```