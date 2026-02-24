# Tiered Alignment

Code implementing **Tiered Alignment**, a framework for releasing open-weight LLMs while restricting access to certain capabilities to users who possess a secret permutation key.

## Overview

Tiered Alignment enables two behavioral tiers from a single set of weights:

- **Public tier (C1)**: Standard model behavior accessible to all users
- **Keyed tier (C2)**: Enhanced capabilities accessible only to users with the secret key

The key specifies a permutation of model parameters — swaps of attention heads and MLP columns across layers — that defines an alternative computation graph over the same underlying weights. Since the permutation is self-inverse, applying the key toggles between C1 and C2 configurations.

### Training Algorithm

Training jointly optimizes both tiers with asymmetric gradient updates:

1. Forward/backward on **C1** (public architecture), mask keyed gradients
2. Apply permutation key to get **C2**
3. Forward/backward on **C2** (keyed architecture), gradients accumulate
4. Scale public gradients by 0.5 to average both passes
5. **Optimizer step while in C2 config** (critical for gradient–weight alignment)
6. Unapply permutation to return to C1

Result:
- **Keyed params (S')**: Updated only by C2 loss
- **Public params (S)**: Updated by the average of C1 and C2 losses

### Private Finetuning

After pretraining, the keyed tier can be finetuned on private data while preserving public behavior:

```
L_ft = (1 − λ) · L_priv(C2) + λ · R_KL(C1_current, C1_frozen)
```

Only keyed weights are updated during finetuning. The KL term regularizes against C1 divergence from the pretrained baseline.

## Installation

```bash
pip install -e .
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.6.0, Transformers ~4.52.4. See `pyproject.toml` for the full dependency list.

## Running Tests

```bash
PYTHONPATH=./src pytest ./src/sgtm/tests -sv
```

## Project Structure

```
├── configs/
│   ├── keys/                     # Pre-generated permutation keys (20% coverage)
│   │   ├── key_32m_20pct_mixed.json
│   │   ├── key_34m_20pct_mixed.json
│   │   └── key_64m_20pct_mixed.json
│   └── wiki/                     # Model architecture configs (34M–254M)
│       ├── 34M.yaml
│       ├── 64M.yaml
│       ├── 125M.yaml
│       └── 254M.yaml
├── notebooks/
│   ├── explore_wiki_bio.ipynb    # Dataset exploration
│   └── wiki/
│       ├── retain_forget_tradeoff.ipynb
│       └── finetuning.ipynb
├── scripts/
│   ├── generate_key.py           # Permutation key generator
│   ├── data/
│   │   ├── prepare_wikipedia.sh  # Wikipedia dataset preparation
│   │   ├── tinystories_tokenize.sh
│   │   └── translate_to_spanish.sh
│   └── wiki/
│       ├── run.sh                # Training launch scripts
│       ├── finetune.sh
│       └── rmu.sh                # Representation misdirection unlearning
├── src/sgtm/
│   ├── model/
│   │   └── gpt.py                # GPTNeoForCausalLMSGTM
│   ├── permutation/
│   │   ├── key.py                # PermutationKey dataclass, load/save/validate
│   │   ├── permute.py            # Weight & gradient swapping (batched)
│   │   ├── masking.py            # Gradient masking (keyed vs public)
│   │   └── scaling.py            # Public gradient scaling
│   ├── train/
│   │   ├── tiered_pretrain.py    # Tiered alignment pretraining
│   │   ├── pretrain.py           # Standard baseline pretraining (no tiered alignment)
│   │   ├── private_finetune.py   # KL-regularized private finetuning
│   │   ├── inference.py          # Compare C1 vs C2 generation
│   │   ├── trainer.py            # Legacy SGTM trainer (reference only)
│   │   └── utils.py              # Model loading, checkpointing, tokenizer
│   ├── data/
│   │   ├── prepare_wikipedia.py  # Wikipedia → forget/adjacent/retain splits
│   │   ├── tinystories_tokenize_and_split.py
│   │   └── explore.py            # Dataset statistics
│   └── tests/
│       ├── test_permutation.py        # Key I/O, apply/unapply identity, weight correctness
│       ├── test_gradients.py          # Gradient masking & weight update verification
│       ├── test_gradient_alignment.py # Gradient–weight alignment after permutation
│       ├── test_end_to_end_training.py
│       └── test_private_finetune.py   # KL + swap_gradients, padding exclusion
├── pretrain.sbatch               # SLURM job: baseline pretraining
├── tiered_pretrain.sbatch        # SLURM job: tiered pretraining
└── pyproject.toml
```

## Permutation Keys

A permutation key specifies which attention heads and MLP columns to swap across layers:

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

Each swap is `[[layer_a, idx_a], [layer_b, idx_b]]`. All swaps are transpositions (pair-swaps), so applying the key twice returns to the original state.

### Generating Keys

```bash
python scripts/generate_key.py \
    --output configs/keys/my_key.json \
    --num_layers 12 --num_heads 32 --hidden_size 512 --mlp_dim 2048 \
    --target_pct 0.20 --attn_ratio 0.25 --seed 42
```

This generates a key covering ~20% of model parameters, split 25% attention / 75% MLP. Cross-layer swaps are enforced for better mixing. Pre-generated keys for the 32M, 34M, and 64M configs are in `configs/keys/`.

## Usage

### Inference: Comparing C1 vs C2

```python
from sgtm.model import GPTNeoForCausalLMSGTM
from sgtm.permutation import load_key

model = GPTNeoForCausalLMSGTM.from_pretrained("path/to/checkpoint")
key = load_key("configs/keys/key_64m_20pct_mixed.json")

# Public configuration (C1)
outputs_c1 = model.generate(input_ids)

# Keyed configuration (C2)
model.apply_key(key)
outputs_c2 = model.generate(input_ids)
model.unapply_key(key)
```

Or use the inference script directly:

```bash
PYTHONPATH=./src python src/sgtm/train/inference.py \
    --checkpoint /path/to/checkpoint \
    --key_path configs/keys/key_64m_20pct_mixed.json \
    --prompt "Once upon a time"
```

### Tiered Pretraining

Train a model with asymmetric gradient updates on both C1 and C2:

```bash
torchrun --standalone --nproc_per_node=3 -m sgtm.train.tiered_pretrain \
    --data_path /path/to/tokenized/data \
    --key_path configs/keys/key_64m_20pct_mixed.json \
    --output_dir /path/to/output \
    --hidden_size 512 \
    --num_heads 32 \
    --num_layers 12 \
    --context_size 1024 \
    --batch_size 16 \
    --learning_rate 6e-4 \
    --min_lr 6e-5 \
    --max_steps 71393 \
    --warmup_steps 500 \
    --wandb_project tiered-alignment \
    --run_name my_tiered_run
```

Supports distributed training via `torchrun` and automatic checkpoint resumption via `--checkpoint`.

### Baseline Pretraining (No Tiered Alignment)

```bash
torchrun --standalone --nproc_per_node=3 -m sgtm.train.pretrain \
    --data_path /path/to/tokenized/data \
    --output_dir /path/to/output \
    --hidden_size 512 --num_heads 32 --num_layers 12 \
    --batch_size 16 --learning_rate 6e-4 --max_steps 71393
```

### Private Finetuning

Finetune the keyed tier on private data while preserving public behavior:

```bash
PYTHONPATH=./src python src/sgtm/train/private_finetune.py \
    --checkpoint /path/to/tiered/checkpoint \
    --key_path configs/keys/key_64m_20pct_mixed.json \
    --private_data /path/to/forget/data \
    --public_data /path/to/retain/data \
    --output_dir /path/to/output \
    --learning_rate 1e-5 \
    --kl_lambda 0.1 \
    --max_steps 10000
```

## Data Preparation

### Wikipedia (Forget/Adjacent/Retain Splits)

Prepares Wikipedia articles into category-based splits using ORES topic labels:

```bash
python -m sgtm.data.prepare_wikipedia \
    --topics-file /path/to/enwiki_topics2020.csv \
    --output-dir /path/to/output \
    --chunk-size 1024 \
    --forget-categories "STEM.Biology" \
    --adjacent-categories "STEM.Earth_and_environment" "STEM.Chemistry" "STEM.Medicine_&_Health"
```

This produces three HuggingFace `DatasetDict`s (forget, adjacent, retain), each with train/test splits, tokenized and chunked to fixed lengths.

### TinyStories (Bilingual)

```bash
python -m sgtm.data.tinystories_tokenize_and_split \
    --dataset-name ffuuugor/tinystories_spanish \
    --output-dir /path/to/output \
    --context-size 1024
```

## Model Configurations

| Config | Layers | Hidden | Heads | Intermediate | ~Params |
|--------|--------|--------|-------|-------------|---------|
| 34M    | 8      | 384    | 32    | 1,536       | 34M     |
| 64M    | 12     | 512    | 32    | 2,048       | 64M     |
| 125M   | 12     | 768    | 32    | 3,072       | 125M    |
| 254M   | 16     | 1,024  | 32    | 4,096       | 254M    |

All models use GPT-Neo architecture with alternating global/local attention, a 1024-token context window, GPT-2 tokenizer (vocab size 50,257), and tied input/output embeddings.

## Key API Reference

| Component | Description |
|-----------|-------------|
| `GPTNeoForCausalLMSGTM` | GPT-Neo extended with `apply_key`, `unapply_key`, `mask_keyed_gradients`, `mask_public_gradients` |
| `PermutationKey` | Dataclass holding `attn_heads` and `mlp_cols` swap lists |
| `load_key` / `save_key` | JSON serialization for keys |
| `validate_key` | Bounds-check key indices against model dimensions |
| `apply_permutation` / `unapply_permutation` | Swap weights according to key (self-inverse) |
| `swap_gradients` | Swap gradients to follow their weight values after `apply_key` |
| `mask_keyed_gradients` | Zero gradients for keyed parameters (used in C1 backward) |
| `mask_public_gradients` | Zero gradients for public parameters (used in private finetuning) |
| `scale_public_gradients` | Scale public gradients by a factor (default 0.5 for averaging C1+C2) |

## Important Implementation Details

**Gradient–weight alignment:** The optimizer step must happen *while the model is in C2 configuration*. If you unapply the key before stepping, gradients end up applied to the wrong weight values. The test suite (`test_gradient_alignment.py`) explicitly verifies this invariant.

**`swap_gradients` in private finetuning:** When computing KL loss on C1 and then switching to C2 for the private loss, the KL gradients must be swapped to follow their weights into C2 positions before accumulating. This is handled by `swap_gradients` in `private_finetune.py`.

**Batched MLP swaps:** MLP column swaps are grouped by layer pair and executed as batch index operations for efficiency, since keys can contain thousands of MLP swaps.

## Citation

```bibtex
(Citation will be added upon paper publication)
```

## License

MIT
