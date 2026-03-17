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

### N-Tier Extension

The framework supports **N keyed tiers** (C2, C3, ..., CN) with two strategies:

- **Sampling** (`multi_tiered_pretrain.py`): Exactly 2 forward+backward passes per step — C1 always, plus one uniformly sampled keyed tier. Each tier's keyed weights see ~1/(N−1) of steps. Constant compute regardless of N.
- **Naive** (`multi_tiered_naive.py`): All N keyed tiers trained every step — (1+K) passes per step. Upper-bound baseline where every tier sees every training step.

### Private Finetuning

After pretraining, the keyed tier can be finetuned on private data while preserving public behavior:

```
L_ft = (1 − λ) · L_priv(C2) + λ · R_KL(C1_current, C1_frozen)
```

Only keyed weights are updated during finetuning. The KL term regularizes against C1 divergence from the pretrained baseline.

**SuperGLUE finetuning** (`superglue_finetune.py`) demonstrates that C2 can learn NLU capabilities hidden behind the key, using answer-only label masking so only answer tokens contribute to the loss.

## Installation

```bash
pip install -e .
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.6.0, Transformers ~4.52.4, PEFT ≥ 0.10. See `pyproject.toml` for the full dependency list.

## Running Tests

```bash
PYTHONPATH=./src pytest ./src/tiered/tests -sv
```

## Project Structure

```
├── configs/
│   └── keys/                          # Pre-generated permutation keys
│       ├── key_64m_20pct_mixed.json   # 64M model, 20% coverage
│       ├── key_530m_14pct.json        # 530M model, 14% coverage
│       ├── key_530m_7pct_{1,2,3}.json # 530M model, 7% keys (for multi-tier)
│       └── key_1b_10pct_mixed.json    # 1B model, 10% coverage
├── scripts/
│   ├── keys/
│   │   └── generate_key.py            # Permutation key generator
│   ├── eval/
│   │   ├── eval_memorization.py       # Synthetic bio memorization eval (C1 vs C2)
│   │   ├── mmlu_qwen_key_ablation.py  # Qwen key-destruction ablation on MMLU
│   │   └── run_qwen_key_destruction_5pct.sh
│   ├── ablation/
│   │   ├── ablation_random_key.py     # Ablation 1: random key experiment
│   │   ├── ablation_corrupt_keyed.py  # Ablation 2: corrupt keyed weights
│   │   ├── ablation_gradual_corrupt.py # Ablation 3: gradual weight corruption
│   │   ├── ablation_gradual_key.py    # Ablation 4: gradual key corruption
│   │   ├── run_ablation_random_key.sh # Shell launcher for ablation 1
│   │   └── run_kl_sweep.sh            # KL lambda hyperparameter sweep
│   ├── data/
│   │   ├── generate_synthetic_bios.py # Synthetic bio dataset generator
│   │   ├── prepare_wikipedia.sh       # Wikipedia dataset preparation
│   │   ├── tinystories_tokenize.sh    # TinyStories tokenization
│   │   └── translate_to_spanish.sh    # Spanish translation pipeline
│   ├── mila/                          # SLURM sbatch scripts (Mila cluster)
│   │   ├── pretrain.sbatch
│   │   ├── tiered_pretrain.sbatch
│   │   ├── tiered_pretrain_batches.sbatch
│   │   └── private_finetune_superglue.sbatch
│   └── snow/                          # SLURM sbatch scripts (Snowflake cluster)
│       ├── data/                      # dataset prep launchers
│       ├── wiki/
│       │   └── pretrain/64m/run.sh
│       ├── wmdp/
│       │   └── finetune/150m/run_multi.sh
│       └── fineweb/
│           ├── pretrain/<size>/       # baseline + keyed pretraining launchers
│           ├── finetune/<size>/<dataset>/  # fine-tuning launchers (incl. KL=0 variants)
│           ├── eval/<size>/<dataset>/ # inference/evaluation launchers
│           └── README.md
├── src/tiered/
│   ├── model/
│   │   └── gpt.py                     # GPTNeoForCausalLMTiered
│   ├── permutation/
│   │   ├── key.py                     # PermutationKey dataclass, load/save/validate
│   │   ├── permute.py                 # Weight & gradient swapping (batched)
│   │   ├── masking.py                 # Gradient masking (keyed vs public)
│   │   ├── scaling.py                 # Public gradient scaling
│   │   └── utils.py                   # Model submodule accessors
│   ├── train/
│   │   ├── tiered_pretrain.py         # 2-tier alignment pretraining (C1/C2)
│   │   ├── multi_tiered_pretrain.py   # N-tier pretraining (sampling, constant compute)
│   │   ├── multi_tiered_naive.py      # N-tier pretraining (naive, all tiers every step)
│   │   ├── pretrain.py                # Standard baseline pretraining (no tiered alignment)
│   │   ├── private_finetune.py        # KL-regularized private finetuning
│   │   ├── superglue_finetune.py      # SuperGLUE private finetuning (answer-only masking)
│   │   ├── inference.py               # Compare C1 vs C2 generation
│   │   └── utils.py                   # Model loading, checkpointing, tokenizer
│   ├── data/
│   │   ├── prepare_wikipedia.py       # Wikipedia → forget/adjacent/retain splits
│   │   ├── prepare_fineweb.py         # FineWeb → tokenized chunks for pretraining
│   │   ├── prepare_superglue.py       # SuperGLUE → answer-only masked dataset
│   │   ├── tinystories_tokenize_and_split.py
│   │   └── explore.py                 # Dataset statistics
│   └── tests/
│       ├── test_permutation.py        # Key I/O, apply/unapply identity, weight correctness
│       ├── test_gradients.py          # Gradient masking & weight update verification
│       ├── test_gradient_alignment.py # Gradient–weight alignment after permutation
│       ├── test_end_to_end_training.py
│       └── test_private_finetune.py   # KL + swap_gradients, padding exclusion
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
python scripts/keys/generate_key.py \
    --output configs/keys/my_key.json \
    --num_layers 12 --num_heads 32 --hidden_size 512 --mlp_dim 2048 \
    --target_pct 0.20 --attn_ratio 0.25 --seed 42
```

This generates a key covering ~20% of model parameters, split 25% attention / 75% MLP. Cross-layer swaps are enforced for better mixing. Pre-generated keys are in `configs/keys/`.

| Key file | Model | Coverage |
|----------|-------|----------|
| `key_64m_20pct_mixed.json` | 64M | 20% |
| `key_530m_14pct.json` | 530M | 14% |
| `key_530m_7pct_{1,2,3}.json` | 530M | 7% each (for multi-tier) |
| `key_1b_10pct_mixed.json` | 1B | 10% |

## Usage

### Inference: Comparing C1 vs C2

```python
from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import load_key

model = GPTNeoForCausalLMTiered.from_pretrained("path/to/checkpoint")
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
PYTHONPATH=./src python src/tiered/train/inference.py \
    --checkpoint /path/to/checkpoint \
    --key_path configs/keys/key_64m_20pct_mixed.json \
    --prompt "Once upon a time"
```

### 2-Tier Pretraining

Train a model with asymmetric gradient updates on C1 and C2:

```bash
torchrun --standalone --nproc_per_node=3 -m tiered.train.tiered_pretrain \
    --data_path /path/to/tokenized/data \
    --key_path configs/keys/key_64m_20pct_mixed.json \
    --output_dir /path/to/output \
    --hidden_size 512 --num_heads 32 --num_layers 12 \
    --context_size 1024 --batch_size 16 \
    --learning_rate 6e-4 --min_lr 6e-5 \
    --max_steps 71393 --warmup_steps 500 \
    --grad_accum_steps 1 \
    --wandb_project tiered-alignment --run_name my_run
```

Supports distributed training via `torchrun`, gradient accumulation, `torch.compile`, and automatic checkpoint resumption via `--checkpoint`.

### N-Tier Pretraining (Sampling)

Train with multiple keyed tiers using constant compute (2 passes per step):

```bash
torchrun --standalone --nproc_per_node=4 -m tiered.train.multi_tiered_pretrain \
    --data_path /path/to/tokenized/data \
    --key_paths key1.json key2.json key3.json \
    --output_dir /path/to/output \
    --hidden_size 1024 --num_heads 16 --num_layers 24 \
    --batch_size 8 --max_steps 100000 \
    --tier_sample uniform
```

Options: `--tier_sample uniform` (random) or `round_robin` (deterministic cycling).

### N-Tier Pretraining (Naive)

Upper-bound baseline — all tiers every step:

```bash
torchrun --standalone --nproc_per_node=4 -m tiered.train.multi_tiered_naive \
    --data_path /path/to/tokenized/data \
    --key_paths key1.json key2.json key3.json \
    --output_dir /path/to/output \
    --max_steps 100000
```

### Baseline Pretraining (No Tiered Alignment)

```bash
torchrun --standalone --nproc_per_node=3 -m tiered.train.pretrain \
    --data_path /path/to/tokenized/data \
    --output_dir /path/to/output \
    --hidden_size 512 --num_heads 32 --num_layers 12 \
    --batch_size 16 --learning_rate 6e-4 --max_steps 71393
```

### Private Finetuning

Finetune the keyed tier on private data while preserving public behavior:

```bash
PYTHONPATH=./src python src/tiered/train/private_finetune.py \
    --checkpoint /path/to/tiered/checkpoint \
    --key_path configs/keys/key_64m_20pct_mixed.json \
    --private_data /path/to/forget/data \
    --public_data /path/to/retain/data \
    --output_dir /path/to/output \
    --learning_rate 1e-5 --kl_lambda 0.1 --max_steps 10000
```

### LoRA Private Finetuning Baseline (Separate Weights)

Train a **PEFT LoRA** adapter on private data where:
- **C1** = base model with adapter disabled
- **C2** = base model with adapter enabled

The script auto-selects the highest LoRA rank whose trainable parameter count
fits the keyed-parameter budget induced by `--key_path`, then reports C1/C2
performance plus FLOPs comparisons against a 2-pass tiered reference.

```bash
torchrun --standalone --nproc_per_node=8 -m tiered.train.lora_private_finetune \
    --checkpoint /path/to/base/checkpoint \
    --key_path configs/keys/key_150m_14pct.json \
    --private_data /path/to/private/data \
    --public_data /path/to/retain/data \
    --output_dir /path/to/output \
    --batch_size 8 --max_steps 10000
```

Key outputs:
- `output_dir/final/adapter_model.safetensors` (or `.bin`): PEFT adapter weights
- `output_dir/final/adapter_config.json`: PEFT adapter config
- `output_dir/final/experiment_metadata.json`: rank/budget metadata
- `output_dir/comparison_summary.json`: perf + FLOPs comparison summary

### SuperGLUE Private Finetuning

Finetune C2 on SuperGLUE NLU tasks with answer-only label masking:

```bash
PYTHONPATH=./src python src/tiered/train/superglue_finetune.py \
    --checkpoint /path/to/pretrained \
    --key_path configs/keys/key_64m_20pct_mixed.json \
    --superglue_data /path/to/superglue \
    --public_data /path/to/wiki_bio/retain \
    --output_dir /path/to/output \
    --kl_lambda 0.1 --max_steps 10000
```

Tracks three validation signals: C1 on retain (stability), C1 on SuperGLUE (no leakage), C2 on SuperGLUE (learning).

## Data Preparation

### FineWeb (Large-Scale Pretraining)

Download and tokenize FineWeb for pretraining:

```bash
python -m tiered.data.prepare_fineweb \
    --output-dir /path/to/output \
    --chunk-size 1024 \
    --max-tokens 100000000000 \
    --subset sample-100BT
```

Subsets: `sample-10BT`, `sample-100BT`, `sample-350BT`, `default`. Uses `tiktoken` (GPT-2 encoding) and `hf_transfer` for fast downloads.

### Wikipedia (Forget/Adjacent/Retain Splits)

Prepares Wikipedia articles into category-based splits using ORES topic labels:

```bash
python -m tiered.data.prepare_wikipedia \
    --topics-file /path/to/enwiki_topics2020.csv \
    --output-dir /path/to/output \
    --chunk-size 1024 \
    --forget-categories "STEM.Biology" \
    --adjacent-categories "STEM.Earth_and_environment" "STEM.Chemistry" "STEM.Medicine_&_Health"
```

This produces three HuggingFace `DatasetDict`s (forget, adjacent, retain), each with train/test splits, tokenized and chunked to fixed lengths.

### SuperGLUE

Prepares all 8 SuperGLUE tasks (BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC) with answer-only label masking:

```bash
PYTHONPATH=./src python src/tiered/data/prepare_superglue.py \
    --output-dir /path/to/output/superglue \
    --context-size 1024
```

Each example stores `input_ids`, `attention_mask`, and `labels` where `labels=-100` for prompt tokens (only answer tokens receive gradient).

### Synthetic Bios (Memorization Evaluation)

Generates a controlled dataset of synthetic biographies for measuring memorization:

```bash
python scripts/data/generate_synthetic_bios.py \
    --output-dir /path/to/output \
    --num-people 400 --seed 42
```

400 people × 24 attribute permutations = 9,600 samples. Each person has unique name, profession, hobby, and salary. Split by person (not sample) to avoid leakage.

### TinyStories (Bilingual)

```bash
python -m tiered.data.tinystories_tokenize_and_split \
    --dataset-name ffuuugor/tinystories_spanish \
    --output-dir /path/to/output \
    --context-size 1024
```

## Evaluation & Ablations

### Memorization Evaluation

Measures how well the model memorizes synthetic bio attribute values (not filler), using precise token-level matching:

```bash
# Single checkpoint
PYTHONPATH=./src python scripts/eval/eval_memorization.py \
    --checkpoint /path/to/checkpoint \
    --bio_metadata /path/to/bios_metadata.json \
    --key_path configs/keys/key_64m_20pct_mixed.json \
    --output_dir /path/to/output

# Sweep across all checkpoints
PYTHONPATH=./src python scripts/eval/eval_memorization.py \
    --checkpoint /path/to/ckpt_dir \
    --bio_metadata /path/to/bios_metadata.json \
    --key_path configs/keys/key_64m_20pct_mixed.json \
    --output_dir /path/to/output --sweep
```

Reports per-attribute breakdowns (age, profession, hobby, salary), top-k accuracy, and exact match rates for both C1 and C2.

### Ablation Studies

| Script | Description |
|--------|-------------|
| `scripts/ablation/ablation_random_key.py` | Evaluates C1, C2 (correct key), and C3 (random keys). Tests whether a random key can unlock C2 capabilities. |
| `scripts/ablation/ablation_corrupt_keyed.py` | Randomizes keyed weight values, measures C1/C2 loss. Tests C1 robustness to keyed weight corruption. |
| `scripts/ablation/ablation_gradual_corrupt.py` | Gradually corrupts keyed weights 1% at a time, tracking degradation curves on private and public data. |
| `scripts/ablation/ablation_gradual_key.py` | Gradually replaces correct key swaps with random ones, tracking C2 degradation as key accuracy decreases. |

All ablations produce charts (loss and top-k accuracy), log to WandB, and save JSON results.

### Qwen Key-Destruction Ablation (MMLU)

Runs MMLU evaluation while increasing key coverage in 5% increments:

```bash
bash scripts/eval/run_qwen_key_destruction_5pct.sh
```

Direct evaluator script:

```bash
python scripts/eval/mmlu_qwen_key_ablation.py --help
```

## Model Configurations

| Config | Layers | Hidden | Heads | Intermediate | ~Params |
|--------|--------|--------|-------|-------------|---------|
| 64M    | 12     | 512    | 32    | 2,048       | 64M     |
| 530M   | 24     | 1,024  | 16    | 4,096       | 530M    |
| 1B     | 24     | 2,048  | 16    | 8,192       | 1B      |

All models use GPT-Neo architecture with alternating global/local attention, a 1024-token context window, GPT-2 tokenizer (vocab size 50,257), and tied input/output embeddings.

## Key API Reference

| Component | Description |
|-----------|-------------|
| `GPTNeoForCausalLMTiered` | GPT-Neo extended with `apply_key`, `unapply_key`, `mask_keyed_gradients`, `mask_public_gradients` |
| `PermutationKey` | Dataclass holding `attn_heads` and `mlp_cols` swap lists |
| `load_key` / `save_key` | JSON serialization for keys |
| `validate_key` | Bounds-check key indices against model dimensions |
| `apply_permutation` / `unapply_permutation` | Swap weights according to key (self-inverse) |
| `build_swap_plan` / `build_mask_plan` | Pre-compute permutation/masking index tensors for efficiency |
| `swap_gradients` | Swap gradients to follow their weight values after `apply_key` |
| `mask_keyed_gradients` | Zero gradients for keyed parameters (used in C1 backward) |
| `mask_public_gradients` | Zero gradients for public parameters (used in private finetuning) |
| `scale_public_gradients` | Scale public gradients by a factor (default 0.5 for averaging C1+C2) |

## Important Implementation Details

**Gradient–weight alignment:** The optimizer step must happen *while the model is in C2 configuration*. If you unapply the key before stepping, gradients end up applied to the wrong weight values. The test suite (`test_gradient_alignment.py`) explicitly verifies this invariant.

**`swap_gradients` in private finetuning:** When computing KL loss on C1 and then switching to C2 for the private loss, the KL gradients must be swapped to follow their weights into C2 positions before accumulating. This is handled by `swap_gradients` in `private_finetune.py`.

**Batched MLP swaps:** MLP column swaps are grouped by layer pair and executed as batch index operations for efficiency, since keys can contain thousands of MLP swaps.

**Pre-computed plans:** `build_swap_plan` and `build_mask_plan` pre-compute index tensors on the target device, avoiding repeated tensor construction per step. This is critical for large keys (530M+).

**N-tier gradient combination (naive):** In the naive all-tiers script, gradients are combined by scaling all gradients by 1/N, then rescaling keyed positions back by N so each tier's keyed weights receive the full (unaveraged) gradient from their own pass.

## Citation

```bibtex
(Citation will be added upon paper publication)
```

## License

MIT
