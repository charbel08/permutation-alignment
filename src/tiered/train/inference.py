"""Inference script for tiered alignment models.

Compare completions from public (C1) and keyed (C2) configurations
across one or more checkpoints (e.g. pretrained vs fine-tuned).

Supports two modes:
  1. Manual prompt:  --prompt "Once upon a time"
  2. From dataset:   --data_path /path/to/private/data --num_examples 3
     Extracts the first N tokens from validation examples as prompts.

Usage:
    # Single checkpoint, manual prompt
    PYTHONPATH=./src python -m tiered.train.inference \\
        --checkpoint /path/to/checkpoint \\
        --key_path configs/keys/key_64m_20pct_mixed.json \\
        --prompt "Once upon a time"

    # Compare pretrained vs fine-tuned on private data
    PYTHONPATH=./src python -m tiered.train.inference \\
        --checkpoint /path/to/pretrained /path/to/finetuned \\
        --checkpoint_labels pretrained finetuned \\
        --key_path configs/keys/key_64m_20pct_mixed.json \\
        --data_path /path/to/private/data \\
        --num_examples 3 \\
        --prompt_tokens 50
"""

import argparse

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import load_key


def parse_args():
    parser = argparse.ArgumentParser(description="Tiered alignment inference")
    parser.add_argument("--checkpoint", type=str, nargs="+", required=True,
                        help="Path(s) to model checkpoint(s)")
    parser.add_argument("--checkpoint_labels", type=str, nargs="+", default=None,
                        help="Labels for each checkpoint (e.g. 'pretrained' 'finetuned')")
    parser.add_argument("--key_path", type=str, required=True,
                        help="Path to permutation key JSON")

    # Prompt source: either manual or from dataset
    parser.add_argument("--prompt", type=str, default=None,
                        help="Manual input prompt")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to tokenized dataset (uses validation examples as prompts)")
    parser.add_argument("--num_examples", type=int, default=3,
                        help="Number of examples to use from dataset (default: 3)")
    parser.add_argument("--prompt_tokens", type=int, default=50,
                        help="Number of tokens from each example to use as prompt (default: 50)")

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Maximum new tokens to generate (default: 100)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true", default=False)
    return parser.parse_args()


def generate(model, input_ids, tokenizer, max_new_tokens, temperature, top_p,
             do_sample, device):
    """Generate text from input token IDs."""
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated portion (after the prompt)
    generated_ids = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def get_prompts_from_dataset(data_path, tokenizer, num_examples, prompt_tokens):
    """Load dataset and extract prompt prefixes from validation examples."""
    dataset = load_from_disk(data_path)

    # Pick the validation split
    if "test" in dataset:
        split = dataset["test"]
    elif "validation" in dataset:
        split = dataset["validation"]
    elif "train" in dataset:
        split = dataset["train"]
    else:
        split = dataset

    prompts = []
    ground_truths = []

    for i in range(min(num_examples, len(split))):
        example = split[i]
        all_ids = example["input_ids"]

        # Use first prompt_tokens tokens as the prompt
        prompt_ids = all_ids[:prompt_tokens]
        # The remaining tokens are the "ground truth" continuation
        gt_ids = all_ids[prompt_tokens:prompt_tokens + 100]

        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        gt_text = tokenizer.decode(gt_ids, skip_special_tokens=True)

        prompts.append((torch.tensor([prompt_ids]), prompt_text))
        ground_truths.append(gt_text)

    return prompts, ground_truths


def run_inference_on_checkpoint(checkpoint_path, label, key, prompts,
                                ground_truths, tokenizer, args, device):
    """Load a checkpoint and generate C1/C2 completions for all prompts."""
    print(f"\n{'#'*70}")
    print(f"# CHECKPOINT: {label}")
    print(f"# Path: {checkpoint_path}")
    print(f"{'#'*70}")

    model = GPTNeoForCausalLMTiered.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()

    for i, ((input_ids, prompt_text), gt_text) in enumerate(zip(prompts, ground_truths)):
        print(f"\n{'='*70}")
        print(f"  EXAMPLE {i+1}")
        print(f"{'='*70}")
        print(f"\n  PROMPT ({input_ids.shape[1]} tokens):")
        print(f"  {prompt_text}")

        if gt_text:
            print(f"\n  GROUND TRUTH (next ~100 tokens):")
            print(f"  {gt_text}")

        # C1 (public)
        response_c1 = generate(
            model, input_ids, tokenizer, args.max_new_tokens,
            args.temperature, args.top_p, args.do_sample, device
        )
        print(f"\n  C1 (Public):")
        print(f"  {response_c1}")

        # C2 (keyed)
        model.apply_key(key)
        response_c2 = generate(
            model, input_ids, tokenizer, args.max_new_tokens,
            args.temperature, args.top_p, args.do_sample, device
        )
        model.unapply_key(key)
        print(f"\n  C2 (Keyed):")
        print(f"  {response_c2}")

    # Free memory
    del model
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    key = load_key(args.key_path)
    print(f"Loaded key with {len(key.attn_heads)} attention swaps, "
          f"{len(key.mlp_cols)} MLP swaps")

    # Build prompts
    if args.data_path:
        print(f"Loading {args.num_examples} examples from {args.data_path}")
        prompts, ground_truths = get_prompts_from_dataset(
            args.data_path, tokenizer, args.num_examples, args.prompt_tokens
        )
    elif args.prompt:
        input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
        prompts = [(input_ids, args.prompt)]
        ground_truths = [""]
    else:
        raise ValueError("Must provide either --prompt or --data_path")

    # Assign labels to checkpoints
    labels = args.checkpoint_labels or [f"checkpoint_{i}" for i in range(len(args.checkpoint))]
    if len(labels) != len(args.checkpoint):
        raise ValueError(f"Got {len(args.checkpoint)} checkpoints but {len(labels)} labels")

    # Run inference on each checkpoint
    for ckpt_path, label in zip(args.checkpoint, labels):
        run_inference_on_checkpoint(
            ckpt_path, label, key, prompts, ground_truths,
            tokenizer, args, device
        )

    print(f"\n{'='*70}")
    print("Done.")


if __name__ == "__main__":
    main()
