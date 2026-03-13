"""Inference script for tiered alignment models.

Compares completions from C1 (public) and C2..CN (keyed tiers) across
one or more checkpoints.

Prompt modes:
  1. Manual prompt:
     --prompt "Once upon a time"
  2. Single dataset:
     --data_path /path/to/data --num_examples 3
  3. Multi-language one-sample mode:
     --language_data en=/path/to/en spa=/path/to/spa deu=/path/to/deu tur=/path/to/tur
"""

import argparse
import os

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
    parser.add_argument("--key_path", type=str, default=None,
                        help="Path to one permutation key JSON (used as C2)")
    parser.add_argument("--key_paths", type=str, nargs="*", default=None,
                        help="Paths to permutation key JSONs for C2..CN")

    # Prompt source: either manual or from dataset
    parser.add_argument("--prompt", type=str, default=None,
                        help="Manual input prompt")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to tokenized dataset (uses validation examples as prompts)")
    parser.add_argument("--language_data", type=str, nargs="+", default=None,
                        help=("Language dataset specs as LANG=PATH. Takes one sample from "
                              "each language and runs it through C1..CN."))
    parser.add_argument("--num_examples", type=int, default=3,
                        help="Number of examples to use from dataset (default: 3)")
    parser.add_argument("--sample_index", type=int, default=0,
                        help="Dataset example index to sample from (default: 0)")
    parser.add_argument("--prompt_tokens", type=int, default=50,
                        help="Number of tokens from each example to use as prompt (default: 50)")

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Maximum new tokens to generate (default: 100)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--output_file", type=str, default=None,
                        help="Optional path to save all results as a text report")
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


def get_dataset_split(dataset):
    """Pick a split from DatasetDict, or return dataset directly if not a dict."""
    if hasattr(dataset, "keys"):
        if "test" in dataset:
            return dataset["test"]
        if "validation" in dataset:
            return dataset["validation"]
        if "train" in dataset:
            return dataset["train"]
    return dataset


def make_prompt_example(example, tokenizer, prompt_tokens, source):
    """Build prompt/ground-truth text fields from one tokenized example."""
    all_ids = example["input_ids"]
    prompt_ids = all_ids[:prompt_tokens]
    gt_ids = all_ids[prompt_tokens:prompt_tokens + 100]

    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
    gt_text = tokenizer.decode(gt_ids, skip_special_tokens=True)

    return {
        "source": source,
        "input_ids": torch.tensor([prompt_ids], dtype=torch.long),
        "prompt_text": prompt_text,
        "ground_truth": gt_text,
    }


def get_prompts_from_dataset(data_path, tokenizer, num_examples, prompt_tokens, sample_index=0):
    """Load dataset and extract prompt prefixes."""
    dataset = load_from_disk(data_path)
    split = get_dataset_split(dataset)

    if len(split) == 0:
        raise ValueError(f"No rows found in dataset split at: {data_path}")

    examples = []
    for offset in range(num_examples):
        idx = (sample_index + offset) % len(split)
        source = f"{data_path} [idx={idx}]"
        examples.append(make_prompt_example(split[idx], tokenizer, prompt_tokens, source))
    return examples


def parse_language_data_specs(language_data):
    """Parse LANG=PATH arguments."""
    parsed = []
    for spec in language_data:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --language_data entry '{spec}'. Expected LANG=PATH."
            )
        lang, path = spec.split("=", 1)
        lang = lang.strip()
        path = path.strip()
        if not lang or not path:
            raise ValueError(
                f"Invalid --language_data entry '{spec}'. Expected LANG=PATH."
            )
        parsed.append((lang, path))
    return parsed


def get_language_prompts(language_data, tokenizer, prompt_tokens, sample_index):
    """Load one prompt per language from tokenized datasets."""
    examples = []
    for lang, data_path in parse_language_data_specs(language_data):
        dataset = load_from_disk(data_path)
        split = get_dataset_split(dataset)
        if len(split) == 0:
            raise ValueError(f"No rows found in dataset split for {lang}: {data_path}")

        idx = sample_index % len(split)
        source = f"{lang} ({data_path}, idx={idx})"
        examples.append(make_prompt_example(split[idx], tokenizer, prompt_tokens, source))
    return examples


def run_inference_on_checkpoint(checkpoint_path, label, tier_keys, examples,
                                tokenizer, args, device, emit):
    """Load a checkpoint and generate C1..CN completions for all prompts."""
    emit(f"\n{'#'*70}")
    emit(f"# CHECKPOINT: {label}")
    emit(f"# Path: {checkpoint_path}")
    emit(f"{'#'*70}")

    model = GPTNeoForCausalLMTiered.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()

    for i, example in enumerate(examples):
        input_ids = example["input_ids"]
        prompt_text = example["prompt_text"]
        gt_text = example["ground_truth"]
        source = example["source"]

        emit(f"\n{'='*70}")
        emit(f"  EXAMPLE {i+1}: {source}")
        emit(f"{'='*70}")
        emit(f"\n  PROMPT ({input_ids.shape[1]} tokens):")
        emit(f"  {prompt_text}")

        if gt_text:
            emit(f"\n  GROUND TRUTH (next ~100 tokens):")
            emit(f"  {gt_text}")

        # C1 (public)
        response_c1 = generate(
            model, input_ids, tokenizer, args.max_new_tokens,
            args.temperature, args.top_p, args.do_sample, device
        )
        emit(f"\n  C1 (Public):")
        emit(f"  {response_c1}")

        # C2..CN (keyed tiers)
        for tier_label, key in tier_keys:
            model.apply_key(key)
            try:
                response_ck = generate(
                    model, input_ids, tokenizer, args.max_new_tokens,
                    args.temperature, args.top_p, args.do_sample, device
                )
            finally:
                model.unapply_key(key)
            emit(f"\n  {tier_label} (Keyed):")
            emit(f"  {response_ck}")

    # Free memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report_lines = []

    def emit(line=""):
        print(line)
        if args.output_file:
            report_lines.append(line)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    key_paths = []
    if args.key_path:
        key_paths.append(args.key_path)
    if args.key_paths:
        for path in args.key_paths:
            if path not in key_paths:
                key_paths.append(path)

    tier_keys = []
    for i, key_path in enumerate(key_paths):
        key = load_key(key_path)
        tier_label = f"C{i + 2}"
        tier_keys.append((tier_label, key))
        emit(
            f"Loaded {tier_label} key ({key_path}): "
            f"{len(key.attn_heads)} attention swaps, {len(key.mlp_cols)} MLP swaps"
        )

    if not tier_keys:
        emit("No key provided; running C1-only inference.")

    # Build prompts
    if args.language_data:
        emit(
            f"Loading one example per language from {len(args.language_data)} datasets "
            f"(sample_index={args.sample_index})"
        )
        examples = get_language_prompts(
            args.language_data, tokenizer, args.prompt_tokens, args.sample_index
        )
    elif args.data_path:
        emit(
            f"Loading {args.num_examples} examples from {args.data_path} "
            f"(starting sample_index={args.sample_index})"
        )
        examples = get_prompts_from_dataset(
            args.data_path, tokenizer, args.num_examples, args.prompt_tokens, args.sample_index
        )
    elif args.prompt:
        input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
        examples = [{
            "source": "manual_prompt",
            "input_ids": input_ids,
            "prompt_text": args.prompt,
            "ground_truth": "",
        }]
    else:
        raise ValueError("Must provide one of: --prompt, --data_path, or --language_data")

    # Assign labels to checkpoints
    labels = args.checkpoint_labels or [f"checkpoint_{i}" for i in range(len(args.checkpoint))]
    if len(labels) != len(args.checkpoint):
        raise ValueError(f"Got {len(args.checkpoint)} checkpoints but {len(labels)} labels")

    # Run inference on each checkpoint
    for ckpt_path, label in zip(args.checkpoint, labels):
        run_inference_on_checkpoint(
            ckpt_path, label, tier_keys, examples,
            tokenizer, args, device, emit
        )

    emit(f"\n{'='*70}")
    emit("Done.")

    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
            f.write("\n")
        print(f"Saved output to {args.output_file}")


if __name__ == "__main__":
    main()
