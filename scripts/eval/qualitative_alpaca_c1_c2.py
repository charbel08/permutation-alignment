#!/usr/bin/env python3
"""Qualitative C1 vs C2 generation on a sampled Alpaca prompt."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import load_key


PROMPT_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:"
)

PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:"
)


def parse_args():
    p = argparse.ArgumentParser(description="Qualitative C1 vs C2 on one Alpaca prompt")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--key_path", type=str, required=True)
    p.add_argument("--alpaca_json", type=str, required=True)
    p.add_argument("--sample_index", type=int, default=None,
                   help="If omitted, sample randomly with --seed")
    p.add_argument("--num_samples", type=int, default=1,
                   help="Number of different Alpaca prompts to evaluate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tokenizer_path", type=str, default=None)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--do_sample", action="store_true", default=False)
    p.add_argument("--output_json", type=str, default=None,
                   help="Optional path to save prompt/C1/C2 outputs")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_prompt(example: dict) -> str:
    instruction = str(example.get("instruction", "")).strip()
    input_text = str(example.get("input", "")).strip()
    if input_text:
        return PROMPT_INPUT.format(instruction=instruction, input=input_text)
    return PROMPT_NO_INPUT.format(instruction=instruction)


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    device: torch.device,
) -> str:
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    out_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    with open(args.alpaca_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list) or len(records) == 0:
        raise ValueError(f"Invalid/empty Alpaca JSON: {args.alpaca_json}")

    if args.num_samples <= 0:
        raise ValueError("--num_samples must be > 0")

    if args.sample_index is None:
        rng = random.Random(args.seed)
        k = min(args.num_samples, len(records))
        sample_indices = rng.sample(range(len(records)), k=k)
    else:
        start = args.sample_index % len(records)
        sample_indices = [((start + i) % len(records)) for i in range(args.num_samples)]

    tok_path = args.tokenizer_path or args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPTNeoForCausalLMTiered.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()

    key = load_key(args.key_path)

    outputs = []
    for sample_index in sample_indices:
        sample = records[sample_index]
        prompt = build_prompt(sample)

        c1 = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            device=device,
        )

        model.apply_key(key)
        try:
            c2 = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
                device=device,
            )
        finally:
            model.unapply_key(key)

        result = {
            "sample_index": sample_index,
            "instruction": sample.get("instruction", ""),
            "input": sample.get("input", ""),
            "reference_output": sample.get("output", ""),
            "prompt": prompt,
            "c1_output": c1,
            "c2_output": c2,
        }
        outputs.append(result)

        print("=" * 80)
        print(f"Alpaca sample index: {sample_index} / {len(records)}")
        print("=" * 80)
        print("\n[Instruction]")
        print(str(result["instruction"]).strip())
        print("\n[Input]")
        print(str(result["input"]).strip() or "<empty>")
        print("\n[Reference Output (dataset)]")
        print(str(result["reference_output"]).strip())
        print("\n[Prompt Given To Model]")
        print(result["prompt"])
        print("\n[C1 (Public) Response]")
        print(result["c1_output"])
        print("\n[C2 (Keyed) Response]")
        print(result["c2_output"])
        print("\n" + "=" * 80)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sample_indices": sample_indices,
            "num_samples": len(outputs),
            "num_records": len(records),
            "outputs": outputs,
            "checkpoint": args.checkpoint,
            "key_path": args.key_path,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Saved qualitative output to: {out_path}")


if __name__ == "__main__":
    main()
