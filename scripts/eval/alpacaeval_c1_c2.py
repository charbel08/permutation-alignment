#!/usr/bin/env python3
"""AlpacaEval C1 vs C2 with Gemini judge.

Generates responses from C1 (public) and C2 (keyed) on AlpacaEval prompts,
then computes C2-vs-C1 win rate using Gemini as the LLM judge.

Designed to run with `torchrun` for multi-GPU generation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import load_key


def parse_args():
    p = argparse.ArgumentParser(description="AlpacaEval C1 vs C2")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--key_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--annotators_config", type=str, required=True)

    p.add_argument("--dataset_name", type=str, default="tatsu-lab/alpaca_eval")
    p.add_argument("--dataset_config", type=str, default="alpaca_eval")
    p.add_argument("--dataset_split", type=str, default="eval")
    p.add_argument("--max_instances", type=int, default=None)

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--do_sample", action="store_true", default=False)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return p.parse_args()


# ── distributed helpers ─────────────────────────────────────────────────────

def setup_distributed(device_arg: str):
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        if device_arg == "cpu":
            device = torch.device("cpu")
        elif device_arg == "cuda":
            device = torch.device("cuda")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, 0, 1, False

    if device_arg == "cpu" or not torch.cuda.is_available():
        backend = "gloo"
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"

    dist.init_process_group(backend=backend)
    return device, dist.get_rank(), dist.get_world_size(), True


def cleanup_distributed(is_distributed: bool):
    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# ── generation ──────────────────────────────────────────────────────────────

def build_prompt(example: dict) -> str:
    instruction = str(example.get("instruction", "")).strip()
    input_text = str(example.get("input", "")).strip()
    if input_text:
        return f"{instruction}\n\nInput:\n{input_text}"
    return instruction


@torch.no_grad()
def generate_batched(
    model, tokenizer, examples: list[dict], indices: list[int],
    batch_size: int, max_new_tokens: int, temperature: float,
    top_p: float, do_sample: bool, device: torch.device,
    desc: str, position: int,
) -> list[dict]:
    records = []
    pbar = tqdm(
        range(0, len(indices), batch_size),
        total=(len(indices) + batch_size - 1) // batch_size,
        desc=desc, position=position, leave=True,
    )
    for start in pbar:
        batch_idxs = indices[start : start + batch_size]
        batch_examples = [examples[i] for i in batch_idxs]
        prompts = [build_prompt(ex) for ex in batch_examples]

        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        out_ids = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature, top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
        for bi, ex in enumerate(batch_examples):
            prompt_len = int(attention_mask[bi].sum().item())
            gen_ids = out_ids[bi, prompt_len:]
            output = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            rec = {"instruction": ex["instruction"], "output": output, "idx": ex["_idx"]}
            if "input" in ex and ex["input"] is not None:
                rec["input"] = ex["input"]
            records.append(rec)
    return records


def gather_records(local_records: list[dict], is_distributed: bool, world_size: int):
    if not is_distributed:
        return local_records
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_records)
    return [r for recs in gathered for r in recs]


# ── dataset loading ─────────────────────────────────────────────────────────

def load_alpacaeval_examples(args):
    try:
        return load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    except Exception as exc:
        if "dataset scripts are no longer supported" in str(exc).lower():
            url = "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json"
            print(f"Falling back to raw JSON: {url}")
            return load_dataset("json", data_files=url, split="train")
        raise


# ── main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device, rank, world_size, is_distributed = setup_distributed(args.device)
    is_main = rank == 0

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)

    ds = load_alpacaeval_examples(args)
    if args.max_instances is not None:
        ds = ds.select(range(min(args.max_instances, len(ds))))
    examples = []
    for i, ex in enumerate(ds):
        e = dict(ex)
        e["_idx"] = i
        examples.append(e)

    rank_indices = list(range(rank, len(examples), world_size))

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = GPTNeoForCausalLMTiered.from_pretrained(args.checkpoint)
    model.to(device).eval()
    key = load_key(args.key_path)

    # C1 (public)
    c1_local = generate_batched(
        model, tokenizer, examples, rank_indices,
        args.batch_size, args.max_new_tokens, args.temperature,
        args.top_p, args.do_sample, device,
        desc=f"Rank {rank} C1", position=2 * rank,
    )

    # C2 (keyed)
    model.apply_key(key)
    try:
        c2_local = generate_batched(
            model, tokenizer, examples, rank_indices,
            args.batch_size, args.max_new_tokens, args.temperature,
            args.top_p, args.do_sample, device,
            desc=f"Rank {rank} C2", position=2 * rank + 1,
        )
    finally:
        model.unapply_key(key)

    if is_distributed:
        dist.barrier()

    c1_all = gather_records(c1_local, is_distributed, world_size)
    c2_all = gather_records(c2_local, is_distributed, world_size)

    if not is_main:
        cleanup_distributed(is_distributed)
        return

    c1_all.sort(key=lambda r: r["idx"])
    c2_all.sort(key=lambda r: r["idx"])

    for r in c1_all:
        r["generator"] = "C1_public"
    for r in c2_all:
        r["generator"] = "C2_keyed"

    c1_path = Path(args.output_dir) / "alpacaeval_c1_outputs.json"
    c2_path = Path(args.output_dir) / "alpacaeval_c2_outputs.json"
    with open(c1_path, "w") as f:
        json.dump(c1_all, f, indent=2)
    with open(c2_path, "w") as f:
        json.dump(c2_all, f, indent=2)
    print(f"Saved C1 outputs: {c1_path}")
    print(f"Saved C2 outputs: {c2_path}")

    # Register gemini decoder with alpaca_eval
    from gemini_decoder import gemini_completions
    import alpaca_eval.decoders as decoders_mod
    decoders_mod.gemini_completions = gemini_completions

    from alpaca_eval.main import evaluate

    alpaca_out = str(Path(args.output_dir) / "alpaca_eval")
    print("\nRunning AlpacaEval with Gemini judge...")
    evaluate(
        model_outputs=str(c2_path),
        reference_outputs=str(c1_path),
        annotators_config=args.annotators_config,
        name="C2_keyed",
        output_path=alpaca_out,
    )
    print(f"\nAlpacaEval results: {alpaca_out}")
    print("Win-rate is C2 vs C1 (public tier baseline).")

    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
