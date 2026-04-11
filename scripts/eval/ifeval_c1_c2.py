#!/usr/bin/env python3
"""IFEval C1 vs C2: instruction-following accuracy for public vs keyed tiers.

Generates responses from C1 (public) and C2 (keyed) on the IFEval benchmark,
then evaluates instruction-following accuracy using deterministic checkers.

Requires: pip install google-research/instruction_following_eval or clone it
and add to PYTHONPATH.

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

from instruction_following_eval import evaluation_lib


def parse_args():
    p = argparse.ArgumentParser(description="IFEval C1 vs C2")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--key_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--max_instances", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=512)
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

@torch.no_grad()
def generate_batched(
    model, tokenizer, prompts: list[str], indices: list[int],
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
        batch_prompts = [prompts[i] for i in batch_idxs]

        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        out_ids = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature, top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
        for bi, idx in enumerate(batch_idxs):
            prompt_len = int(attention_mask[bi].sum().item())
            gen_ids = out_ids[bi, prompt_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            records.append({"idx": idx, "response": text})
    return records


def gather_records(local_records: list[dict], is_distributed: bool, world_size: int):
    if not is_distributed:
        return local_records
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_records)
    return [r for recs in gathered for r in recs]


# ── IFEval scoring ──────────────────────────────────────────────────────────

def score_ifeval(examples: list[dict], responses: list[str]) -> dict:
    """Run IFEval strict + loose evaluation, return summary dict."""
    prompt_to_response = {ex["prompt"]: resp for ex, resp in zip(examples, responses)}

    inp_examples = []
    for ex in examples:
        # HF dataset includes all possible kwarg keys with None for unused ones;
        # build_description() only accepts the relevant subset, so drop Nones.
        cleaned_kwargs = [
            {k: v for k, v in kw.items() if v is not None}
            for kw in ex["kwargs"]
        ]
        inp_examples.append(evaluation_lib.InputExample(
            key=ex["key"],
            instruction_id_list=ex["instruction_id_list"],
            prompt=ex["prompt"],
            kwargs=cleaned_kwargs,
        ))

    strict_results = []
    loose_results = []
    for inp in inp_examples:
        strict_results.append(
            evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
        )
        loose_results.append(
            evaluation_lib.test_instruction_following_loose(inp, prompt_to_response)
        )

    # Prompt-level: all instructions satisfied
    strict_prompt_acc = sum(1 for r in strict_results if r.follow_all_instructions) / len(strict_results)
    loose_prompt_acc = sum(1 for r in loose_results if r.follow_all_instructions) / len(loose_results)

    # Instruction-level
    strict_instr_total = sum(len(r.follow_instruction_list) for r in strict_results)
    strict_instr_follow = sum(sum(r.follow_instruction_list) for r in strict_results)
    loose_instr_total = sum(len(r.follow_instruction_list) for r in loose_results)
    loose_instr_follow = sum(sum(r.follow_instruction_list) for r in loose_results)

    return {
        "strict_prompt_accuracy": strict_prompt_acc,
        "strict_instruction_accuracy": strict_instr_follow / strict_instr_total,
        "loose_prompt_accuracy": loose_prompt_acc,
        "loose_instruction_accuracy": loose_instr_follow / loose_instr_total,
        "n_prompts": len(inp_examples),
        "n_instructions": strict_instr_total,
    }


# ── main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device, rank, world_size, is_distributed = setup_distributed(args.device)
    is_main = rank == 0

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load IFEval
    ds = load_dataset("google/IFEval", split="train")
    if args.max_instances is not None:
        ds = ds.select(range(min(args.max_instances, len(ds))))
    examples = [dict(ex) for ex in ds]
    prompts = [ex["prompt"] for ex in examples]

    rank_indices = list(range(rank, len(prompts), world_size))

    # Model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = GPTNeoForCausalLMTiered.from_pretrained(args.checkpoint)
    model.to(device).eval()
    key = load_key(args.key_path)

    # C1 (public)
    c1_local = generate_batched(
        model, tokenizer, prompts, rank_indices,
        args.batch_size, args.max_new_tokens, args.temperature,
        args.top_p, args.do_sample, device,
        desc=f"Rank {rank} C1", position=2 * rank,
    )

    # C2 (keyed)
    model.apply_key(key)
    try:
        c2_local = generate_batched(
            model, tokenizer, prompts, rank_indices,
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

    # Sort by original index
    c1_all.sort(key=lambda r: r["idx"])
    c2_all.sort(key=lambda r: r["idx"])
    c1_responses = [r["response"] for r in c1_all]
    c2_responses = [r["response"] for r in c2_all]

    # Save raw outputs
    c1_path = Path(args.output_dir) / "ifeval_c1_outputs.json"
    c2_path = Path(args.output_dir) / "ifeval_c2_outputs.json"
    with open(c1_path, "w") as f:
        json.dump([{"prompt": p, "response": r} for p, r in zip(prompts, c1_responses)], f, indent=2)
    with open(c2_path, "w") as f:
        json.dump([{"prompt": p, "response": r} for p, r in zip(prompts, c2_responses)], f, indent=2)

    # Score
    print("\nScoring C1 (public)...")
    c1_scores = score_ifeval(examples, c1_responses)
    print("\nScoring C2 (keyed)...")
    c2_scores = score_ifeval(examples, c2_responses)

    results = {"c1": c1_scores, "c2": c2_scores}
    results_path = Path(args.output_dir) / "ifeval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("IFEval Results")
    print("=" * 60)
    for tier, scores in results.items():
        print(f"\n  {tier.upper()}:")
        print(f"    Strict prompt acc:       {scores['strict_prompt_accuracy']:.4f}")
        print(f"    Strict instruction acc:  {scores['strict_instruction_accuracy']:.4f}")
        print(f"    Loose prompt acc:        {scores['loose_prompt_accuracy']:.4f}")
        print(f"    Loose instruction acc:   {scores['loose_instruction_accuracy']:.4f}")
    print(f"\n  Prompts: {c1_scores['n_prompts']}  |  Instructions: {c1_scores['n_instructions']}")
    print("=" * 60)

    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
