#!/usr/bin/env python3
"""LLM-as-a-judge pairwise evaluation: C1 vs C2.

Phase 1 (multi-GPU via torchrun): Generate C1/C2 responses on AlpacaEval.
Phase 2 (rank 0 only, single GPU): Judge each pair with a local LLM.

Both phases use plain transformers — no vLLM, no external APIs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import load_key


JUDGE_PROMPT = """\
You are a fair and impartial judge. You will be given a user instruction and two \
AI-generated responses (Response A and Response B). Your job is to decide which \
response better follows the instruction and is more helpful, relevant, accurate, \
and well-written.

Evaluate on these criteria:
1. Instruction following: Does the response address what was asked?
2. Helpfulness: Is the response useful and informative?
3. Accuracy: Is the information correct?
4. Writing quality: Is the response clear, coherent, and well-structured?

After your analysis, output your final verdict on its own line in exactly this format:
  [[A]] if Response A is better
  [[B]] if Response B is better
  [[tie]] if they are equally good

[User Instruction]
{instruction}

[Response A]
{response_a}

[Response B]
{response_b}

Judge:"""


def parse_args():
    p = argparse.ArgumentParser(description="LLM-as-judge C1 vs C2")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--key_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--max_instances", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--do_sample", action="store_true", default=False)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    p.add_argument("--judge_model", type=str, default="openai/gpt-oss-120b")
    p.add_argument("--judge_batch_size", type=int, default=4)
    p.add_argument("--judge_max_tokens", type=int, default=1024)
    return p.parse_args()


# ── distributed helpers ─────────────────────────────────────────────────────

def setup_distributed(device_arg: str):
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
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
        batch = [examples[i] for i in batch_idxs]
        prompts = [build_prompt(ex) for ex in batch]

        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        out_ids = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature, top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
        for bi, (ex, idx) in enumerate(zip(batch, batch_idxs)):
            prompt_len = int(attention_mask[bi].sum().item())
            gen_ids = out_ids[bi, prompt_len:]
            output = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            records.append({"idx": idx, "instruction": ex["instruction"], "output": output})
    return records


def gather_records(local_records: list[dict], is_distributed: bool, world_size: int):
    if not is_distributed:
        return local_records
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_records)
    return [r for recs in gathered for r in recs]


def load_alpacaeval_examples():
    try:
        return load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")
    except Exception as exc:
        if "dataset scripts are no longer supported" in str(exc).lower():
            url = "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json"
            return load_dataset("json", data_files=url, split="train")
        raise


# ── judging ─────────────────────────────────────────────────────────────────

def parse_verdict(text: str) -> str:
    match = re.search(r"\[\[(A|B|tie)\]\]", text, re.IGNORECASE)
    if match:
        v = match.group(1).lower()
        return "tie" if v == "tie" else v.upper()
    return "error"


@torch.no_grad()
def run_judge(c1_outputs: list[dict], c2_outputs: list[dict], args) -> list[dict]:
    """Run pairwise judging with a local transformers model."""
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Build all judge prompts with position debiasing
    judge_prompts = []
    swap_flags = []
    for c1, c2 in zip(c1_outputs, c2_outputs):
        swap = random.random() < 0.5
        swap_flags.append(swap)
        resp_a, resp_b = (c2["output"], c1["output"]) if swap else (c1["output"], c2["output"])
        text = JUDGE_PROMPT.format(
            instruction=c1["instruction"], response_a=resp_a, response_b=resp_b,
        )
        messages = [{"role": "user", "content": text}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        judge_prompts.append(formatted)

    raw_outputs = []
    for start in tqdm(range(0, len(judge_prompts), args.judge_batch_size), desc="Judge"):
        batch = judge_prompts[start : start + args.judge_batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

        out_ids = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=args.judge_max_tokens,
            do_sample=False, temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        for bi in range(len(batch)):
            prompt_len = int(attention_mask[bi].sum().item())
            gen_ids = out_ids[bi, prompt_len:]
            raw_outputs.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

    results = []
    for i, raw in enumerate(raw_outputs):
        verdict = parse_verdict(raw)
        if verdict == "A":
            winner = "C2" if swap_flags[i] else "C1"
        elif verdict == "B":
            winner = "C1" if swap_flags[i] else "C2"
        else:
            winner = verdict
        results.append({
            "instruction": c1_outputs[i]["instruction"],
            "c1_output": c1_outputs[i]["output"],
            "c2_output": c2_outputs[i]["output"],
            "swapped": swap_flags[i],
            "raw_verdict": verdict,
            "winner": winner,
            "judge_reasoning": raw,
        })
    return results


# ── main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device, rank, world_size, is_distributed = setup_distributed(args.device)
    is_main = rank == 0

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)

    ds = load_alpacaeval_examples()
    if args.max_instances is not None:
        ds = ds.select(range(min(args.max_instances, len(ds))))
    examples = [dict(ex) for ex in ds]

    rank_indices = list(range(rank, len(examples), world_size))

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = GPTNeoForCausalLMTiered.from_pretrained(args.checkpoint)
    model.to(device).eval()
    key = load_key(args.key_path)

    c1_local = generate_batched(
        model, tokenizer, examples, rank_indices,
        args.batch_size, args.max_new_tokens, args.temperature,
        args.top_p, args.do_sample, device,
        desc=f"Rank {rank} C1", position=2 * rank,
    )

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

    # Fully tear down distributed on all ranks; non-main exits here.
    del model
    torch.cuda.empty_cache()
    cleanup_distributed(is_distributed)

    if not is_main:
        return

    c1_all.sort(key=lambda r: r["idx"])
    c2_all.sort(key=lambda r: r["idx"])

    c1_path = Path(args.output_dir) / "c1_outputs.json"
    c2_path = Path(args.output_dir) / "c2_outputs.json"
    with open(c1_path, "w") as f:
        json.dump(c1_all, f, indent=2)
    with open(c2_path, "w") as f:
        json.dump(c2_all, f, indent=2)
    print(f"Saved C1 outputs: {c1_path}")
    print(f"Saved C2 outputs: {c2_path}")

    print(f"\nLoading judge model: {args.judge_model}")
    results = run_judge(c1_all, c2_all, args)

    n = len(results)
    c2_wins = sum(1 for r in results if r["winner"] == "C2")
    c1_wins = sum(1 for r in results if r["winner"] == "C1")
    ties = sum(1 for r in results if r["winner"] == "tie")
    errors = sum(1 for r in results if r["winner"] == "error")

    summary = {
        "judge_model": args.judge_model,
        "n": n,
        "c2_win_rate": c2_wins / n if n else 0,
        "c1_win_rate": c1_wins / n if n else 0,
        "tie_rate": ties / n if n else 0,
        "error_rate": errors / n if n else 0,
        "c2_wins": c2_wins, "c1_wins": c1_wins, "ties": ties, "errors": errors,
    }

    results_path = Path(args.output_dir) / "judge_results.json"
    with open(results_path, "w") as f:
        json.dump({"summary": summary, "details": results}, f, indent=2)

    print("\n" + "=" * 60)
    print(f"LLM Judge Results  (judge: {args.judge_model})")
    print("=" * 60)
    print(f"  C2 win rate:  {summary['c2_win_rate']:.1%}  ({c2_wins}/{n})")
    print(f"  C1 win rate:  {summary['c1_win_rate']:.1%}  ({c1_wins}/{n})")
    print(f"  Tie rate:     {summary['tie_rate']:.1%}  ({ties}/{n})")
    if errors:
        print(f"  Parse errors: {summary['error_rate']:.1%}  ({errors}/{n})")
    print("=" * 60)


if __name__ == "__main__":
    main()
