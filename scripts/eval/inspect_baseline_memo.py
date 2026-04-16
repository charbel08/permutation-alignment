#!/usr/bin/env python3
"""Inspect which bios have non-zero memorization accuracy at baseline.

Loads a pretrained model (no finetuning), runs greedy-decode memorization
eval, and prints the bios where the model gets any value tokens right.

Usage:
    PYTHONPATH=./src python scripts/eval/inspect_baseline_memo.py \
        --checkpoint /path/to/pretrained \
        --bio_metadata /path/to/bios_metadata.json \
        [--key_path /path/to/key.json]  # for C2 eval
        [--eval_split test]
"""

import argparse
import json

import torch
from collections import defaultdict
from transformers import AutoTokenizer

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import load_key
from tiered.permutation.permute import apply_permutation, unapply_permutation, build_swap_plan
from tiered.train.finetune.private_finetune import _bio_value_string, _bio_value_span


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--bio_metadata", type=str, required=True)
    parser.add_argument("--key_path", type=str, default=None)
    parser.add_argument("--eval_split", type=str, default="test",
                        choices=["train", "test", "all"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    with open(args.bio_metadata) as f:
        meta = json.load(f)

    # Select bios by split
    if args.eval_split == "all":
        bios = meta["bios"]
    else:
        people = set(meta.get(f"{args.eval_split}_people", []))
        bios = [b for b in meta["bios"] if b["person_id"] in people]

    spans = [_bio_value_span(tokenizer, b) for b in bios]
    valid = sum(1 for s in spans if s is not None)
    print(f"Selected {len(bios)} bios ({args.eval_split} split), {valid} with valid spans")

    model = GPTNeoForCausalLMTiered.from_pretrained(args.checkpoint).to(device)
    model.eval()

    configs = [("C1", None)]
    if args.key_path:
        key = load_key(args.key_path)
        plan = build_swap_plan(model, key, device)
        configs.append(("C2", key))

    for config_name, cfg_key in configs:
        print(f"\n{'='*70}")
        print(f"  {config_name} evaluation (greedy decoding)")
        print(f"{'='*70}")

        if cfg_key is not None:
            apply_permutation(model, cfg_key, plan=plan)

        hits = []
        total_results = []
        with torch.no_grad():
            for i in range(len(bios)):
                bio = bios[i]
                span = spans[i]
                if span is None:
                    continue
                vs, ve = span

                enc = tokenizer(bio["text"], add_special_tokens=False,
                                return_tensors="pt")["input_ids"].squeeze(0)
                seq_len = enc.shape[0]
                if vs < 1 or ve > seq_len:
                    continue

                target_tokens = enc[vs:ve]
                n_tok = target_tokens.shape[0]
                if n_tok == 0:
                    continue

                # Feed prefix, greedy decode the rest
                prefix_ids = enc[:vs].unsqueeze(0).to(device)
                n_generate = seq_len - vs

                generated = prefix_ids
                for _ in range(n_generate):
                    logits = model(generated).logits
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)

                gen_value = generated[0, vs:ve].cpu()
                gen_all = generated[0, vs:].cpu()

                # Token-level accuracy
                token_hits = (gen_value == target_tokens).float()
                top1_acc = token_hits.mean().item()
                exact = (gen_value == target_tokens).all().item()

                # String-level metrics
                target_str = _bio_value_string(bio).strip().lower()
                gen_str = tokenizer.decode(gen_all.tolist()).strip().lower()
                contains = target_str in gen_str
                prefix_match = gen_str.startswith(target_str)

                result = {
                    "name": bio["name"],
                    "target_attr": bio["target_attr"],
                    "value": _bio_value_string(bio),
                    "top1_acc": top1_acc,
                    "exact_match": exact,
                    "contains": contains,
                    "prefix_match": prefix_match,
                    "gen_str": tokenizer.decode(gen_all.tolist()),
                    "prefix": bio["prefix"],
                }
                total_results.append(result)

                if top1_acc > 0 or contains or prefix_match:
                    # Token-by-token details
                    token_details = []
                    for t in range(n_tok):
                        target_tok = tokenizer.decode([target_tokens[t].item()])
                        pred_tok = tokenizer.decode([gen_value[t].item()])
                        hit = "Y" if token_hits[t] > 0 else "N"
                        token_details.append({
                            "target": target_tok,
                            "predicted": pred_tok,
                            "hit": hit,
                        })
                    result["token_details"] = token_details
                    hits.append(result)

        if cfg_key is not None:
            unapply_permutation(model, cfg_key, plan=plan)

        # Print results
        print(f"\n{len(hits)} bios with non-zero accuracy (out of {valid} valid)")

        # Group by attribute
        by_attr = defaultdict(list)
        for h in hits:
            by_attr[h["target_attr"]].append(h)

        for attr in sorted(by_attr):
            attr_hits = by_attr[attr]
            print(f"\n--- {attr} ({len(attr_hits)} hits) ---")
            for h in sorted(attr_hits, key=lambda x: -x["top1_acc"])[:10]:
                print(f"\n  {h['name']} | value={h['value']!r} | "
                      f"top1={h['top1_acc']:.2f} exact={h['exact_match']} "
                      f"contains={h['contains']} prefix={h['prefix_match']}")
                print(f"  generated: {h['gen_str']!r}")
                print(f"  context:   ...{h['prefix'][-60:]}")
                if "token_details" in h:
                    for td in h["token_details"]:
                        print(f"    target={td['target']!r:>12}  "
                              f"pred={td['predicted']!r:>12}  {td['hit']}")

        # Summary stats
        print(f"\n--- Summary ---")
        n_total = len(total_results)
        print(f"  Total evaluated:      {n_total}")
        print(f"  Any token hit:        "
              f"{sum(1 for r in total_results if r['top1_acc'] > 0)}/{n_total}")
        print(f"  Exact match:          "
              f"{sum(1 for r in total_results if r['exact_match'])}/{n_total}")
        print(f"  Contains:             "
              f"{sum(1 for r in total_results if r['contains'])}/{n_total}")
        print(f"  Prefix match:         "
              f"{sum(1 for r in total_results if r['prefix_match'])}/{n_total}")

        all_by_attr = defaultdict(list)
        for r in total_results:
            all_by_attr[r["target_attr"]].append(r)
        for attr in sorted(all_by_attr):
            recs = all_by_attr[attr]
            na = len(recs)
            print(f"  {attr:>12}: top1={sum(r['top1_acc'] for r in recs)/na:.3f}  "
                  f"exact={sum(r['exact_match'] for r in recs)/na:.3f}  "
                  f"contains={sum(r['contains'] for r in recs)/na:.3f}  "
                  f"prefix={sum(r['prefix_match'] for r in recs)/na:.3f}")


if __name__ == "__main__":
    main()
