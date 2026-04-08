#!/usr/bin/env python3
"""Inspect which bios have non-zero memorization accuracy at baseline.

Loads a pretrained model (no finetuning), runs memorization eval,
and prints the bios where the model gets any value tokens right.

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
    parser.add_argument("--batch_size", type=int, default=32)
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
        print(f"  {config_name} evaluation")
        print(f"{'='*70}")

        if cfg_key is not None:
            apply_permutation(model, cfg_key, plan=plan)

        hits = []
        with torch.no_grad():
            for i in range(0, len(bios), args.batch_size):
                batch_bios = bios[i:i + args.batch_size]
                batch_spans = spans[i:i + args.batch_size]

                encodings = [
                    tokenizer(b["text"], add_special_tokens=False,
                              return_tensors="pt")["input_ids"].squeeze(0)
                    for b in batch_bios
                ]
                max_len = max(e.shape[0] for e in encodings)
                ids = torch.full((len(batch_bios), max_len),
                                 tokenizer.pad_token_id, dtype=torch.long)
                mask = torch.zeros(len(batch_bios), max_len, dtype=torch.long)
                for j, enc in enumerate(encodings):
                    ids[j, :enc.shape[0]] = enc
                    mask[j, :enc.shape[0]] = 1

                logits = model(ids.to(device), attention_mask=mask.to(device)).logits

                for j, (bio, span) in enumerate(zip(batch_bios, batch_spans)):
                    if span is None:
                        continue
                    vs, ve = span
                    seq_len = encodings[j].shape[0]
                    if vs < 1 or ve > seq_len:
                        continue

                    pred_logits = logits[j, vs - 1:ve - 1, :]
                    targets = ids[j, vs:ve].to(device)
                    n_tok = targets.shape[0]
                    if n_tok == 0:
                        continue

                    top3 = pred_logits.topk(3, dim=-1).indices
                    top1_hits = (top3[:, 0] == targets).float()
                    top3_hits = (top3 == targets.unsqueeze(-1)).any(dim=-1).float()

                    top1_acc = top1_hits.mean().item()
                    top3_acc = top3_hits.mean().item()

                    if top1_acc > 0 or top3_acc > 0:
                        # Decode token-by-token details
                        token_details = []
                        for t in range(n_tok):
                            target_tok = tokenizer.decode([targets[t].item()])
                            pred1 = tokenizer.decode([top3[t, 0].item()])
                            pred2 = tokenizer.decode([top3[t, 1].item()])
                            pred3 = tokenizer.decode([top3[t, 2].item()])
                            hit1 = "✓" if top1_hits[t] > 0 else "✗"
                            hit3 = "✓" if top3_hits[t] > 0 else "✗"
                            token_details.append({
                                "target": target_tok,
                                "top3": [pred1, pred2, pred3],
                                "top1_hit": hit1,
                                "top3_hit": hit3,
                            })

                        hits.append({
                            "name": bio["name"],
                            "target_attr": bio["target_attr"],
                            "value": _bio_value_string(bio),
                            "top1_acc": top1_acc,
                            "top3_acc": top3_acc,
                            "token_details": token_details,
                            "prefix": bio["prefix"],
                        })

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
            # Sort by top3_acc descending
            for h in sorted(attr_hits, key=lambda x: -x["top3_acc"])[:10]:
                print(f"\n  {h['name']} | value={h['value']!r} | "
                      f"top1={h['top1_acc']:.2f} top3={h['top3_acc']:.2f}")
                print(f"  prefix: {h['prefix'][:80]}...")
                for td in h["token_details"]:
                    print(f"    target={td['target']!r:>12}  "
                          f"top3={td['top3']}  "
                          f"top1={td['top1_hit']} top3={td['top3_hit']}")

        # Summary stats
        print(f"\n--- Summary ---")
        total_valid = valid
        total_hits = len(hits)
        print(f"  Bios with any top1 hit: "
              f"{sum(1 for h in hits if h['top1_acc'] > 0)}/{total_valid}")
        print(f"  Bios with any top3 hit: "
              f"{total_hits}/{total_valid}")
        for attr in sorted(by_attr):
            print(f"  {attr:>12}: {len(by_attr[attr])} hits")


if __name__ == "__main__":
    main()
