"""Inference script for tiered alignment models.

Compare responses from public (C1) and keyed (C2) configurations.

Usage:
    python src/sgtm/train/inference.py \
        --checkpoint /path/to/checkpoint \
        --key_path examples/key_32m.json \
        --prompt "Once upon a time" \
        --max_length 100
"""

import argparse

import torch
from transformers import AutoTokenizer

from sgtm.model import GPTNeoForCausalLMSGTM
from sgtm.permutation import load_key


def parse_args():
    parser = argparse.ArgumentParser(description="Tiered alignment inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--key_path", type=str, required=True,
                        help="Path to permutation key JSON")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Input prompt")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true", default=True)
    return parser.parse_args()


def generate(model, tokenizer, prompt, max_length, temperature, top_p, do_sample, device):
    """Generate text from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model from {args.checkpoint}")
    model = GPTNeoForCausalLMSGTM.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    key = load_key(args.key_path)
    print(f"Loaded key with {len(key.attn_heads)} attention swaps, {len(key.mlp_cols)} MLP swaps")
    
    print("\n" + "="*60)
    print(f"PROMPT: {args.prompt}")
    print("="*60)
    
    # Generate without key (C1 - public)
    print("\n--- C1 (Public Model) ---")
    response_c1 = generate(
        model, tokenizer, args.prompt, args.max_length,
        args.temperature, args.top_p, args.do_sample, device
    )
    print(response_c1)
    
    # Generate with key (C2 - keyed)
    print("\n--- C2 (Keyed Model) ---")
    model.apply_key(key)
    response_c2 = generate(
        model, tokenizer, args.prompt, args.max_length,
        args.temperature, args.top_p, args.do_sample, device
    )
    model.unapply_key(key)
    print(response_c2)
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
