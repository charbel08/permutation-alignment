#!/usr/bin/env python3
"""Qualitative C1 vs C2 comparison on fixed EN/ES continuation prompts.

Prompts are written for base-model behavior (plain text continuation), not
instruction/chat tuning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from tiered.model import GPTNeoForCausalLMTiered
from tiered.permutation import load_key


PROMPTS = [
    {
        "lang": "en",
        "id": "en_1",
        "prompt": (
            "Article excerpt:\n"
            "Regular exercise is linked to better mood, improved sleep quality, and long-term cardiovascular health. "
            "Researchers also note that moderate routines are easier to sustain over time.\n\n"
            "Key takeaway:"
        ),
    },
    {
        "lang": "en",
        "id": "en_2",
        "prompt": (
            "Subject: Meeting schedule update\n"
            "Hi Alex,\n\n"
            "I hope you're doing well. I wanted to check whether we could move our meeting from Tuesday to Thursday, "
            "since"
        ),
    },
    {
        "lang": "en",
        "id": "en_3",
        "prompt": (
            "Travel note:\n"
            "Barcelona is known for modernist architecture, neighborhood markets, and late dinners. "
            "Most visitors recommend booking popular museums in advance.\n\n"
            "If you're visiting for three days, start with"
        ),
    },
    {
        "lang": "en",
        "id": "en_4",
        "prompt": (
            "Recipe draft:\n"
            "For a quick tomato pasta sauce, heat olive oil, add garlic, then stir in crushed tomatoes and salt. "
            "Simmer for 15 minutes and finish with basil.\n\n"
            "To make the flavor deeper, you can"
        ),
    },
    {
        "lang": "en",
        "id": "en_5",
        "prompt": (
            "Notebook entry:\n"
            "When debugging a training run, first verify the data pipeline, then confirm loss masking, and finally check "
            "learning-rate scheduling. Small shape mismatches often appear as unstable loss spikes.\n\n"
            "A practical first check is"
        ),
    },
    {
        "lang": "es",
        "id": "es_1",
        "prompt": (
            "Texto:\n"
            "La fotosíntesis es el proceso por el cual las plantas transforman la luz del sol en energía química. "
            "Gracias a este proceso, también liberan oxígeno al ambiente.\n\n"
            "Resumen breve:"
        ),
    },
    {
        "lang": "es",
        "id": "es_2",
        "prompt": (
            "Asunto: Gracias por tu ayuda\n"
            "Hola Marta,\n\n"
            "Quería escribirte para agradecerte tu apoyo en el proyecto de esta semana. "
            "Tu ayuda con"
        ),
    },
    {
        "lang": "es",
        "id": "es_3",
        "prompt": (
            "Nota de viaje:\n"
            "Sevilla combina historia, arquitectura y una vida de barrio muy activa. "
            "En primavera, el clima suele ser agradable para caminar por el centro.\n\n"
            "Para un itinerario de dos días, conviene"
        ),
    },
    {
        "lang": "es",
        "id": "es_4",
        "prompt": (
            "Borrador de receta:\n"
            "Para preparar una tortilla de patatas clásica, corta las patatas en láminas finas, "
            "cocínalas a fuego medio con cebolla y luego mézclalas con huevo batido.\n\n"
            "Un consejo para que quede jugosa es"
        ),
    },
    {
        "lang": "es",
        "id": "es_5",
        "prompt": (
            "Apunte técnico:\n"
            "Cuando un experimento de entrenamiento falla, conviene revisar primero los datos de entrada, "
            "luego la función de pérdida y por último la programación del learning rate.\n\n"
            "El primer paso práctico suele ser"
        ),
    },
]


def parse_args():
    p = argparse.ArgumentParser(description="Qualitative EN/ES prompts for C1 vs C2")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--key_path", type=str, required=True)
    p.add_argument("--tokenizer_path", type=str, default=None)
    p.add_argument("--max_new_tokens", type=int, default=192)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--do_sample", action="store_true", default=False)
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    tok_src = args.tokenizer_path or args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPTNeoForCausalLMTiered.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()
    key = load_key(args.key_path)

    outputs = []
    for i, item in enumerate(PROMPTS, start=1):
        prompt = item["prompt"]
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

        rec = {
            "index": i,
            "id": item["id"],
            "lang": item["lang"],
            "prompt": prompt,
            "c1_output": c1,
            "c2_output": c2,
        }
        outputs.append(rec)

        print("=" * 80)
        print(f"Prompt {i} [{item['lang']}] ({item['id']})")
        print("-" * 80)
        print("[Prompt]")
        print(prompt)
        print("\n[C1 (Public)]")
        print(c1)
        print("\n[C2 (Keyed)]")
        print(c2)
        print("=" * 80 + "\n")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": args.checkpoint,
            "key_path": args.key_path,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "do_sample": args.do_sample,
            "outputs": outputs,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Saved outputs to: {out_path}")


if __name__ == "__main__":
    main()
