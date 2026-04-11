"""Gemini completion function compatible with alpaca_eval's fn_completions interface."""

from __future__ import annotations

import os
import time
from typing import Optional, Sequence, Union

import google.generativeai as genai


def gemini_completions(
    prompts: Sequence[str],
    model_name: str = "gemini-2.5-flash",
    max_tokens: Union[int, Sequence[int]] = 2048,
    temperature: Optional[float] = 0.0,
    top_p: Optional[float] = 1.0,
    num_procs: Optional[int] = 1,
    **kwargs,
) -> dict[str, list]:
    """Drop-in replacement for alpaca_eval's openai_completions using Gemini SDK."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    if isinstance(max_tokens, int):
        max_tokens = [max_tokens] * len(prompts)

    completions = []
    price_per_example = []
    time_per_example = []
    completions_all = []

    for prompt, mt in zip(prompts, max_tokens):
        t0 = time.time()
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=mt,
                    temperature=temperature,
                    top_p=top_p,
                ),
            )
            text = response.text.strip() if response.text else ""
        except Exception as e:
            # Safety filter or other API error — treat as empty response
            text = ""
        dt = time.time() - t0

        completions.append(text)
        price_per_example.append(0.0)
        time_per_example.append(dt)
        completions_all.append([{"text": text, "total_tokens": 0}])

    return {
        "completions": completions,
        "price_per_example": price_per_example,
        "time_per_example": time_per_example,
        "completions_all": completions_all,
    }
