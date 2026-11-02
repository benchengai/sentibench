"""Published per-token prices, used to report what each LLM call costs.

USD per million tokens: (input, cached input, output), standard tier, short
context. Local models have no per-token price; their cost is machine time,
reported as latency. Same table as ElicitSense's `pricing.py`.
"""

from __future__ import annotations

AS_OF = "2026-11-01"
SOURCE = "https://developers.openai.com/api/docs/pricing"

PRICES = {
    "gpt-6.1-sol": (2.00, 0.10, 10.00),
    "gpt-6.1-terra": (1.00, 0.05, 5.00),
    "gpt-6-luna": (0.10, 0.01, 0.50),
}

# USD per million input tokens.
EMBEDDING_PRICES = {
    "text-embedding-3-small": 0.02,
}


def _lookup(table: dict, model: str):
    """Exact id, or a dated snapshot of a listed id (gpt-6.1-sol-2026-...)."""
    if model in table:
        return table[model]
    return next((p for m, p in table.items() if model.startswith(m + "-")), None)


def cost_usd(provider: str, model: str, input_tokens: int, cached_input_tokens: int,
             output_tokens: int) -> float | None:
    if provider == "ollama":
        return 0.0
    if (p := _lookup(PRICES, model)) is None:
        return None
    uncached = input_tokens - cached_input_tokens
    return (uncached * p[0] + cached_input_tokens * p[1] + output_tokens * p[2]) / 1_000_000


def embedding_cost_usd(model: str, input_tokens: int) -> float | None:
    if (p := _lookup(EMBEDDING_PRICES, model)) is None:
        return None
    return input_tokens * p / 1_000_000
