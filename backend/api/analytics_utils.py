import math
from typing import Sequence

MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {
        "input": 2.50 / 1_000_000,
        "output": 10.00 / 1_000_000,
    },
    "gpt-4o-mini": {
        "input": 0.15 / 1_000_000,
        "output": 0.60 / 1_000_000,
    },
    "claude-3-5-haiku-latest": {
        "input": 0.80 / 1_000_000,
        "output": 4.00 / 1_000_000,
    },
    "claude-3-5-sonnet-latest": {
        "input": 3.00 / 1_000_000,
        "output": 15.00 / 1_000_000,
    },
}

DEFAULT_PRICING = {
    "input": 0.15 / 1_000_000,
    "output": 0.60 / 1_000_000,
}


def calculate_percentile(values: Sequence[float | int], percentile: float) -> float:
    """Calculate statistical percentile using standard linear interpolation."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])

    sorted_vals = sorted(float(v) for v in values)
    n = len(sorted_vals)
    rank = (percentile / 100.0) * (n - 1)
    k = int(math.floor(rank))
    d = rank - k

    if k + 1 < n:
        return round(sorted_vals[k] + d * (sorted_vals[k + 1] - sorted_vals[k]), 4)
    return round(float(sorted_vals[k]), 4)


def calculate_run_cost(input_tokens: int, output_tokens: int, model_used: str | None = None) -> float:
    """Compute run cost in USD based on input/output tokens and model pricing."""
    model_key = (model_used or "gpt-4o-mini").lower()
    rates = MODEL_PRICING.get(model_key, DEFAULT_PRICING)

    input_cost = (input_tokens or 0) * rates["input"]
    output_cost = (output_tokens or 0) * rates["output"]
    return input_cost + output_cost
