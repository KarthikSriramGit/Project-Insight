"""
Inference performance metrics: latency, TTFT, throughput, inter-token latency.
Metrics: p50/p90 total latency, p50/p90 Time to First Token, sustained throughput (tok/s), p50/p90 inter-token latency.
"""

import time
from typing import Any, Callable, Optional

import numpy as np


def pct(values: list[float], percentile: float) -> float:
    """Compute percentile of values."""
    if not values:
        return float("nan")
    return float(np.percentile(values, percentile))


def timed_generate(
    generate_fn: Callable[[], Any],
    device: str = "cuda",
    runs: int = 5,
) -> tuple[list[float], int]:
    """
    Time inference generation over multiple runs.

    Measure wall-clock, sync GPU if cuda, return latencies and token count.

    Args:
        generate_fn: Callable that returns (output_text_or_ids, num_tokens).
        device: cuda or cpu.
        runs: Number of timing runs.

    Returns:
        (latencies_sec, total_tokens) or (latencies_sec, 0) if token count unavailable.
    """
    import torch

    latencies = []
    total_tokens = 0

    for _ in range(runs):
        t0 = time.perf_counter()
        result = generate_fn()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        if isinstance(result, tuple) and len(result) >= 2:
            total_tokens += result[1]
        elif isinstance(result, str):
            total_tokens += len(result.split())
        elif isinstance(result, list):
            for r in result:
                if isinstance(r, str):
                    total_tokens += len(r.split())
                elif hasattr(r, "__len__"):
                    total_tokens += len(r)

    return latencies, total_tokens


def compute_metrics(
    total_latencies: list[float],
    first_token_latencies: Optional[list[float]] = None,
    token_counts: Optional[list[int]] = None,
    itl_sec_per_token: Optional[list[float]] = None,
) -> dict[str, float]:
    """
    Compute p50/p90 latency, TTFT, throughput from raw measurements.

    Metrics:
    - p50/p90 total latency (sec)
    - p50/p90 TTFT (sec)
    - sustained throughput (tokens/sec)
    - p50/p90 throughput (tokens/sec, from 1/ITL)
    - inter-token latency p50/p90 (ms/token)
    """
    metrics = {}

    if total_latencies:
        metrics["p50_latency_s"] = pct(total_latencies, 50)
        metrics["p90_latency_s"] = pct(total_latencies, 90)

    if first_token_latencies:
        metrics["p50_ttft_s"] = pct(first_token_latencies, 50)
        metrics["p90_ttft_s"] = pct(first_token_latencies, 90)

    if total_latencies and token_counts and sum(token_counts) > 0:
        total_tokens = sum(token_counts)
        total_time = sum(total_latencies)
        metrics["throughput_sustained_tok_s"] = total_tokens / total_time if total_time > 0 else float("nan")

    if itl_sec_per_token:
        metrics["itl_p50_ms_per_tok"] = 1000.0 * pct(itl_sec_per_token, 50)
        metrics["itl_p90_ms_per_tok"] = 1000.0 * pct(itl_sec_per_token, 90)
        p50_itl = pct(itl_sec_per_token, 50)
        p90_itl = pct(itl_sec_per_token, 90)
        metrics["throughput_p50_tok_s"] = 1.0 / p50_itl if p50_itl > 0 else float("nan")
        metrics["throughput_p90_tok_s"] = 1.0 / p90_itl if p90_itl > 0 else float("nan")

    return metrics
