"""
Benchmark: pandas vs cuDF for telemetry loading and analytics. Measures wall-clock time, peak memory, and operation throughput for load, groupby, filter, sort.
"""

import gc
import time
from pathlib import Path
from typing import Any, Callable, Optional, Union

import pandas as pd

try:
    import cudf

    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False


def _get_memory_mb() -> float:
    """Approximate current process memory usage in MB. Cross-platform via psutil."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    try:
        import resource
        import sys

        val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return val / (1024 * 1024)
        return val / 1024
    except Exception:
        return 0.0


def _measure(
    fn: Callable[[], Any],
    warmup: int = 1,
) -> tuple[float, float, Any]:
    """Run function, return (elapsed_sec, peak_memory_mb, result)."""
    for _ in range(warmup):
        _ = fn()
    gc.collect()
    m0 = _get_memory_mb()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    m1 = _get_memory_mb()
    peak_mb = max(m0, m1)
    return elapsed, peak_mb, result


def load_pandas(path: Path) -> pd.DataFrame:
    """Load with pandas."""
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_cudf(path: Path, spill: bool = True) -> "cudf.DataFrame":
    """Load with cuDF and UVM spill."""
    if not CUDF_AVAILABLE:
        raise RuntimeError("cuDF not installed")
    cudf.set_option("spill", spill)
    if path.suffix.lower() in (".parquet", ".pq"):
        return cudf.read_parquet(path)
    return cudf.read_csv(path)


def run_benchmark(
    path: Union[str, Path],
    operations: Optional[list[str]] = None,
    spill: bool = True,
) -> dict[str, dict[str, float]]:
    """
    Run pandas vs cuDF benchmark on telemetry file.

    Args:
        path: Path to Parquet or CSV.
        operations: List of ops to benchmark: "load", "groupby", "filter", "sort".
                   Default all.
        spill: Enable cuDF UVM spill.

    Returns:
        Dict mapping backend ("pandas", "cudf") to dict of operation -> {time_s, memory_mb}.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    operations = operations or ["load", "groupby", "filter", "sort"]
    results: dict[str, dict[str, float]] = {"pandas": {}, "cudf": {}}

    def _pandas_load():
        return load_pandas(path)

    elapsed, mem, df_pd = _measure(_pandas_load)
    results["pandas"]["load"] = {"time_s": elapsed, "memory_mb": mem}

    if "groupby" in operations:
        def _pd_groupby():
            return df_pd.groupby("vehicle_id")["brake_pressure_pct"].agg(["max", "mean"])

        elapsed, mem, _ = _measure(_pd_groupby)
        results["pandas"]["groupby"] = {"time_s": elapsed, "memory_mb": mem}

    if "filter" in operations and "brake_pressure_pct" in df_pd.columns:
        def _pd_filter():
            return df_pd[df_pd["brake_pressure_pct"] > 90]

        elapsed, mem, _ = _measure(_pd_filter)
        results["pandas"]["filter"] = {"time_s": elapsed, "memory_mb": mem}

    if "sort" in operations:
        def _pd_sort():
            return df_pd.sort_values("timestamp_ns")

        elapsed, mem, _ = _measure(_pd_sort)
        results["pandas"]["sort"] = {"time_s": elapsed, "memory_mb": mem}

    del df_pd
    gc.collect()

    if not CUDF_AVAILABLE:
        return results

    def _cudf_load():
        return load_cudf(path, spill=spill)

    elapsed, mem, df_cu = _measure(_cudf_load)
    results["cudf"]["load"] = {"time_s": elapsed, "memory_mb": mem}

    if "groupby" in operations and "vehicle_id" in df_cu.columns:
        def _cu_groupby():
            return df_cu.groupby("vehicle_id")["brake_pressure_pct"].agg(["max", "mean"])

        elapsed, mem, _ = _measure(_cu_groupby)
        results["cudf"]["groupby"] = {"time_s": elapsed, "memory_mb": mem}

    if "filter" in operations and "brake_pressure_pct" in df_cu.columns:
        def _cu_filter():
            return df_cu[df_cu["brake_pressure_pct"] > 90]

        elapsed, mem, _ = _measure(_cu_filter)
        results["cudf"]["filter"] = {"time_s": elapsed, "memory_mb": mem}

    if "sort" in operations:
        def _cu_sort():
            return df_cu.sort_values("timestamp_ns")

        elapsed, mem, _ = _measure(_cu_sort)
        results["cudf"]["sort"] = {"time_s": elapsed, "memory_mb": mem}

    del df_cu
    gc.collect()

    return results


def benchmark_to_dataframe(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Convert benchmark results to a pandas DataFrame for display/plotting."""
    rows = []
    for backend, ops in results.items():
        for op, metrics in ops.items():
            rows.append(
                {
                    "backend": backend,
                    "operation": op,
                    "time_s": metrics["time_s"],
                    "memory_mb": metrics["memory_mb"],
                }
            )
    return pd.DataFrame(rows)
