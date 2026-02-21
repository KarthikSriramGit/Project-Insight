"""
Benchmark: pandas vs cuDF vs cudf.pandas for telemetry loading and analytics.

Measures wall-clock time, peak memory, and operation throughput for load, groupby, filter, sort.
cudf.pandas is RAPIDS' drop-in pandas API that runs on GPU with zero code change.
Optional: run cudf.pandas in a subprocess to avoid mutating global pandas in the main process.
"""

import gc
import importlib
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, Union

import pandas as pd

try:
    import cudf

    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False

try:
    import cudf.pandas as _cudf_pandas_mod

    CUDF_PANDAS_AVAILABLE = True
except ImportError:
    CUDF_PANDAS_AVAILABLE = False


def _gpu_memory_mb() -> float:
    """Return current GPU memory allocated in MB, or 0 if not available."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    return 0.0


def _measure(
    fn: Callable[[], Any],
    warmup: int = 2,
    track_gpu_memory: bool = False,
) -> tuple[float, float, Any]:
    """Run function, return (elapsed_sec, peak_memory_mb, result).
    peak_memory_mb is host (tracemalloc) unless track_gpu_memory=True for GPU backends.
    """
    for _ in range(warmup):
        _ = fn()
    gc.collect()

    if track_gpu_memory:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
        except ImportError:
            pass

    try:
        import tracemalloc

        tracemalloc.start()
        t0 = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / (1024 * 1024)
        if track_gpu_memory:
            gpu_mb = _gpu_memory_mb()
            if gpu_mb > 0:
                peak_mb = gpu_mb  # Report GPU memory for GPU backends when requested
        return elapsed, peak_mb, result
    except Exception:
        t0 = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - t0
        return elapsed, 0.0, result


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


def _load_cudf_pandas(path: Path, spill: bool = True):
    """Load with cudf.pandas (pandas API on GPU). Must be called after cudf.pandas.install()."""
    if spill and CUDF_AVAILABLE:
        cudf.set_option("spill", spill)
    pd_gpu = sys.modules.get("pandas") or importlib.import_module("pandas")
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd_gpu.read_parquet(path)
    return pd_gpu.read_csv(path)


def _run_cudf_pandas_in_subprocess(
    path_str: str,
    operations: list[str],
    spill: bool,
    track_gpu_memory: bool,
    result_queue: Optional[Any] = None,
) -> dict[str, dict[str, float]]:
    """Run cudf.pandas benchmark in a separate process. Puts cudf_pandas results dict into result_queue if provided."""
    path = Path(path_str)
    results: dict[str, dict[str, float]] = {}

    if not CUDF_PANDAS_AVAILABLE or not CUDF_AVAILABLE:
        return results

    _cudf_pandas_mod.install()
    if spill:
        cudf.set_option("spill", spill)
    pd_gpu = importlib.import_module("pandas")

    def _load():
        return _load_cudf_pandas(path, spill=spill)

    elapsed, mem, df_cp = _measure(_load, warmup=2, track_gpu_memory=track_gpu_memory)
    results["load"] = {"time_s": elapsed, "memory_mb": mem}

    if "groupby" in operations and "vehicle_id" in df_cp.columns:
        def _cp_groupby():
            return df_cp.groupby("vehicle_id")["brake_pressure_pct"].agg(["max", "mean"])
        elapsed, mem, _ = _measure(_cp_groupby, warmup=2, track_gpu_memory=track_gpu_memory)
        results["groupby"] = {"time_s": elapsed, "memory_mb": mem}
    if "filter" in operations and "brake_pressure_pct" in df_cp.columns:
        def _cp_filter():
            return df_cp[df_cp["brake_pressure_pct"] > 90]
        elapsed, mem, _ = _measure(_cp_filter, warmup=2, track_gpu_memory=track_gpu_memory)
        results["filter"] = {"time_s": elapsed, "memory_mb": mem}
    if "sort" in operations:
        def _cp_sort():
            return df_cp.sort_values("timestamp_ns")
        elapsed, mem, _ = _measure(_cp_sort, warmup=2, track_gpu_memory=track_gpu_memory)
        results["sort"] = {"time_s": elapsed, "memory_mb": mem}

    del df_cp
    gc.collect()
    if result_queue is not None:
        result_queue.put(results)
    return results


def run_benchmark(
    path: Union[str, Path],
    operations: Optional[list[str]] = None,
    spill: bool = True,
    track_gpu_memory: bool = False,
    cudf_pandas_subprocess: bool = True,
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Run pandas vs cuDF vs cudf.pandas benchmark on telemetry file.

    Args:
        path: Path to Parquet or CSV.
        operations: List of ops to benchmark: "load", "groupby", "filter", "sort".
                   Default all.
        spill: Enable cuDF UVM spill (for cuDF and cudf.pandas).
        track_gpu_memory: If True, report GPU memory for cuDF/cudf.pandas instead of host tracemalloc.
        cudf_pandas_subprocess: If True (default), run cudf.pandas in a subprocess to avoid mutating pandas.

    Returns:
        Dict mapping backend ("pandas", "cudf", "cudf_pandas") to dict of operation -> {time_s, memory_mb}.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    operations = operations or ["load", "groupby", "filter", "sort"]
    results: dict[str, dict[str, dict[str, float]]] = {"pandas": {}, "cudf": {}, "cudf_pandas": {}}

    # 1. pandas (CPU baseline) - run first before cudf.pandas patches pandas
    def _pandas_load():
        return load_pandas(path)

    elapsed, mem, df_pd = _measure(_pandas_load, track_gpu_memory=False)
    results["pandas"]["load"] = {"time_s": elapsed, "memory_mb": mem}

    if "groupby" in operations:
        def _pd_groupby():
            return df_pd.groupby("vehicle_id")["brake_pressure_pct"].agg(["max", "mean"])

        elapsed, mem, _ = _measure(_pd_groupby, track_gpu_memory=False)
        results["pandas"]["groupby"] = {"time_s": elapsed, "memory_mb": mem}

    if "filter" in operations and "brake_pressure_pct" in df_pd.columns:
        def _pd_filter():
            return df_pd[df_pd["brake_pressure_pct"] > 90]

        elapsed, mem, _ = _measure(_pd_filter, track_gpu_memory=False)
        results["pandas"]["filter"] = {"time_s": elapsed, "memory_mb": mem}

    if "sort" in operations:
        def _pd_sort():
            return df_pd.sort_values("timestamp_ns")

        elapsed, mem, _ = _measure(_pd_sort, track_gpu_memory=False)
        results["pandas"]["sort"] = {"time_s": elapsed, "memory_mb": mem}

    del df_pd
    gc.collect()

    # 2. cuDF (explicit GPU API)
    if CUDF_AVAILABLE:
        def _cudf_load():
            return load_cudf(path, spill=spill)

        elapsed, mem, df_cu = _measure(_cudf_load, track_gpu_memory=track_gpu_memory)
        results["cudf"]["load"] = {"time_s": elapsed, "memory_mb": mem}

        if "groupby" in operations and "vehicle_id" in df_cu.columns:
            def _cu_groupby():
                return df_cu.groupby("vehicle_id")["brake_pressure_pct"].agg(["max", "mean"])

            elapsed, mem, _ = _measure(_cu_groupby, track_gpu_memory=track_gpu_memory)
            results["cudf"]["groupby"] = {"time_s": elapsed, "memory_mb": mem}

        if "filter" in operations and "brake_pressure_pct" in df_cu.columns:
            def _cu_filter():
                return df_cu[df_cu["brake_pressure_pct"] > 90]

            elapsed, mem, _ = _measure(_cu_filter, track_gpu_memory=track_gpu_memory)
            results["cudf"]["filter"] = {"time_s": elapsed, "memory_mb": mem}

        if "sort" in operations:
            def _cu_sort():
                return df_cu.sort_values("timestamp_ns")

            elapsed, mem, _ = _measure(_cu_sort, track_gpu_memory=track_gpu_memory)
            results["cudf"]["sort"] = {"time_s": elapsed, "memory_mb": mem}

        del df_cu
        gc.collect()

    # 3. cudf.pandas (pandas API on GPU) - run in subprocess to avoid mutating global pandas
    if CUDF_PANDAS_AVAILABLE and CUDF_AVAILABLE:
        if cudf_pandas_subprocess:
            try:
                ctx = multiprocessing.get_context("spawn")
                q = ctx.Queue()
                proc = ctx.Process(
                    target=_run_cudf_pandas_in_subprocess,
                    args=(str(path), operations, spill, track_gpu_memory, q),
                )
                proc.start()
                proc.join(timeout=300)
                if proc.exitcode == 0 and not q.empty():
                    results["cudf_pandas"] = q.get_nowait()
                else:
                    results["cudf_pandas"] = {}
            except Exception:
                results["cudf_pandas"] = {}
        else:
            _cudf_pandas_mod.install()
            if spill:
                cudf.set_option("spill", spill)
            pd_gpu = importlib.import_module("pandas")

            def _cudf_pandas_load():
                return _load_cudf_pandas(path, spill=spill)

            elapsed, mem, df_cp = _measure(_cudf_pandas_load, track_gpu_memory=track_gpu_memory)
            results["cudf_pandas"]["load"] = {"time_s": elapsed, "memory_mb": mem}

            if "groupby" in operations and "vehicle_id" in df_cp.columns:
                def _cp_groupby():
                    return df_cp.groupby("vehicle_id")["brake_pressure_pct"].agg(["max", "mean"])
                elapsed, mem, _ = _measure(_cp_groupby, track_gpu_memory=track_gpu_memory)
                results["cudf_pandas"]["groupby"] = {"time_s": elapsed, "memory_mb": mem}
            if "filter" in operations and "brake_pressure_pct" in df_cp.columns:
                def _cp_filter():
                    return df_cp[df_cp["brake_pressure_pct"] > 90]
                elapsed, mem, _ = _measure(_cp_filter, track_gpu_memory=track_gpu_memory)
                results["cudf_pandas"]["filter"] = {"time_s": elapsed, "memory_mb": mem}
            if "sort" in operations:
                def _cp_sort():
                    return df_cp.sort_values("timestamp_ns")
                elapsed, mem, _ = _measure(_cp_sort, track_gpu_memory=track_gpu_memory)
                results["cudf_pandas"]["sort"] = {"time_s": elapsed, "memory_mb": mem}

            del df_cp
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
