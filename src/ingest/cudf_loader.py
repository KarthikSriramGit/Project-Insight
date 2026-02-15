"""
GPU-accelerated telemetry loading with cuDF and Unified Virtual Memory.

cudf.set_option('spill', True)
enables UVM spill so datasets larger than GPU VRAM can be processed by spilling to CPU RAM.
"""

from pathlib import Path
from typing import Optional, Union

try:
    import cudf

    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False

import pandas as pd


def load_telemetry(
    path: Union[str, Path],
    spill: bool = True,
    use_cudf: Optional[bool] = None,
) -> Union["cudf.DataFrame", pd.DataFrame]:
    """
    Load telemetry from Parquet or CSV with GPU acceleration when available.

    When cuDF is available and use_cudf is True, enables UVM spill (Course 3 pattern)
    so datasets larger than GPU memory can be loaded. Data spills to CPU RAM as needed.

    Args:
        path: Path to Parquet or CSV file.
        spill: Enable Unified Virtual Memory spill (cuDF only). Default True.
        use_cudf: Force cuDF if True, pandas if False. If None, use cuDF when available.

    Returns:
        DataFrame (cuDF or pandas).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Telemetry file not found: {path}")

    use_gpu = use_cudf if use_cudf is not None else CUDF_AVAILABLE

    if use_gpu and CUDF_AVAILABLE:
        cudf.set_option("spill", spill)
        if path.suffix.lower() in (".parquet", ".pq"):
            df = cudf.read_parquet(path)
        elif path.suffix.lower() == ".csv":
            df = cudf.read_csv(path)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
        return df

    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported format: {path.suffix}")


def filter_by_time_range(
    df: Union["cudf.DataFrame", pd.DataFrame],
    start_ns: Optional[int] = None,
    end_ns: Optional[int] = None,
) -> Union["cudf.DataFrame", pd.DataFrame]:
    """Filter telemetry by timestamp range (nanoseconds)."""
    if "timestamp_ns" not in df.columns:
        return df
    if start_ns is not None:
        df = df[df["timestamp_ns"] >= start_ns]
    if end_ns is not None:
        df = df[df["timestamp_ns"] <= end_ns]
    return df


def filter_by_vehicle(
    df: Union["cudf.DataFrame", pd.DataFrame],
    vehicle_ids: list[str],
) -> Union["cudf.DataFrame", pd.DataFrame]:
    """Filter telemetry by vehicle ID(s)."""
    if not vehicle_ids or "vehicle_id" not in df.columns:
        return df
    return df[df["vehicle_id"].isin(vehicle_ids)]


def aggregate_can_stats(
    df: Union["cudf.DataFrame", pd.DataFrame],
    group_cols: Optional[list[str]] = None,
) -> Union["cudf.DataFrame", pd.DataFrame]:
    """
    Aggregate CAN bus statistics per vehicle (or other grouping).

    Useful for anomaly detection: max brake pressure, avg speed, etc.
    """
    if "sensor_type" in df.columns:
        can_df = df[df["sensor_type"] == "can"].copy()
    else:
        can_df = df.copy()

    group_cols = group_cols or ["vehicle_id"]
    group_cols = [c for c in group_cols if c in can_df.columns]
    if not group_cols:
        return can_df

    agg_dict = {}
    if "brake_pressure_pct" in can_df.columns:
        agg_dict["brake_pressure_pct"] = ["max", "mean"]
    if "vehicle_speed_kmh" in can_df.columns:
        agg_dict["vehicle_speed_kmh"] = ["max", "mean"]
    if "throttle_position_pct" in can_df.columns:
        agg_dict["throttle_position_pct"] = ["max", "mean"]

    if not agg_dict:
        return can_df

    return can_df.groupby(group_cols).agg(agg_dict).reset_index()


def get_anomaly_windows(
    df: Union["cudf.DataFrame", pd.DataFrame],
    column: str,
    threshold: float,
    vehicle_id: Optional[str] = None,
) -> Union["cudf.DataFrame", pd.DataFrame]:
    """
    Find time windows where a numeric column exceeds threshold (anomaly detection).

    Returns rows where column > threshold, optionally filtered by vehicle.
    """
    if column not in df.columns:
        return df.head(0)

    subset = df[df[column] > threshold]
    if vehicle_id and "vehicle_id" in subset.columns:
        subset = subset[subset["vehicle_id"] == vehicle_id]
    return subset
