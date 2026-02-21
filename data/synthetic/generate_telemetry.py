"""
Synthetic telemetry generator for ROS2/NVIDIA DRIVE fleet data.

Generates millions of rows mimicking real sensor streams: IMU, LiDAR, CAN bus, GPS, and camera.
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _generate_imu(
    n_rows: int,
    vehicle_ids: list[str],
    timestamps_ns: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate IMU data (sensor_msgs/Imu)."""
    n = n_rows
    data = {
        "timestamp_ns": rng.choice(timestamps_ns, n),
        "vehicle_id": rng.choice(vehicle_ids, n),
        "sensor_type": ["imu"] * n,
        "accel_x": rng.normal(0, 0.5, n),
        "accel_y": rng.normal(0, 0.5, n),
        "accel_z": rng.normal(9.81, 0.2, n),
        "gyro_x": rng.normal(0, 0.01, n),
        "gyro_y": rng.normal(0, 0.01, n),
        "gyro_z": rng.normal(0, 0.01, n),
        "orientation_w": rng.uniform(0.99, 1.0, n),
        "orientation_x": rng.normal(0, 0.01, n),
        "orientation_y": rng.normal(0, 0.01, n),
        "orientation_z": rng.normal(0, 0.01, n),
    }
    return pd.DataFrame(data)


def _generate_lidar(
    n_rows: int,
    vehicle_ids: list[str],
    timestamps_ns: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate LiDAR point cloud stats (sensor_msgs/PointCloud2)."""
    n = n_rows
    data = {
        "timestamp_ns": rng.choice(timestamps_ns, n),
        "vehicle_id": rng.choice(vehicle_ids, n),
        "sensor_type": ["lidar"] * n,
        "point_count": rng.integers(10000, 500000, n),
        "min_range": rng.uniform(0.5, 2.0, n),
        "max_range": rng.uniform(50, 150, n),
        "mean_intensity": rng.uniform(0, 255, n),
        "frame_id": ["lidar_front"] * n,
    }
    return pd.DataFrame(data)


def _generate_can(
    n_rows: int,
    vehicle_ids: list[str],
    timestamps_ns: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate CAN bus / Ethernet virtual sensor data (NVIDIA DRIVE)."""
    n = n_rows
    data = {
        "timestamp_ns": rng.choice(timestamps_ns, n),
        "vehicle_id": rng.choice(vehicle_ids, n),
        "sensor_type": ["can"] * n,
        "vehicle_speed_kmh": rng.uniform(0, 120, n),
        "brake_pressure_pct": rng.uniform(0, 100, n),
        "steering_angle_deg": rng.uniform(-540, 540, n),
        "throttle_position_pct": rng.uniform(0, 100, n),
        "engine_rpm": rng.uniform(0, 6000, n),
        "gear_position": rng.integers(0, 8, n),
    }
    return pd.DataFrame(data)


def _generate_gps(
    n_rows: int,
    vehicle_ids: list[str],
    timestamps_ns: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate GPS/Odometry data (nav_msgs/NavSatFix)."""
    n = n_rows
    data = {
        "timestamp_ns": rng.choice(timestamps_ns, n),
        "vehicle_id": rng.choice(vehicle_ids, n),
        "sensor_type": ["gps"] * n,
        "latitude": rng.uniform(37.0, 38.0, n),
        "longitude": rng.uniform(-122.5, -121.5, n),
        "altitude_m": rng.uniform(0, 500, n),
        "velocity_north": rng.uniform(-20, 20, n),
        "velocity_east": rng.uniform(-20, 20, n),
    }
    return pd.DataFrame(data)


def _generate_camera(
    n_rows: int,
    vehicle_ids: list[str],
    timestamps_ns: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate camera metadata (sensor_msgs/Image)."""
    n = n_rows
    data = {
        "timestamp_ns": rng.choice(timestamps_ns, n),
        "vehicle_id": rng.choice(vehicle_ids, n),
        "sensor_type": ["camera"] * n,
        "frame_id": rng.choice(["camera_front", "camera_left", "camera_right"], n),
        "exposure_ms": rng.uniform(1, 30, n),
        "object_count": rng.integers(0, 50, n),
        "resolution_w": [1920] * n,
        "resolution_h": [1080] * n,
    }
    return pd.DataFrame(data)


def generate_telemetry(
    n_rows: int = 5_000_000,
    vehicle_count: int = 10,
    duration_hours: float = 24.0,
    seed: int = 42,
    chunk_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate fleet telemetry spanning multiple sensor types.

    For large n_rows (e.g. 50M), set chunk_size (e.g. 5_000_000) to generate
    in chunks and reduce peak host memory.

    Args:
        n_rows: Total number of rows across all sensors.
        vehicle_count: Number of vehicle IDs.
        duration_hours: Time span in hours for timestamps.
        seed: Random seed for reproducibility.
        chunk_size: If set, generate in chunks of this many rows and concat; reduces memory.

    Returns:
        DataFrame with unified telemetry schema. Rows are split across
        imu, lidar, can, gps, camera sensors.
    """
    rng = np.random.default_rng(seed)
    vehicle_ids = [f"V{i:03d}" for i in range(vehicle_count)]

    duration_ns = int(duration_hours * 3600 * 1e9)
    ts_size = min(n_rows // 5, 1_000_000)
    timestamps_ns = np.linspace(0, duration_ns, ts_size).astype(np.int64)

    if chunk_size is not None and chunk_size > 0 and n_rows > chunk_size:
        # Chunked generation to reduce peak memory for very large datasets
        chunks = []
        remaining = n_rows
        chunk_seed = seed
        while remaining > 0:
            take = min(chunk_size, remaining)
            chunk_rng = np.random.default_rng(chunk_seed)
            chunk_seed += 1
            n_per_sensor = take // 5
            dfs = [
                _generate_imu(n_per_sensor, vehicle_ids, timestamps_ns, chunk_rng),
                _generate_lidar(n_per_sensor, vehicle_ids, timestamps_ns, chunk_rng),
                _generate_can(n_per_sensor, vehicle_ids, timestamps_ns, chunk_rng),
                _generate_gps(n_per_sensor, vehicle_ids, timestamps_ns, chunk_rng),
                _generate_camera(take - 4 * n_per_sensor, vehicle_ids, timestamps_ns, chunk_rng),
            ]
            chunk = pd.concat(dfs, ignore_index=True)
            chunks.append(chunk)
            remaining -= take
        combined = pd.concat(chunks, ignore_index=True)
    else:
        n_per_sensor = n_rows // 5
        dfs = [
            _generate_imu(n_per_sensor, vehicle_ids, timestamps_ns, rng),
            _generate_lidar(n_per_sensor, vehicle_ids, timestamps_ns, rng),
            _generate_can(n_per_sensor, vehicle_ids, timestamps_ns, rng),
            _generate_gps(n_per_sensor, vehicle_ids, timestamps_ns, rng),
            _generate_camera(n_rows - 4 * n_per_sensor, vehicle_ids, timestamps_ns, rng),
        ]
        combined = pd.concat(dfs, ignore_index=True)

    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    combined = combined.sort_values("timestamp_ns").reset_index(drop=True)

    return combined


def _ensure_full_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing columns with NaN to match unified schema."""
    all_cols = [
        "timestamp_ns", "vehicle_id", "sensor_type",
        "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z",
        "orientation_w", "orientation_x", "orientation_y", "orientation_z",
        "point_count", "min_range", "max_range", "mean_intensity", "frame_id",
        "vehicle_speed_kmh", "brake_pressure_pct", "steering_angle_deg",
        "throttle_position_pct", "engine_rpm", "gear_position",
        "latitude", "longitude", "altitude_m", "velocity_north", "velocity_east",
        "exposure_ms", "object_count", "resolution_w", "resolution_h",
    ]
    for c in all_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[all_cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic fleet telemetry")
    parser.add_argument(
        "--rows",
        type=int,
        default=5_000_000,
        help="Total rows (default 5M)",
    )
    parser.add_argument(
        "--vehicles",
        type=int,
        default=10,
        help="Number of vehicle IDs",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=24.0,
        help="Duration span in hours",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Generate in chunks of this many rows to reduce memory (e.g. 5000000 for 50M)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic",
        help="Output directory for Parquet/CSV",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv", "both"],
        default="parquet",
        help="Output format",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = generate_telemetry(
        n_rows=args.rows,
        vehicle_count=args.vehicles,
        duration_hours=args.hours,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )
    df = _ensure_full_schema(df)

    parquet_path = out_dir / "fleet_telemetry.parquet"
    csv_path = out_dir / "fleet_telemetry.csv"

    if args.format in ("parquet", "both"):
        df.to_parquet(parquet_path, index=False)
        print(f"Wrote {parquet_path} ({len(df)} rows)")

    if args.format in ("csv", "both"):
        df.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
