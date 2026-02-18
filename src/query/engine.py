"""
Telemetry query engine: cuDF retrieval + NIM natural-language summarization.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Optional, Union

from .prompts import SYSTEM_PROMPT, format_user_query
from ..ingest.telemetry_schema import get_columns_for_sensor

try:
    import cudf

    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False

import pandas as pd


def _get_query_config() -> dict:
    try:
        from ..config import get_query_config, get_deploy_config
        q = get_query_config()
        d = get_deploy_config()
        nim = d.get("nim", {})
        return {
            "max_context_rows": q.get("max_context_rows", 1000),
            "nim_base_url": nim.get("base_url", "http://localhost:8000"),
            "nim_model": nim.get("model", "meta/llama3-8b-instruct"),
            "nim_stream": nim.get("stream", False),
        }
    except Exception:
        return {}


class TelemetryQueryEngine:
    """
    Orchestrates: load telemetry with cuDF, filter/aggregate, format context,
    send to NIM for natural-language answer. Reuses a single NIM client; supports concurrent queries.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        nim_base_url: Optional[str] = None,
        nim_model: Optional[str] = None,
        max_context_rows: Optional[int] = None,
        use_cudf: Optional[bool] = None,
        nim_stream: Optional[bool] = None,
    ):
        self.data_path = Path(data_path)
        cfg = _get_query_config()
        self.nim_base_url = nim_base_url or cfg.get("nim_base_url", "http://localhost:8000")
        self.nim_model = nim_model or cfg.get("nim_model", "meta/llama3-8b-instruct")
        self.max_context_rows = max_context_rows or cfg.get("max_context_rows", 1000)
        self.nim_stream = nim_stream if nim_stream is not None else cfg.get("nim_stream", False)
        self.use_cudf = use_cudf if use_cudf is not None else CUDF_AVAILABLE

        self._df: Optional[Union[pd.DataFrame, "cudf.DataFrame"]] = None
        self._columns_hint: Optional[list[str]] = None
        self._nim_client: Optional[Any] = None

    def _get_nim_client(self) -> Any:
        """Reuse a single NIM client instance."""
        if self._nim_client is None:
            from ..deploy.nim_client import NIMClient
            self._nim_client = NIMClient(
                base_url=self.nim_base_url,
                model=self.nim_model,
                stream=self.nim_stream,
            )
        return self._nim_client

    def _load_data(self, columns: Optional[list[str]] = None) -> Union[pd.DataFrame, "cudf.DataFrame"]:
        """Load telemetry with cuDF or pandas. Optional column pruning when columns is set."""
        from ..ingest.cudf_loader import load_telemetry
        from ..config import get_ingest_config
        ingest_cfg = get_ingest_config()
        cudf_cfg = ingest_cfg.get("cudf", {})
        spill = cudf_cfg.get("spill_enabled", True)
        return load_telemetry(
            self.data_path,
            spill=spill,
            use_cudf=self.use_cudf,
            columns=columns,
        )

    def _ensure_loaded(self, columns: Optional[list[str]] = None) -> None:
        if self._df is None:
            self._df = self._load_data(columns=columns)
            self._columns_hint = columns

    def _data_to_context(
        self,
        df: Union[pd.DataFrame, "cudf.DataFrame"],
        max_rows: Optional[int] = None,
        describe_sample_cap: Optional[int] = 10000,
    ) -> str:
        """Build context: per-vehicle stats and sample on GPU when cuDF, then convert only small result to string."""
        n = max_rows or self.max_context_rows
        describe_cap = describe_sample_cap or 10000

        parts = []
        is_cudf = hasattr(df, "to_pandas")

        if is_cudf and CUDF_AVAILABLE:
            # Keep groupby and aggregates on GPU; convert only small results for text
            if "vehicle_id" in df.columns:
                dtype_str = lambda c: str(getattr(df[c], "dtype", ""))
                numeric = [c for c in df.columns if c != "timestamp_ns" and ("float" in dtype_str(c) or "int" in dtype_str(c))]
                agg_cols = [c for c in numeric if c in df.columns][:20]
                if agg_cols:
                    try:
                        per_vehicle = df.groupby("vehicle_id")[agg_cols].mean()
                        per_vehicle_str = per_vehicle.to_pandas().to_string()
                        parts.append(f"Per-vehicle averages ({len(per_vehicle)} vehicles):\n{per_vehicle_str}")
                    except Exception:
                        pass
            sample_for_describe = df.head(describe_cap) if len(df) > describe_cap else df
            try:
                desc = sample_for_describe.describe(include="all")
            except Exception:
                desc = sample_for_describe.describe()
            desc_pdf = desc.to_pandas() if hasattr(desc, "to_pandas") else desc
            stats_str = f"Summary statistics (sample of {len(sample_for_describe):,} rows):\n{desc_pdf.to_string()}"
            parts.append(stats_str)
            subset = df.head(n)
            rows_pdf = subset.to_pandas()
            rows_str = f"\nSample rows (first {min(n, len(df))}):\n{rows_pdf.to_string(max_rows=n)}"
            parts.append(rows_str)
        else:
            pdf = df.to_pandas() if hasattr(df, "to_pandas") else df
            if "vehicle_id" in pdf.columns:
                numeric = pdf.select_dtypes(include=["number"]).columns
                agg_cols = [c for c in numeric if c != "timestamp_ns" and c in pdf.columns]
                if agg_cols:
                    try:
                        per_vehicle = pdf.groupby("vehicle_id", observed=True)[agg_cols].mean()
                        parts.append(f"Per-vehicle averages ({len(per_vehicle)} vehicles):\n{per_vehicle.to_string()}")
                    except Exception:
                        pass
            sample = pdf.head(describe_cap) if len(pdf) > describe_cap else pdf
            desc = sample.describe(include="all")
            parts.append(f"Summary statistics ({len(sample):,} rows):\n{desc.to_string()}")
            subset = pdf.head(n)
            parts.append(f"\nSample rows (first {min(n, len(pdf))}):\n{subset.to_string(max_rows=n)}")

        return "\n\n".join(parts)

    def retrieve(
        self,
        vehicle_ids: Optional[list[str]] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        sensor_type: Optional[str] = None,
        brake_threshold: Optional[float] = None,
    ) -> Union[pd.DataFrame, "cudf.DataFrame"]:
        """
        Retrieve telemetry with filters. Uses cuDF when available.
        When sensor_type is set and data not yet loaded, loads only relevant columns (lazy/partial).
        """
        from ..ingest.cudf_loader import (
            filter_by_time_range,
            filter_by_vehicle,
            get_anomaly_windows,
        )

        columns_hint = get_columns_for_sensor(sensor_type) if sensor_type else None
        self._ensure_loaded(columns=columns_hint)
        df = self._df

        if vehicle_ids:
            df = filter_by_vehicle(df, vehicle_ids)
        if start_ns is not None or end_ns is not None:
            df = filter_by_time_range(df, start_ns, end_ns)
        if sensor_type and "sensor_type" in df.columns:
            df = df[df["sensor_type"] == sensor_type]
            # Keep only columns relevant to this sensor type (removes NaN-only columns)
            relevant_cols = get_columns_for_sensor(sensor_type)
            relevant_cols = [c for c in relevant_cols if c in df.columns]
            if relevant_cols:
                df = df[relevant_cols]
        if brake_threshold is not None and "brake_pressure_pct" in df.columns:
            df = df[df["brake_pressure_pct"] > brake_threshold]

        return df

    def query(
        self,
        user_query: str,
        vehicle_ids: Optional[list[str]] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        sensor_type: Optional[str] = None,
        brake_threshold: Optional[float] = None,
    ) -> str:
        """
        Execute natural-language query over telemetry.

        1. Retrieve data with filters (inferred from query or explicit).
        2. Format as context.
        3. Send to NIM for summarization.
        """
        df = self.retrieve(
            vehicle_ids=vehicle_ids,
            start_ns=start_ns,
            end_ns=end_ns,
            sensor_type=sensor_type,
            brake_threshold=brake_threshold,
        )
        context = self._data_to_context(df)
        client = self._get_nim_client()
        user_msg = format_user_query(user_query, context)
        return client.ask(user_msg, system_context=SYSTEM_PROMPT)

    def query_many(
        self,
        queries: List[dict[str, Any]],
        max_workers: int = 4,
    ) -> List[str]:
        """
        Run multiple natural-language queries concurrently via NIM.
        Each element of queries is a dict with keys: user_query, and optional
        vehicle_ids, start_ns, end_ns, sensor_type, brake_threshold.
        Returns responses in the same order as queries.
        """
        def run_one(q: dict) -> str:
            return self.query(
                user_query=q["user_query"],
                vehicle_ids=q.get("vehicle_ids"),
                start_ns=q.get("start_ns"),
                end_ns=q.get("end_ns"),
                sensor_type=q.get("sensor_type"),
                brake_threshold=q.get("brake_threshold"),
            )

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_idx = {ex.submit(run_one, q): i for i, q in enumerate(queries)}
            results: List[str] = [""] * len(queries)
            for f in as_completed(future_to_idx):
                results[future_to_idx[f]] = f.result()
            return results
