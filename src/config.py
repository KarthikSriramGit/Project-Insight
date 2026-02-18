"""
Load pipeline configuration from YAML.

Centralizes ingest, inference, deploy, and query settings so they can be
tuned without code changes.
"""

from pathlib import Path
from typing import Any, Optional, Union

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Default config path relative to project root
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "pipeline_config.yaml"

_cached_config: Optional[dict[str, Any]] = None


def load_config(path: Optional[Union[str, Path]] = None) -> dict[str, Any]:
    """
    Load pipeline config from YAML. Returns cached dict if already loaded.

    Args:
        path: Path to pipeline_config.yaml. If None, uses config/pipeline_config.yaml.

    Returns:
        Nested dict with keys: dataset, ingest, inference, deploy, query.
    """
    global _cached_config
    if _cached_config is not None:
        return _cached_config

    if not YAML_AVAILABLE:
        _cached_config = _default_config_dict()
        return _cached_config

    p = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    if not p.exists():
        _cached_config = _default_config_dict()
        return _cached_config

    with open(p, encoding="utf-8") as f:
        _cached_config = yaml.safe_load(f) or _default_config_dict()

    return _cached_config


def _default_config_dict() -> dict[str, Any]:
    """Fallback when YAML is missing or PyYAML not installed."""
    return {
        "dataset": {
            "synthetic_path": "data/synthetic",
            "parquet_filename": "fleet_telemetry.parquet",
            "default_row_count": 5_000_000,
            "large_row_count": 50_000_000,
            "vehicle_count": 10,
            "duration_hours": 24,
        },
        "ingest": {
            "cudf": {"spill_enabled": True, "spill_on_demand": False},
            "fallback_cpu": True,
        },
        "inference": {
            "model_id": "meta/llama3-8b-instruct",
            "batch_sizes": [1, 8, 32, 64, 128],
            "max_new_tokens": 128,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "temperature": 0.0,
            "warmup_runs": 2,
            "runs_per_batch": 10,
            "collect_metrics": True,
        },
        "deploy": {
            "nim": {
                "base_url": "http://localhost:8000",
                "chat_completions_path": "/v1/chat/completions",
                "model": "meta/llama3-8b-instruct",
                "max_tokens": 256,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "stream": False,
            },
            "gke": {
                "cluster_name": "nim-demo",
                "namespace": "nim",
                "gpu_type": "nvidia-l4",
                "gpu_count": 1,
            },
        },
        "query": {
            "max_context_rows": 1000,
            "summarize_results": True,
        },
    }


def get_ingest_config() -> dict[str, Any]:
    """Ingest section (cudf spill, fallback_cpu)."""
    return load_config().get("ingest", _default_config_dict()["ingest"])


def get_inference_config() -> dict[str, Any]:
    """Inference section (model_id, batch_sizes, warmup_runs, etc.)."""
    return load_config().get("inference", _default_config_dict()["inference"])


def get_deploy_config() -> dict[str, Any]:
    """Deploy section (nim, gke)."""
    return load_config().get("deploy", _default_config_dict()["deploy"])


def get_query_config() -> dict[str, Any]:
    """Query section (max_context_rows, summarize_results)."""
    return load_config().get("query", _default_config_dict()["query"])


def get_dataset_config() -> dict[str, Any]:
    """Dataset section (paths, row counts)."""
    return load_config().get("dataset", _default_config_dict()["dataset"])
