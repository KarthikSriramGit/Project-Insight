"""
Model format selector based on lifecycle table.

Format choice determines where a model can run: Safetensors for sharing,
GGUF for local/quantized, TensorRT for production on NVIDIA GPUs, ONNX for portability.
"""

from typing import Literal

Format = Literal["safetensors", "gguf", "tensorrt", "onnx"]
Hardware = Literal["cpu", "gpu", "edge", "mixed"]
UseCase = Literal["research", "sharing", "local", "production", "portable"]

FORMAT_RATIONALE: dict[Format, str] = {
    "safetensors": (
        "Fast, secure weight serialization. Memory-mapped loading, no arbitrary code "
        "execution. Default for Hugging Face and open-source sharing."
    ),
    "gguf": (
        "Compact, quantized format for local inference. Powers llama.cpp and "
        "run-on-laptop workflows. Ideal for CPU or limited GPU."
    ),
    "tensorrt": (
        "Compiled engine for NVIDIA GPUs. Pre-optimized kernels, lowest latency and "
        "highest throughput in production. Requires fixed batch/seq params at compile."
    ),
    "onnx": (
        "Graph-level interchange format. Framework-agnostic, runs on ONNX Runtime, "
        "OpenVINO, TensorRT. Best for heterogeneous environments and long-term archive."
    ),
}

LIFECYCLE_TABLE: dict[UseCase, Format] = {
    "research": "safetensors",
    "sharing": "safetensors",
    "local": "gguf",
    "production": "tensorrt",
    "portable": "onnx",
}


def select_format(
    use_case: UseCase,
    hardware: Hardware = "gpu",
) -> tuple[Format, str]:
    """
    Select model format by target use case and hardware.

    Args:
        use_case: research, sharing, local, production, portable.
        hardware: cpu, gpu, edge, mixed.

    Returns:
        (format, rationale) tuple.
    """
    fmt = LIFECYCLE_TABLE.get(use_case, "safetensors")

    if use_case == "production" and hardware == "cpu":
        fmt = "onnx"
    elif use_case == "production" and hardware == "edge":
        fmt = "gguf"
    elif use_case == "local" and hardware == "gpu":
        fmt = "gguf"

    rationale = FORMAT_RATIONALE[fmt]
    return fmt, rationale
