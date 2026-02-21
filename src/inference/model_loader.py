"""
Model loaders by format (HF, GGUF, TensorRT).

Wire format_selector.select_format() to actual loading: HuggingFace is implemented;
GGUF and TensorRT raise NotImplementedError with guidance (e.g. llama-cpp, Optimum-NVIDIA).
"""

from typing import Any, Callable, Optional, Tuple

from .format_selector import Format, select_format, UseCase, Hardware


def load_hf(
    model_id: str,
    device: str = "cuda",
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """
    Load model and tokenizer from Hugging Face (safetensors / transformers).

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device if device == "cuda" else None,
        **kwargs,
    )
    if device == "cuda":
        model = model.cuda()
    return model, tokenizer


def load_gguf(
    model_path: str,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """
    Load GGUF model (e.g. via llama-cpp-python). Not implemented in-repo.

    To implement: use llama-cpp-python Llama(model_path=...) and wrap for
    generate() / tokenizer interface.
    """
    raise NotImplementedError(
        "GGUF loading not implemented. Use llama-cpp-python or similar; "
        "select_format(use_case='local') returns 'gguf' for this path."
    )


def load_tensorrt(
    engine_path: str,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """
    Load TensorRT-LLM engine. Not implemented in-repo.

    To implement: use TensorRT-LLM runtime or Optimum-NVIDIA to load
    a compiled .engine; wrap for generate() / tokenizer interface.
    """
    raise NotImplementedError(
        "TensorRT loading not implemented. Use TensorRT-LLM or Optimum-NVIDIA; "
        "select_format(use_case='production', hardware='gpu') returns 'tensorrt'."
    )


def load_onnx(model_path: str, **kwargs: Any) -> Tuple[Any, Any]:
    """Load ONNX model. Not implemented in-repo."""
    raise NotImplementedError(
        "ONNX loading not implemented. Use ONNX Runtime or Optimum; "
        "select_format(use_case='portable') returns 'onnx'."
    )


def get_loader_for_format(fmt: Format) -> Callable[..., Tuple[Any, Any]]:
    """Return the loader function for the given format."""
    loaders: dict[Format, Callable[..., Tuple[Any, Any]]] = {
        "safetensors": load_hf,
        "gguf": load_gguf,
        "tensorrt": load_tensorrt,
        "onnx": load_onnx,
    }
    return loaders.get(fmt, load_hf)


def load_model_for_use_case(
    use_case: UseCase,
    hardware: Hardware = "gpu",
    model_id: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """
    Select format from use_case/hardware and load model and tokenizer.
    """
    fmt, _ = select_format(use_case=use_case, hardware=hardware)
    loader = get_loader_for_format(fmt)
    default_id = model_id or "meta/llama3-8b-instruct"
    if fmt == "safetensors" or loader == load_hf:
        return load_hf(default_id, **kwargs)
    return loader(default_id, **kwargs)
