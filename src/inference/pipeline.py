"""
Inference pipeline: preprocessing, batching, forward pass, decoding, postprocessing.

Tokenization with padding, attention mask, greedy decoding (logits[:, -1, :].argmax), batch_decode.
Supports config-driven batch sizes, warmup, and optional TTFT capture.
"""

import time
from typing import Any, Optional

import torch


def _get_inference_config() -> dict:
    """Load inference section from pipeline config if available."""
    try:
        from ..config import get_inference_config
        return get_inference_config()
    except Exception:
        return {}


class InferencePipeline:
    """
    End-to-end inference pipeline for telemetry query generation.

    Pipeline: tokenize -> pad/batch -> forward pass -> decode -> detokenize.
    When given a list of prompts, batches them per model.generate() for better GPU utilization.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cuda",
        max_new_tokens: int = 128,
        batch_size: Optional[int] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        cfg = _get_inference_config()
        self.max_new_tokens = cfg.get("max_new_tokens", max_new_tokens)
        batch_sizes = cfg.get("batch_sizes", [1]) if cfg else [1]
        self._batch_size = batch_size if batch_size is not None else batch_sizes[0]

    def _tokenize(
        self,
        prompts: list[str],
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        """Preprocessing: tokenize with padding and attention mask."""
        enc = self.tokenizer(
            prompts,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def _forward_pass(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass only (no generation). Returns logits."""
        with torch.inference_mode():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        return out.logits

    def _greedy_decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """
        Greedy decoding loop (no KV cache). Kept for debug/education only.
        For production use model.generate() via generate().
        """
        generated = input_ids.clone()
        mask = attention_mask.clone()

        for _ in range(max_new_tokens):
            with torch.inference_mode():
                out = self.model(input_ids=generated, attention_mask=mask)
                next_id = out.logits[:, -1, :].argmax(dim=-1)

            generated = torch.cat([generated, next_id.unsqueeze(-1)], dim=-1)
            mask = torch.cat([mask, torch.ones_like(next_id).unsqueeze(-1)], dim=-1)

        return generated

    def _postprocess(self, generated_ids: torch.Tensor) -> list[str]:
        """Postprocessing: batch_decode with skip_special_tokens."""
        decoded = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        return decoded

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: Optional[int] = None,
    ) -> list[str]:
        """
        Full pipeline: tokenize, pad, generate, detokenize.
        Batches prompts in chunks of self._batch_size for better GPU utilization.

        Uses model.generate() with KV caching for memory-efficient inference.

        Args:
            prompts: List of text prompts.
            max_new_tokens: Override default.

        Returns:
            List of generated text strings.
        """
        max_tok = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        all_outputs: list[str] = []
        batch_size = max(1, self._batch_size)

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            enc = self._tokenize(batch)
            prompt_len = enc["input_ids"].shape[1]

            with torch.inference_mode():
                generated = self.model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=max_tok,
                    do_sample=False,
                )

            new_tokens = generated[:, prompt_len:]
            all_outputs.extend(self._postprocess(new_tokens))

        return all_outputs

    def generate_with_timing(
        self,
        prompts: list[str],
        max_new_tokens: Optional[int] = None,
    ) -> tuple[list[str], list[float], Optional[list[float]]]:
        """
        Generate and return outputs plus total latencies per prompt.
        First-token latencies can be filled by streaming backends (e.g. NIM); here we return None.

        Returns:
            (outputs, total_latencies_sec, first_token_latencies_sec or None)
        """
        max_tok = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        all_outputs: list[str] = []
        total_latencies: list[float] = []

        for p in prompts:
            enc = self._tokenize([p])
            prompt_len = enc["input_ids"].shape[1]

            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.inference_mode():
                generated = self.model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=max_tok,
                    do_sample=False,
                )

            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            total_latencies.append(time.perf_counter() - t0)

            new_tokens = generated[:, prompt_len:]
            out = self._postprocess(new_tokens)
            all_outputs.extend(out)

        return all_outputs, total_latencies, None
