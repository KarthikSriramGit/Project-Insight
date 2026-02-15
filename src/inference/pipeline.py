"""
Inference pipeline: preprocessing, batching, forward pass, decoding, postprocessing.

Tokenization with padding, attention mask, greedy decoding (logits[:, -1, :].argmax), batch_decode.
"""

from typing import Any, Optional

import torch


class InferencePipeline:
    """
    End-to-end inference pipeline for telemetry query generation.

    Pipeline: tokenize -> pad/batch -> forward pass -> decode -> detokenize.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cuda",
        max_new_tokens: int = 128,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens

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
        Greedy decoding loop.
        next_id = out.logits[:, -1, :].argmax(dim=-1)
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
        Full pipeline: tokenize, pad, decode, detokenize.

        Args:
            prompts: List of text prompts.
            max_new_tokens: Override default.

        Returns:
            List of generated text strings.
        """
        max_tok = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        enc = self._tokenize(prompts)

        generated = self._greedy_decode(
            enc["input_ids"],
            enc["attention_mask"],
            max_tok,
        )
        return self._postprocess(generated)
