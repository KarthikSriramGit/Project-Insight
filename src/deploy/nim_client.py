"""
NVIDIA NIM chat completions API client.

POST to /v1/chat/completions with system/user messages, model meta/llama3-8b-instruct,
max_tokens, top_p, frequency_penalty.
"""

import json
from typing import Any, Optional

import requests


def chat_completion(
    base_url: str = "http://localhost:8000",
    messages: Optional[list[dict[str, str]]] = None,
    model: str = "meta/llama3-8b-instruct",
    max_tokens: int = 128,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    stream: bool = False,
) -> dict[str, Any]:
    """
    Call NIM chat completions endpoint.

    Args:
        base_url: NIM base URL (e.g. http://localhost:8000 when port-forwarded).
        messages: List of {"role": "system"|"user"|"assistant", "content": "..."}.
        model: Model ID.
        max_tokens: Max tokens to generate.
        top_p: Nucleus sampling.
        frequency_penalty: Penalty for repetition.
        stream: Enable streaming.

    Returns:
        Response JSON with choices[0].message.content.
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"

    if messages is None:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello."},
        ]

    payload = {
        "messages": messages,
        "model": model,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "n": 1,
        "stream": stream,
        "frequency_penalty": frequency_penalty,
    }

    resp = requests.post(
        url,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


class NIMClient:
    """Wrapper for NIM chat completions with telemetry context. Supports streaming for better TTFT."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "meta/llama3-8b-instruct",
        max_tokens: int = 256,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        stream: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.stream = stream

    def ask(
        self,
        user_query: str,
        system_context: Optional[str] = None,
    ) -> str:
        """
        Send a natural-language query and return the model response.
        When stream=True, consumes the stream and returns concatenated content (better TTFT).
        """
        system = system_context or (
            "You are an AI assistant analyzing fleet telemetry data from "
            "autonomous vehicles with ROS2 and NVIDIA DRIVE sensors."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_query},
        ]
        if self.stream:
            return self._ask_stream(messages)
        resp = chat_completion(
            base_url=self.base_url,
            messages=messages,
            model=self.model,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            stream=False,
        )
        choice = resp.get("choices", [{}])[0]
        msg = choice.get("message", {})
        return msg.get("content", "")

    def _ask_stream(self, messages: list[dict[str, str]]) -> str:
        """Consume streaming chat completion and return full content."""
        url = self.base_url + "/v1/chat/completions"
        payload = {
            "messages": messages,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "n": 1,
            "stream": True,
            "frequency_penalty": self.frequency_penalty,
        }
        resp = requests.post(
            url,
            headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
            json=payload,
            timeout=120,
            stream=True,
        )
        resp.raise_for_status()
        chunks: list[str] = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
                for choice in obj.get("choices", []):
                    delta = choice.get("delta", {})
                    content = delta.get("content")
                    if content:
                        chunks.append(content)
            except Exception:
                continue
        return "".join(chunks)
