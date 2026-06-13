"""Thin async client for the llama.cpp server `/completion` endpoint.

The LLM half of Spark-TTS (a Qwen2.5-0.5B finetune) is served by llama.cpp from
the quantized GGUF. Given the TTS prompt it autoregressively emits a string of
special tokens like ``<|bicodec_global_42|>`` and ``<|bicodec_semantic_1337|>``.
This module builds that prompt, calls llama.cpp, and parses the integers back
out so the BiCodec vocoder can turn them into audio.
"""

import json
import re
from typing import AsyncIterator, List, Optional, Tuple

import httpx

_SEMANTIC_RE = re.compile(r"<\|bicodec_semantic_(\d+)\|>")
_GLOBAL_RE = re.compile(r"<\|bicodec_global_(\d+)\|>")


def build_prompt(text: str) -> str:
    """Build the Spark-TTS prompt the model completes into audio tokens."""
    return (
        "<|task_tts|>"
        "<|start_content|>"
        f"{text}"
        "<|end_content|>"
        "<|start_global_token|>"
    )


def parse_audio_tokens(generated_text: str) -> Tuple[List[int], List[int]]:
    """Extract (global_ids, semantic_ids) from the model's generated text."""
    semantic_ids = [int(t) for t in _SEMANTIC_RE.findall(generated_text)]
    global_ids = [int(t) for t in _GLOBAL_RE.findall(generated_text)]
    return global_ids, semantic_ids


class LlamaClient:
    def __init__(self, base_url: str, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def health(self) -> bool:
        try:
            resp = await self._client.get(f"{self.base_url}/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: Optional[int] = None,
    ) -> str:
        """Call llama.cpp /completion and return the raw generated text."""
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            # Keep the (identical) TTS prefix cached across requests.
            "cache_prompt": True,
            "stream": False,
        }
        if seed is not None:
            payload["seed"] = seed

        resp = await self._client.post(f"{self.base_url}/completion", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("content", "")

    async def stream_generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Stream generated text pieces from llama.cpp as they are produced."""
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "cache_prompt": True,
            "stream": True,
        }
        if seed is not None:
            payload["seed"] = seed

        async with self._client.stream(
            "POST", f"{self.base_url}/completion", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                piece = obj.get("content", "")
                if piece:
                    yield piece
                if obj.get("stop"):
                    break

    async def aclose(self) -> None:
        await self._client.aclose()
