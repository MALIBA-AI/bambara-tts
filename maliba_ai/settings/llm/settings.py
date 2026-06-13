from dataclasses import dataclass
from typing import Optional


class Settings:
    model_repo: str = "MALIBA-AI/bambara-llm-v1"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95


@dataclass
class GenerationOutput:
    text: Optional[str] = None
    error_message: Optional[str] = None
