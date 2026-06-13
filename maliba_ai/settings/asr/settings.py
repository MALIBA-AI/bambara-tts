from dataclasses import dataclass
from typing import Optional


class Settings:
    # Default model repositories for the supported ASR architectures.
    whisper_model_repo: str = "MALIBA-AI/bambara-asr-v2"
    gemma_model_repo: str = "MALIBA-AI/gemma-asr-v1"

    # Default instruction used by instruction-tuned ASR models (e.g. gemma-3n).
    transcription_prompt: str = "Transcribe this audio."


@dataclass
class TranscriptionOutput:
    text: Optional[str] = None
    error_message: Optional[str] = None
