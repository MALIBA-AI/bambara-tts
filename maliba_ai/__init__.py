__version__ = "v2.0.0-alpha"
__author__ = "sudoping01"

from maliba_ai.core import (LLM, Embeddings, MachineTranslation,
                            SpeechRecognition, TTS)

__all__ = [
    "SpeechRecognition",
    "MachineTranslation",
    "Embeddings",
    "LLM",
    "TTS",
]
