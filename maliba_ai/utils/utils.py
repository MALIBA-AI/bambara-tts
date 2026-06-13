"""Shared utilities for the Bambara TTS package."""

from maliba_ai.config.settings import NORMALIZE_OPTIONS


def normalize_text(text: str) -> str:
    """Normalize Bambara text exactly the way the model was trained on.

    Centralised here so the local PyTorch inference path and the llama.cpp
    inference server feed the model identical input.

    Args:
        text: Raw Bambara input text.

    Returns:
        The normalized text.
    """
    if not isinstance(text, str):
        raise TypeError("text should be a string")
    if not text.strip():
        raise ValueError("text can not be empty")

    # Imported lazily so importing this module never hard-fails when the
    # optional normalizer dependency is missing (e.g. minimal server images).
    from bambara_normalizer import normalize

    return normalize(text, **NORMALIZE_OPTIONS)
