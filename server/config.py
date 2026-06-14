"""Runtime configuration for the TTS gateway, all overridable via env vars."""

import os


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


class Config:
    # Where the llama.cpp server (serving the GGUF) is reachable.
    llama_url: str = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080")
    llama_timeout: float = float(os.getenv("LLAMA_TIMEOUT", "300"))

    # Spark-TTS / BiCodec assets.
    spark_model_dir: str = os.getenv("SPARK_MODEL_DIR", "Spark-TTS-0.5B")
    # Force a device for the vocoder ("cpu" / "cuda"); empty = auto-detect.
    device: str = os.getenv("TTS_DEVICE", "")

    # Gateway bind address.
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    # Default sampling parameters (match the package defaults).
    default_temperature: float = float(os.getenv("TTS_TEMPERATURE", "0.8"))
    default_top_k: int = int(os.getenv("TTS_TOP_K", "50"))
    default_top_p: float = float(os.getenv("TTS_TOP_P", "1.0"))
    default_max_tokens: int = int(os.getenv("TTS_MAX_TOKENS", "2048"))

    sample_rate: int = int(os.getenv("TTS_SAMPLE_RATE", "16000"))

    # Streaming: emit audio every N new semantic tokens, decoded with a left
    # context of M tokens (trimmed off) to avoid clicks at chunk boundaries.
    # ~50 tokens/sec, so 25 tokens ~= 0.5s of audio per emitted chunk.
    stream_chunk_tokens: int = int(os.getenv("TTS_STREAM_CHUNK_TOKENS", "25"))
    stream_context_tokens: int = int(os.getenv("TTS_STREAM_CONTEXT_TOKENS", "16"))
    # Samples crossfaded between consecutive chunks to remove seam clicks
    # (the same region is decoded twice, in different contexts, and blended).
    # ~24 ms at 16 kHz.
    stream_xfade_samples: int = int(os.getenv("TTS_STREAM_XFADE_SAMPLES", "384"))

    # Split input on punctuation and synthesize sentence-by-sentence: lower
    # time-to-first-audio, and any buffer underrun lands on a natural pause
    # instead of mid-word. Gap of silence inserted between sentences (ms).
    stream_split_sentences: bool = _get_bool("TTS_STREAM_SPLIT", True)
    stream_sentence_gap_ms: int = int(os.getenv("TTS_STREAM_SENTENCE_GAP_MS", "120"))
    stream_max_sentence_chars: int = int(os.getenv("TTS_STREAM_MAX_SENTENCE_CHARS", "200"))

    # Normalize input text before building the prompt.
    normalize: bool = _get_bool("TTS_NORMALIZE", True)


config = Config()
