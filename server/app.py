"""Bambara TTS gateway.

Flow for one synthesis request:

    text  -> normalize -> build Spark-TTS prompt
          -> llama.cpp /completion  (LLM emits bicodec_* tokens)
          -> parse global + semantic token ids
          -> BiCodec vocoder        (tokens -> waveform)
          -> WAV bytes

The LLM runs in a separate llama.cpp process; the BiCodec vocoder runs in-process
here (it is the part llama.cpp cannot serve).
"""

import io
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from maliba_ai.config.settings import Settings, Speakers

from server.config import config
from server.llama_client import LlamaClient, build_prompt, parse_audio_tokens
from server.vocoder import BiCodecVocoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bambara-tts")

# Populated on startup.
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading BiCodec vocoder (device=%s) ...", config.device or "auto")
    state["vocoder"] = BiCodecVocoder(device=config.device, model_dir=config.spark_model_dir)
    state["llama"] = LlamaClient(config.llama_url, timeout=config.llama_timeout)
    logger.info(
        "Ready. sample_rate=%d, llama_url=%s", state["vocoder"].sample_rate, config.llama_url
    )
    try:
        yield
    finally:
        await state["llama"].aclose()


app = FastAPI(title="Bambara TTS", version="2.0.0", lifespan=lifespan)


class TTSRequest(BaseModel):
    text: str = Field(..., description="Bambara text to synthesize")
    speaker: str = Field("SPEAKER_1", description="Speaker id, e.g. SPEAKER_1 .. SPEAKER_10")
    temperature: float = Field(config.default_temperature, ge=0.0, le=2.0)
    top_k: int = Field(config.default_top_k, ge=0)
    top_p: float = Field(config.default_top_p, ge=0.0, le=1.0)
    max_tokens: int = Field(config.default_max_tokens, ge=1)
    seed: Optional[int] = Field(None, description="Optional seed for reproducibility")
    normalize: Optional[bool] = Field(None, description="Override text normalization")


def _resolve_speaker(speaker: str) -> str:
    """Accept either an id (SPEAKER_3) or a name (Bourama) and return the id."""
    s = speaker.strip()
    if s.upper() in Settings.speakers_ids:
        return s.upper()
    try:
        return Speakers.get_speaker_by_name(s).id
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown speaker '{speaker}'. Valid ids: {Settings.speakers_ids}",
        )


def _maybe_normalize(text: str, normalize: Optional[bool]) -> str:
    do_norm = config.normalize if normalize is None else normalize
    if not do_norm:
        return text
    from maliba_ai.utils.utils import normalize_text

    return normalize_text(text)


async def _synthesize(req: TTSRequest) -> np.ndarray:
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text can not be empty")

    speaker_id = _resolve_speaker(req.speaker)
    text = _maybe_normalize(req.text, req.normalize)
    prompt = build_prompt(f"{speaker_id}: {text}")

    try:
        generated = await state["llama"].generate(
            prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            seed=req.seed,
        )
    except Exception as exc:  # llama.cpp unreachable / errored
        logger.exception("llama.cpp generation failed")
        raise HTTPException(status_code=502, detail=f"LLM backend error: {exc}")

    global_ids, semantic_ids = parse_audio_tokens(generated)
    if not semantic_ids:
        raise HTTPException(
            status_code=500,
            detail="Model produced no semantic tokens; try a different seed or check the GGUF vocab.",
        )

    wav = state["vocoder"].tokens_to_wav(global_ids, semantic_ids)
    if wav.size == 0:
        raise HTTPException(status_code=500, detail="Vocoder produced empty audio")
    return wav


def _wav_bytes(wav: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@app.get("/health")
async def health():
    llama_ok = await state["llama"].health()
    return JSONResponse(
        {
            "status": "ok" if llama_ok else "degraded",
            "llama_cpp": "up" if llama_ok else "down",
            "vocoder": "up",
            "sample_rate": state["vocoder"].sample_rate,
        },
        status_code=200 if llama_ok else 503,
    )


@app.get("/speakers")
async def speakers():
    return {
        "speakers": [
            {"name": name, "id": getattr(Speakers, name).id}
            for name in [
                "Adama", "Moussa", "Bourama", "Modibo", "Seydou",
                "Amadou", "Bakary", "Ngolo", "Amara", "Ibrahima",
            ]
        ]
    }


@app.post("/tts")
async def tts(req: TTSRequest):
    """Synthesize speech and return WAV bytes (audio/wav)."""
    wav = await _synthesize(req)
    sample_rate = state["vocoder"].sample_rate
    audio = _wav_bytes(wav, sample_rate)
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={
            "X-Sample-Rate": str(sample_rate),
            "X-Audio-Duration": f"{wav.shape[0] / sample_rate:.3f}",
            "Content-Disposition": 'inline; filename="speech.wav"',
        },
    )


@app.post("/tts/json")
async def tts_json(req: TTSRequest):
    """Synthesize speech and return base64 WAV + metadata."""
    import base64

    wav = await _synthesize(req)
    sample_rate = state["vocoder"].sample_rate
    audio = _wav_bytes(wav, sample_rate)
    return {
        "audio_base64": base64.b64encode(audio).decode("ascii"),
        "format": "wav",
        "sample_rate": sample_rate,
        "duration_seconds": round(wav.shape[0] / sample_rate, 3),
        "num_samples": int(wav.shape[0]),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port)
