
import io
import logging
import re
import struct
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from maliba_ai.config.settings import Settings, Speakers

from server.config import config
from server.llama_client import LlamaClient, build_prompt, parse_audio_tokens
from server.vocoder import BiCodecVocoder

# One BiCodec token, e.g. <|bicodec_semantic_1337|> or <|bicodec_global_42|>.
_STREAM_TOKEN_RE = re.compile(r"<\|bicodec_(global|semantic)_(\d+)\|>")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bambara-tts")


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

_STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
async def dashboard():
    """Live streaming test dashboard."""
    return FileResponse(_STATIC_DIR / "index.html")


class TTSRequest(BaseModel):
    text: str = Field(..., description="Bambara text to synthesize")
    speaker: str = Field("SPEAKER_1", description="Speaker id, e.g. SPEAKER_1 .. SPEAKER_10")
    temperature: float = Field(config.default_temperature, ge=0.0, le=2.0)
    top_k: int = Field(config.default_top_k, ge=0)
    top_p: float = Field(config.default_top_p, ge=0.0, le=1.0)
    max_tokens: int = Field(config.default_max_tokens, ge=1)
    seed: Optional[int] = Field(None, description="Optional seed for reproducibility")
    normalize: Optional[bool] = Field(None, description="Override text normalization")


class TTSStreamRequest(TTSRequest):
    response_format: str = Field("pcm", description="Streaming container: pcm | wav")
    chunk_tokens: Optional[int] = Field(
        None, description="Emit audio every N new semantic tokens (default: server config)"
    )
    context_tokens: Optional[int] = Field(
        None, description="Left-context tokens decoded then trimmed to avoid seams"
    )


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


async def _run_tts(
    *,
    text: str,
    speaker: str,
    temperature: float,
    top_k: int,
    top_p: float,
    max_tokens: int,
    seed: Optional[int],
    normalize: Optional[bool],
) -> np.ndarray:
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text can not be empty")

    speaker_id = _resolve_speaker(speaker)
    normalized = _maybe_normalize(text, normalize)
    prompt = build_prompt(f"{speaker_id}: {normalized}")

    try:
        generated = await state["llama"].generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )
    except Exception as exc:  
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


async def _synthesize(req: TTSRequest) -> np.ndarray:
    return await _run_tts(
        text=req.text,
        speaker=req.speaker,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
        seed=req.seed,
        normalize=req.normalize,
    )


def _apply_speed(wav: np.ndarray, speed: float) -> np.ndarray:
    """Time-stretch the waveform (pitch-preserving) for OpenAI-style `speed`."""
    if speed is None or abs(speed - 1.0) < 1e-3:
        return wav
    import librosa

    return librosa.effects.time_stretch(np.ascontiguousarray(wav), rate=speed)


_SNDFILE_FORMATS = {
    "wav": ("WAV", "PCM_16", "audio/wav"),
    "flac": ("FLAC", "PCM_16", "audio/flac"),
    "opus": ("OGG", "OPUS", "audio/ogg"),
}
_FFMPEG_FORMATS = {
    "mp3": (["-f", "mp3"], "audio/mpeg"),
    "aac": (["-f", "adts"], "audio/aac"),
}


def _wav_bytes(wav: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _encode_audio(wav: np.ndarray, sample_rate: int, response_format: str):
    """Encode a waveform to the requested format.

    Returns (audio_bytes, media_type, actual_format). Falls back to wav if the
    requested format cannot be produced in this environment.
    """
    fmt = (response_format or "wav").lower()

    if fmt == "pcm":  
        pcm = np.clip(wav, -1.0, 1.0)
        return (pcm * 32767.0).astype("<i2").tobytes(), "audio/pcm", "pcm"

    if fmt in _SNDFILE_FORMATS:
        sf_format, subtype, media = _SNDFILE_FORMATS[fmt]
        try:
            buf = io.BytesIO()
            sf.write(buf, wav, sample_rate, format=sf_format, subtype=subtype)
            return buf.getvalue(), media, fmt
        except Exception:
            logger.warning("libsndfile cannot encode %s; falling back to wav", fmt)
            return _wav_bytes(wav, sample_rate), "audio/wav", "wav"

    if fmt in _FFMPEG_FORMATS:
        args, media = _FFMPEG_FORMATS[fmt]
        encoded = _ffmpeg_encode(_wav_bytes(wav, sample_rate), args)
        if encoded is not None:
            return encoded, media, fmt
        logger.warning("ffmpeg unavailable; falling back to wav for %s", fmt)
        return _wav_bytes(wav, sample_rate), "audio/wav", "wav"

    return _wav_bytes(wav, sample_rate), "audio/wav", "wav"


def _ffmpeg_encode(wav_bytes: bytes, ffmpeg_args: list):
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        return None
    cmd = ["ffmpeg", "-loglevel", "error", "-i", "pipe:0", *ffmpeg_args, "pipe:1"]
    proc = subprocess.run(cmd, input=wav_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        logger.warning("ffmpeg encode failed: %s", proc.stderr.decode("utf-8", "ignore")[:200])
        return None
    return proc.stdout


# --------------------------------------------------------------------------- #
# Streaming synthesis                                                          #
# The model is autoregressive: llama.cpp streams tokens, we accumulate the     #
# global (speaker) tokens first, then decode semantic tokens in chunks as they #
# arrive so audio starts playing before generation finishes.                   #
# --------------------------------------------------------------------------- #
def _pcm16_bytes(wav: np.ndarray) -> bytes:
    """float32 [-1, 1] waveform -> 16-bit signed little-endian PCM bytes."""
    pcm = np.clip(wav, -1.0, 1.0)
    return (pcm * 32767.0).astype("<i2").tobytes()


def _streaming_wav_header(sample_rate: int, channels: int = 1, bits: int = 16) -> bytes:
    """A WAV header with unknown/maximal sizes, for streaming playback."""
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    return (
        b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits)
        + b"data" + struct.pack("<I", 0xFFFFFFFF)
    )


class _CrossfadeStreamer:
    """Decodes semantic tokens in chunks and stitches them seamlessly.

    Each chunk is decoded with `context` left tokens for vocoder warm-up. To
    avoid a click where two independently-decoded chunks meet, the last
    `xfade` samples of every emitted chunk are held back and crossfaded with the
    *same region* re-decoded (in a longer context) on the next chunk. The result
    is a continuous waveform with no seams and no inserted silence.
    """

    def __init__(self, vocoder, global_ids, semantic, context: int, xfade: int):
        self.v = vocoder
        self.global_ids = global_ids
        self.semantic = semantic
        self.context = context
        self.xfade = xfade
        self.committed = 0           # tokens whose audio is fully emitted
        self.hold = None             # held tail (audio ending at `committed`)

    def _decode(self, end: int):
        start = max(0, self.committed - self.context)
        n = end - start
        if n <= 0 or not self.global_ids:
            return None, 0, 0
        wav = self.v.tokens_to_wav(self.global_ids, self.semantic[start:end])
        if wav.size == 0:
            return None, 0, 0
        spt = len(wav) / n
        c = int(round((self.committed - start) * spt))  # sample index of `committed`
        return wav, c, len(wav)

    def step(self, end: int) -> bytes:
        wav, c, end_s = self._decode(end)
        if wav is None:
            return b""
        parts = []
        if self.hold is not None:
            xf = min(self.xfade, c, self.hold.shape[0])
            if xf > 0:
                fade_out = np.linspace(1.0, 0.0, xf, dtype=np.float32)
                fade_in = np.linspace(0.0, 1.0, xf, dtype=np.float32)
                parts.append(self.hold[-xf:] * fade_out + wav[c - xf:c] * fade_in)
        seg = wav[c:end_s]
        if seg.shape[0] > self.xfade:
            parts.append(seg[:-self.xfade])
            self.hold = seg[-self.xfade:].copy()
        else:
            parts.append(seg)
            self.hold = None
        self.committed = end
        if not parts:
            return b""
        return _pcm16_bytes(np.concatenate(parts))

    def flush(self) -> bytes:
        out = self.step(len(self.semantic)) if len(self.semantic) > self.committed else b""
        if self.hold is not None and self.hold.size:
            out += _pcm16_bytes(self.hold)
            self.hold = None
        return out


async def _pcm_stream(req: TTSStreamRequest) -> AsyncIterator[bytes]:
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text can not be empty")

    vocoder = state["vocoder"]
    chunk_tokens = req.chunk_tokens or config.stream_chunk_tokens
    context_tokens = req.context_tokens if req.context_tokens is not None else config.stream_context_tokens

    speaker_id = _resolve_speaker(req.speaker)
    normalized = _maybe_normalize(req.text, req.normalize)
    prompt = build_prompt(f"{speaker_id}: {normalized}")

    if req.response_format.lower() == "wav":
        yield _streaming_wav_header(vocoder.sample_rate)

    global_ids: list = []
    semantic: list = []
    buffer = ""
    streamer = _CrossfadeStreamer(
        vocoder, global_ids, semantic, context_tokens, config.stream_xfade_samples
    )

    async for piece in state["llama"].stream_generate(
        prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        seed=req.seed,
    ):
        buffer += piece
        matches = list(_STREAM_TOKEN_RE.finditer(buffer))
        if matches:
            for m in matches:
                kind, num = m.group(1), int(m.group(2))
                (global_ids if kind == "global" else semantic).append(num)
            buffer = buffer[matches[-1].end():]

        while len(semantic) - streamer.committed >= chunk_tokens and global_ids:
            data = streamer.step(len(semantic))
            if data:
                yield data

    data = streamer.flush()
    if data:
        yield data


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


@app.post("/tts/stream")
async def tts_stream(req: TTSStreamRequest):
    """Stream audio as it is generated (raw PCM s16le by default, or wav).

    Play directly, e.g.:
      curl -N -X POST localhost:8000/tts/stream -H 'content-type: application/json' \\
        -d '{"text":"Aw ni ce","speaker":"Bourama"}' | ffplay -f s16le -ar 16000 -nodisp -
    """
    sample_rate = state["vocoder"].sample_rate
    fmt = req.response_format.lower()
    media_type = "audio/wav" if fmt == "wav" else "audio/L16"
    return StreamingResponse(
        _pcm_stream(req),
        media_type=media_type,
        headers={
            "X-Sample-Rate": str(sample_rate),
            "Cache-Control": "no-cache",
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



class OpenAISpeechRequest(BaseModel):
    model: str = Field("bambara-tts", description="Accepted for compatibility; ignored")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field("SPEAKER_1", description="Speaker id (SPEAKER_3) or name (Bourama)")
    response_format: str = Field("mp3", description="mp3 | wav | flac | opus | aac | pcm")
    speed: float = Field(1.0, ge=0.25, le=4.0, description="Playback speed (pitch preserved)")
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    seed: Optional[int] = None


@app.post("/v1/audio/speech")
async def openai_audio_speech(req: OpenAISpeechRequest):
    wav = await _run_tts(
        text=req.input,
        speaker=req.voice,
        temperature=req.temperature if req.temperature is not None else config.default_temperature,
        top_k=req.top_k if req.top_k is not None else config.default_top_k,
        top_p=req.top_p if req.top_p is not None else config.default_top_p,
        max_tokens=config.default_max_tokens,
        seed=req.seed,
        normalize=None,
    )
    wav = _apply_speed(wav, req.speed)
    sample_rate = state["vocoder"].sample_rate
    audio, media_type, actual_format = _encode_audio(wav, sample_rate, req.response_format)
    return Response(
        content=audio,
        media_type=media_type,
        headers={
            "X-Sample-Rate": str(sample_rate),
            "X-Audio-Duration": f"{wav.shape[0] / sample_rate:.3f}",
            "X-Audio-Format": actual_format,
            "Content-Disposition": f'inline; filename="speech.{actual_format}"',
        },
    )


@app.get("/v1/models")
async def openai_models():
    return {
        "object": "list",
        "data": [
            {"id": "bambara-tts", "object": "model", "owned_by": "MALIBA-AI"},
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port)
