# Bambara TTS Inference Server

Serve [MALIBA-AI/bambara-tts](https://huggingface.co/MALIBA-AI/bambara-tts) efficiently by splitting the model into the two halves it naturally has:

| Half | What it is | Served by |
|------|------------|-----------|
| **LLM** | Qwen2.5-0.5B finetune, quantized to GGUF ([MALIBA-AI/bambara-tts-gguf](https://huggingface.co/MALIBA-AI/bambara-tts-gguf)) | **llama.cpp** |
| **BiCodec vocoder** | Neural audio codec that turns tokens into a waveform (PyTorch) | **this gateway** |

llama.cpp cannot run the audio codec, so the codec is served separately and the gateway stitches the two together.

## Flow

```
                       ┌────────────────────────── gateway (FastAPI) ──────────────────────────┐
  text ──normalize──▶ build prompt ──▶ POST /completion ──▶ parse <|bicodec_*|> ──▶ BiCodec ──▶ WAV
                                          │  (llama.cpp)        global + semantic     vocoder
                                          ▼                        token ids        (in-process)
                            <|task_tts|>...<|start_global_token|>
```

A request to `POST /tts`:
1. Normalize the Bambara text (same `bambara-text-normalizer` config as training).
2. Build the prompt `…<|start_content|>SPEAKER_3: <text><|end_content|><|start_global_token|>`.
3. Call llama.cpp `/completion`; the LLM emits `<|bicodec_global_*|>` and `<|bicodec_semantic_*|>` tokens.
4. Parse the integers out of the generated string.
5. Decode them to a 16 kHz waveform with the BiCodec vocoder (decode-only — **no wav2vec2 loaded**).
6. Return `audio/wav`.

## Quick start (Docker, recommended)

```bash
# 1. Get the GGUF (and BiCodec assets) into ./models
pip install huggingface_hub
python -m server.download_models --gguf bambara-tts-Q4_K_M.gguf   # or bambara-tts-Q8_0.gguf

# 2. Bring up llama.cpp + gateway
docker compose -f server/docker-compose.yml up --build

# 3. Synthesize
curl -X POST localhost:8000/tts \
  -H 'content-type: application/json' \
  -d '{"text":"Aw ni ce, i ka kɛnɛ wa?","speaker":"Bourama"}' \
  --output hello.wav
```

## Quick start (bare metal)

Two processes. **Terminal 1 — llama.cpp:**

```bash
# Build llama.cpp (https://github.com/ggml-org/llama.cpp) so `llama-server` is on PATH, then:
python -m server.download_models --gguf bambara-tts-Q4_K_M.gguf
MODEL_PATH=models/bambara-tts-Q4_K_M.gguf bash server/run_llama_server.sh
```

**Terminal 2 — gateway:**

```bash
pip install -r server/requirements.txt
pip install --no-deps -e .                      # installs maliba_ai (vocoder + config)
export LLAMA_SERVER_URL=http://localhost:8080
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Then:

```bash
python -m server.client_example --text "Aw ni ce" --speaker Bourama --out hello.wav
```

## Live dashboard

Open **`http://localhost:8000/`** in a browser for a self-contained test page: type Bambara text, pick a speaker and sampling params, hit **Speak (stream)**, and the audio plays automatically as it streams in. It shows a live waveform, the **time-to-first-audio**, seconds received, buffered-ahead, and an **underrun** counter.

Playback uses an **AudioWorklet ring buffer running at a native 16 kHz context** — the main thread feeds PCM and the worklet pulls one sample per frame, emitting silence only on genuine starvation. This avoids the two classic streaming glitches: per-chunk resampling clicks (fixed by the 16 kHz context) and hard gaps from scheduling many small buffers (fixed by the ring buffer + the **Buffer (s)** pre-roll).

> AudioWorklet requires a **secure context** — `https://…` (e.g. an ngrok URL) or `http://localhost`. It will not run over plain `http://<ip>`.

## API

### `POST /tts` → `audio/wav`
```json
{
  "text": "Aw ni ce",
  "speaker": "Bourama",        // id (SPEAKER_3) or name (Bourama)
  "temperature": 0.8,
  "top_k": 50,
  "top_p": 1.0,
  "max_tokens": 2048,
  "seed": null,
  "normalize": true
}
```
Response headers include `X-Sample-Rate` and `X-Audio-Duration`.

Only `text` is required; every other field falls back to the server default
(configurable via the env vars below). Request fields:

| field | type | default | range / notes |
|-------|------|---------|---------------|
| `text` | string | — (required) | Bambara text to synthesize |
| `speaker` | string | `SPEAKER_1` | id (`SPEAKER_3`) or name (`Bourama`) |
| `temperature` | float | `0.8` | `0.0`–`2.0`. ~0.8 is the sweet spot; `<0.6` sounds robotic, `>1.0` adds artifacts |
| `top_k` | int | `50` | `>= 0` |
| `top_p` | float | `1.0` | `0.0`–`1.0` |
| `max_tokens` | int | `2048` | `>= 1`, max audio token budget |
| `seed` | int | `null` | set it for reproducible output (same audio every call) |
| `normalize` | bool | `true` | override server-side text normalization |

Example with sampling params:

```bash
curl -X POST localhost:8000/tts \
  -H 'content-type: application/json' \
  -d '{
        "text": "Aw ni ce, i ka kɛnɛ wa?",
        "speaker": "Bourama",
        "temperature": 0.6,
        "top_k": 30,
        "top_p": 0.9,
        "max_tokens": 1024,
        "seed": 42
      }' \
  --output hello.wav
```

> Tip: interactive docs to try every parameter from the browser are at
> **`http://localhost:8000/docs`**.

### `POST /tts/json`
Same body, returns `{ audio_base64, format, sample_rate, duration_seconds, num_samples }`.

### `POST /tts/stream` (real-time streaming)

The model is autoregressive, so audio can start playing before generation finishes. llama.cpp streams tokens; the gateway collects the **global/speaker tokens first** (they are required to decode *anything*), then decodes the **semantic tokens in chunks** as they arrive and streams the audio out.

Body is the same as `/tts` plus:

| field | default | notes |
|-------|---------|-------|
| `response_format` | `pcm` | `pcm` (raw s16le mono) or `wav` (streaming header) |
| `chunk_tokens` | `25` | emit audio every N new semantic tokens (~50 tok/s, so 25 ≈ 0.5 s). Lower = lower latency, more overhead |
| `context_tokens` | `16` | left-context tokens decoded then trimmed to avoid clicks at chunk seams |

**Token → audio:** semantic tokens run at ~50/sec, so each is ≈ **20 ms** (320 samples @ 16 kHz). First audio arrives once all global tokens + the first `chunk_tokens` semantic tokens are generated. The decoder measures samples-per-token from each decode, so timing is exact even if the codec's rate differs.

Play straight from curl with ffplay:

```bash
curl -N -X POST localhost:8000/tts/stream \
  -H 'content-type: application/json' \
  -d '{"text":"Aw ni ce, i ka kɛnɛ wa?","speaker":"Bourama"}' \
  | ffplay -f s16le -ar 16000 -nodisp -autoexit -
```

Or save the streamed PCM and convert:

```bash
curl -N -X POST localhost:8000/tts/stream \
  -H 'content-type: application/json' \
  -d '{"text":"Aw ni ce","speaker":"Bourama","chunk_tokens":15}' \
  --output out.pcm
ffmpeg -f s16le -ar 16000 -ac 1 -i out.pcm out.wav
```

Lower `chunk_tokens` (e.g. 10–15) for snappier first-audio latency; raise `context_tokens` if you hear faint clicks at chunk boundaries.

**Seamless playback.** Two things keep speech continuous rather than "cut … cut …":
- *Server-side crossfade* — consecutive chunks are decoded in different contexts, so their waveforms don't line up exactly at the seam. The server holds back the last `TTS_STREAM_XFADE_SAMPLES` (~24 ms) of each chunk and crossfades it with the same region re-decoded on the next chunk, removing the click.
- *Client-side pre-roll buffer* — llama.cpp emits tokens in bursts and (on CPU) often slower than real time, which starves naive immediate playback. The dashboard buffers a configurable amount of audio (**Buffer (s)**, default 1.2 s) before starting, so brief stalls don't cause gaps. If you still see **Underruns** climbing on the dashboard, raise the buffer or use a faster (GPU) llama.cpp backend.

### `POST /v1/audio/speech` (OpenAI-compatible)

Drop-in for the [OpenAI TTS API](https://platform.openai.com/docs/api-reference/audio/createSpeech) — point existing OpenAI-SDK code at this server by changing only `base_url`. `model` is accepted and ignored; `voice` is a **native speaker id/name** (`SPEAKER_3` or `Bourama`) — voices are not renamed to OpenAI's.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

client.audio.speech.create(
    model="bambara-tts",
    voice="Bourama",            # SPEAKER_3 .. or any speaker id/name
    input="Aw ni ce, i ka kɛnɛ wa?",
    response_format="mp3",      # mp3 | wav | flac | opus | aac | pcm
    speed=1.0,                  # 0.25–4.0, pitch preserved
).stream_to_file("hello.mp3")
```

| field | default | notes |
|-------|---------|-------|
| `input` | — (required) | text to synthesize |
| `voice` | `SPEAKER_1` | speaker id or name |
| `model` | `bambara-tts` | accepted, ignored |
| `response_format` | `mp3` | `mp3`/`aac` need ffmpeg; `wav`/`flac`/`opus`/`pcm` via libsndfile. Unavailable formats fall back to wav (see `X-Audio-Format` header) |
| `speed` | `1.0` | time-stretch, `0.25`–`4.0` |
| `temperature`/`top_k`/`top_p`/`seed` | server defaults | non-OpenAI extras, optional |

`GET /v1/models` lists `bambara-tts` for SDKs that probe it.

> The Docker image ships ffmpeg, so `mp3`/`aac` work out of the box. On bare metal, install ffmpeg for those formats.

### `GET /speakers`
Lists the 10 speakers (name + id).

### `GET /health`
Reports gateway + llama.cpp status (503 if the LLM backend is down).

## Configuration (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `LLAMA_SERVER_URL` | `http://localhost:8080` | llama.cpp server URL |
| `LLAMA_TIMEOUT` | `300` | seconds |
| `SPARK_MODEL_DIR` | `Spark-TTS-0.5B` | BiCodec asset dir (auto-downloaded if missing) |
| `TTS_DEVICE` | auto | `cpu` / `cuda` for the vocoder |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | gateway bind |
| `TTS_TEMPERATURE` / `TTS_TOP_K` / `TTS_TOP_P` / `TTS_MAX_TOKENS` | `0.8` / `50` / `1.0` / `2048` | sampling defaults |
| `TTS_NORMALIZE` | `true` | normalize input text |

## Notes

- **`--special` matters.** The prompt and outputs rely on added/special tokens; llama.cpp must parse them. The provided run script and compose file set it.
- **Vocoder is decode-only.** `detokenize_only=True` skips the ~300M-param wav2vec2 feature extractor and only downloads `config.yaml` + `BiCodec/` — faster startup, less memory. Voice cloning (encoding reference audio) is not part of this server.
- **GPU:** set `LLAMA_NGL` (e.g. `999`) for the llama.cpp side and `TTS_DEVICE=cuda` for the vocoder; use a CUDA torch wheel in the gateway image.
