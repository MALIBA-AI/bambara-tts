# MALIBA-AI Bambara TTS Inference Server

This is the inference server for the MALIBA-AI Bambara Text-to-Speech system. It serves the model over a simple HTTP API so you can generate Bambara speech from any language or application, stream audio in real time, and even reuse existing OpenAI client code without changes.

The model has two halves, and this server runs each one on the tool best suited to it:

| Half | What it is | Served by |
|------|------------|-----------|
| Language model | A Qwen2.5-0.5B fine-tune, quantized to GGUF ([MALIBA-AI/bambara-tts-gguf](https://huggingface.co/MALIBA-AI/bambara-tts-gguf)) | llama.cpp |
| BiCodec vocoder | A neural audio codec that turns tokens into a waveform (PyTorch) | This gateway |

llama.cpp cannot run the audio codec, so the codec runs inside the gateway and the gateway stitches the two halves together for each request.

## Table of Contents

1. [How It Works](#how-it-works)
2. [Quick Start with Docker](#quick-start-with-docker)
3. [Quick Start on Bare Metal](#quick-start-on-bare-metal)
4. [Live Dashboard](#live-dashboard)
5. [API Reference](#api-reference)
   - 5.1 [Generate Speech](#generate-speech)
   - 5.2 [Generate Speech as JSON](#generate-speech-as-json)
   - 5.3 [Stream Speech in Real Time](#stream-speech-in-real-time)
   - 5.4 [OpenAI Compatible Endpoint](#openai-compatible-endpoint)
   - 5.5 [Speakers and Health](#speakers-and-health)
6. [GPU Acceleration](#gpu-acceleration)
7. [Configuration](#configuration)
8. [Implementation Notes](#implementation-notes)

---

## How It Works

A request to `POST /tts` goes through six steps:

1. Normalize the Bambara text using the same `bambara-text-normalizer` configuration the model was trained on.
2. Build the prompt `<|task_tts|><|start_content|>SPEAKER_3: your text<|end_content|><|start_global_token|>`.
3. Send the prompt to llama.cpp, which generates a string of `<|bicodec_global_*|>` and `<|bicodec_semantic_*|>` tokens.
4. Parse the integer ids out of that string.
5. Decode the ids into a 16 kHz waveform with the BiCodec vocoder. The server loads the decode path only, so the large wav2vec2 feature extractor is never loaded.
6. Return the audio.

The global (speaker) tokens come first and are required before any audio can be decoded. The semantic tokens follow, and each one is worth about 20 ms of audio (320 samples at 16 kHz, roughly 50 tokens per second).

---

## Quick Start with Docker

This is the recommended path. It brings up llama.cpp and the gateway together.

```bash
# 1. Download the GGUF and the BiCodec assets into ./models
pip install huggingface_hub
python -m server.download_models --gguf bambara-tts-Q4_K_M.gguf   # or bambara-tts-Q8_0.gguf

# 2. Start both services
docker compose -f server/docker-compose.yml up --build

# 3. Generate speech
curl -X POST localhost:8000/tts \
  -H 'content-type: application/json' \
  -d '{"text":"Aw ni ce, i ka kɛnɛ wa?","speaker":"Bourama"}' \
  --output hello.wav
```

The compose file runs llama.cpp on the GPU by default. See [GPU Acceleration](#gpu-acceleration) for the requirements and how to confirm the GPU is being used.

---

## Quick Start on Bare Metal

This runs the two services as separate processes.

In the first terminal, start llama.cpp. You need a llama.cpp build with `llama-server` on your PATH (https://github.com/ggml-org/llama.cpp).

```bash
python -m server.download_models --gguf bambara-tts-Q4_K_M.gguf
MODEL_PATH=models/bambara-tts-Q4_K_M.gguf bash server/run_llama_server.sh
```

In the second terminal, start the gateway.

```bash
pip install -r server/requirements.txt
pip install --no-deps -e .                    # installs maliba_ai (vocoder and config)
export LLAMA_SERVER_URL=http://localhost:8080
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Then generate speech with the example client.

```bash
python -m server.client_example --text "Aw ni ce" --speaker Bourama --out hello.wav
```

---

## Live Dashboard

Open `http://localhost:8000/` in a browser for a self contained test page. Type Bambara text, pick a speaker and sampling parameters, press **Speak (stream)**, and the audio plays automatically as it streams in. The page shows a live waveform, the time to first audio, seconds received, how much audio is buffered ahead, and an underrun counter.

Playback uses an AudioWorklet ring buffer running at a native 16 kHz context. The page feeds PCM to the worklet, and the worklet plays one sample per output frame, emitting silence only when it genuinely runs dry. This avoids the two common streaming glitches: per-chunk resampling clicks (avoided by the 16 kHz context) and hard gaps from scheduling many small buffers (avoided by the ring buffer plus the **Buffer (s)** pre-roll).

**Note**: AudioWorklet requires a secure context. Use `https://` (for example an ngrok URL) or `http://localhost`. It does not run over a plain `http://<ip>` address.

---

## API Reference

### Generate Speech

`POST /tts` returns a WAV file (`audio/wav`).

```json
{
  "text": "Aw ni ce",
  "speaker": "Bourama",
  "temperature": 0.8,
  "top_k": 50,
  "top_p": 1.0,
  "max_tokens": 2048,
  "seed": null,
  "normalize": true
}
```

Only `text` is required. Every other field falls back to a server default, which you can change with the environment variables in [Configuration](#configuration).

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `text` | string | required | Bambara text to synthesize |
| `speaker` | string | `SPEAKER_1` | id such as `SPEAKER_3` or a name such as `Bourama` |
| `temperature` | float | `0.8` | Range 0.0 to 2.0. About 0.8 is the sweet spot. Below 0.6 can sound robotic, above 1.0 can add artifacts |
| `top_k` | int | `50` | 0 or greater |
| `top_p` | float | `1.0` | Range 0.0 to 1.0 |
| `max_tokens` | int | `2048` | Maximum audio token budget |
| `seed` | int | `null` | Set it for reproducible output, the same audio on every call |
| `normalize` | bool | `true` | Override server side text normalization |

The response headers include `X-Sample-Rate` and `X-Audio-Duration`.

Example with sampling parameters:

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

Interactive documentation where you can try every parameter from the browser is available at `http://localhost:8000/docs`.

### Generate Speech as JSON

`POST /tts/json` takes the same body and returns JSON with the audio encoded in base64, which is convenient for programmatic clients.

```json
{ "audio_base64": "...", "format": "wav", "sample_rate": 16000, "duration_seconds": 1.84, "num_samples": 29440 }
```

### Stream Speech in Real Time

`POST /tts/stream` returns audio as it is generated, so playback can start before the full clip is ready. llama.cpp streams tokens, the gateway collects the global tokens first, then decodes the semantic tokens in chunks and streams the audio out.

The body is the same as `/tts` plus the following fields.

| Field | Default | Notes |
|-------|---------|-------|
| `response_format` | `pcm` | `pcm` (raw signed 16-bit little-endian mono) or `wav` (streaming header) |
| `chunk_tokens` | `25` | Emit audio every N new semantic tokens. About 50 tokens per second, so 25 is roughly 0.5 s. Lower means lower latency and more overhead |
| `context_tokens` | `16` | Left context tokens that are decoded then trimmed to avoid clicks at chunk seams |
| `split` | `true` | Split text on punctuation and synthesize sentence by sentence |
| `lock_voice` | `true` | Reuse the first sentence's speaker tokens for every later sentence so the voice stays consistent |

Play it straight from curl with ffplay:

```bash
curl -N -X POST localhost:8000/tts/stream \
  -H 'content-type: application/json' \
  -d '{"text":"Aw ni ce, i ka kɛnɛ wa?","speaker":"Bourama"}' \
  | ffplay -f s16le -ar 16000 -nodisp -autoexit -
```

Or save the streamed PCM and convert it:

```bash
curl -N -X POST localhost:8000/tts/stream \
  -H 'content-type: application/json' \
  -d '{"text":"Aw ni ce","speaker":"Bourama","chunk_tokens":15}' \
  --output out.pcm
ffmpeg -f s16le -ar 16000 -ac 1 -i out.pcm out.wav
```

Three mechanisms keep the stream smooth:

- **Sentence by sentence synthesis** (`split`, on by default). The text is split on punctuation and each sentence is generated in sequence into one continuous stream, with a short silence between sentences set by `TTS_STREAM_SENTENCE_GAP_MS`. This lowers the time to first audio because the first sentence is short, and it makes any pause fall on a sentence boundary so it sounds natural.
- **Consistent voice across sentences** (`lock_voice`, on by default). The speaker is encoded by the global tokens, which the model samples at the start of each generation. Without locking, every sentence would sample its own and the voice would drift. With locking, the global tokens generated for the first sentence are injected into the prompt of every later sentence, so the model only generates new content and the voice stays identical.
- **Overlapping generation and decoding.** Inside each sentence, a producer keeps pulling tokens from llama.cpp while the previous chunk is decoded in a worker thread, so the two stages run at the same time.
- **Crossfade at chunk seams.** Consecutive chunks are decoded in different contexts, so their waveforms do not line up exactly where they meet. The server holds back the last `TTS_STREAM_XFADE_SAMPLES` (about 24 ms) of each chunk and crossfades it with the same region re-decoded on the next chunk, which removes the click.

Real time streaming needs the language model to generate tokens at least as fast as the audio plays, which is about 50 tokens per second. A GPU clears this easily. A CPU does not, so on CPU the stream will keep running dry no matter how large the buffer is. See [GPU Acceleration](#gpu-acceleration).

### OpenAI Compatible Endpoint

`POST /v1/audio/speech` mirrors the [OpenAI text to speech API](https://platform.openai.com/docs/api-reference/audio/createSpeech), so existing OpenAI client code can target this server by changing only the `base_url`. The `model` field is accepted and ignored. The `voice` field is a native speaker id or name such as `SPEAKER_3` or `Bourama`, the speakers are not renamed to OpenAI voices.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

client.audio.speech.create(
    model="bambara-tts",
    voice="Bourama",
    input="Aw ni ce, i ka kɛnɛ wa?",
    response_format="mp3",
    speed=1.0,
).stream_to_file("hello.mp3")
```

| Field | Default | Notes |
|-------|---------|-------|
| `input` | required | Text to synthesize |
| `voice` | `SPEAKER_1` | Speaker id or name |
| `model` | `bambara-tts` | Accepted and ignored |
| `response_format` | `mp3` | `mp3` and `aac` need ffmpeg. `wav`, `flac`, `opus`, and `pcm` use libsndfile. A format that cannot be produced falls back to wav, shown in the `X-Audio-Format` header |
| `speed` | `1.0` | Time stretch from 0.25 to 4.0, pitch preserved |
| `temperature`, `top_k`, `top_p`, `seed` | server defaults | Optional extras beyond the OpenAI fields |

`GET /v1/models` lists `bambara-tts` for SDKs that probe it.

**Note**: The Docker image ships with ffmpeg, so `mp3` and `aac` work out of the box. On bare metal, install ffmpeg for those formats.

### Speakers and Health

`GET /speakers` lists the ten speakers with their names and ids.

`GET /health` reports the status of the gateway and llama.cpp. It returns 503 if the language model backend is down.

---

## GPU Acceleration

The language model is the bottleneck for streaming. On CPU the 0.5B model generates well below real time, so live streaming will stutter. On a GPU it runs many times faster than real time and streaming is smooth.

The Docker compose file already uses the CUDA image, reserves the GPU for the container, and offloads all layers with `--n-gpu-layers 99`. The host needs the NVIDIA Container Toolkit so Docker can pass the GPU through:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Recreate the llama.cpp service and confirm the GPU is in use:

```bash
docker compose -f server/docker-compose.yml up -d --force-recreate llama
docker logs server-llama-1 2>&1 | grep -iE "CUDA|offloaded"   # expect the model offloaded to the GPU
nvidia-smi                                                     # expect non-zero memory and a llama process
```

On bare metal, the launch script offloads all layers by default. Override the count with `LLAMA_NGL` if needed.

To measure throughput, the magic number is 32000 bytes per second, which is 16000 Hz times 2 bytes, equal to real time:

```bash
start=$(date +%s.%N)
curl -N -s -X POST http://localhost:8000/tts/stream -H 'content-type: application/json' \
  -d '{"text":"Aw ni ce, i ka kɛnɛ wa? Ne tɔgɔ ye Adama.","speaker":"Bourama","split":false}' \
  --output /tmp/s.pcm
end=$(date +%s.%N)
python3 -c "b=$(wc -c </tmp/s.pcm); e=$end-$start; print(f'{b/e:.0f} B/s ({b/32000/e:.2f}x realtime)')"
```

A result above 1x means the server produces audio faster than it plays, so streaming will be smooth.

---

## Configuration

All settings are environment variables.

| Variable | Default | Meaning |
|----------|---------|---------|
| `LLAMA_SERVER_URL` | `http://localhost:8080` | llama.cpp server URL |
| `LLAMA_TIMEOUT` | `300` | Request timeout in seconds |
| `SPARK_MODEL_DIR` | `Spark-TTS-0.5B` | BiCodec asset directory, downloaded automatically if missing |
| `TTS_DEVICE` | auto | `cpu` or `cuda` for the vocoder |
| `HOST` | `0.0.0.0` | Gateway bind address |
| `PORT` | `8000` | Gateway bind port |
| `TTS_TEMPERATURE` | `0.8` | Default sampling temperature |
| `TTS_TOP_K` | `50` | Default top-k |
| `TTS_TOP_P` | `1.0` | Default top-p |
| `TTS_MAX_TOKENS` | `2048` | Default audio token budget |
| `TTS_NORMALIZE` | `true` | Normalize input text |
| `TTS_STREAM_CHUNK_TOKENS` | `25` | Semantic tokens per streamed chunk |
| `TTS_STREAM_CONTEXT_TOKENS` | `16` | Left context tokens per chunk |
| `TTS_STREAM_XFADE_SAMPLES` | `384` | Crossfade length between chunks |
| `TTS_STREAM_SPLIT` | `true` | Split text into sentences when streaming |
| `TTS_STREAM_SENTENCE_GAP_MS` | `120` | Silence between sentences |
| `TTS_STREAM_MAX_SENTENCE_CHARS` | `200` | Maximum characters per sentence before further splitting |

---

## Implementation Notes

- **Special tokens must be parsed.** The prompt and the model output rely on added special tokens, so llama.cpp must run with `--special`. The provided launch script and compose file set this.
- **The vocoder is decode only.** The server loads the BiCodec decode path and skips the wav2vec2 feature extractor, which saves about 300M parameters of memory and a large download. Voice cloning, which encodes reference audio, is not part of this server.
- **The gateway does not need unsloth.** It depends on PyTorch, transformers, and the audio stack only, so its image stays small.
