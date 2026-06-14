#!/usr/bin/env bash
# Launch the llama.cpp server for the Bambara TTS GGUF.
#
# Requires a llama.cpp build with `llama-server` on PATH, or run the official
# container (see docker-compose.yml). The GGUF special tokens (<|task_tts|>,
# <|bicodec_*|>, ...) must be parsed from the prompt, so we keep --special.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-models/bambara-tts-Q4_K_M.gguf}"
HOST="${LLAMA_HOST:-0.0.0.0}"
PORT="${LLAMA_PORT:-8080}"
CTX="${LLAMA_CTX:-4096}"
NGL="${LLAMA_NGL:-99}"       # GPU layers to offload; 0 = CPU only, 99 = all
PARALLEL="${LLAMA_PARALLEL:-1}"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Model not found at $MODEL_PATH" >&2
  echo "Download it first: python -m server.download_models --gguf bambara-tts-Q4_K_M.gguf" >&2
  exit 1
fi

exec llama-server \
  --model "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --ctx-size "$CTX" \
  --n-gpu-layers "$NGL" \
  --parallel "$PARALLEL" \
  --special \
  --no-webui
