"""Helper to fetch the assets the server needs.

1. The quantized GGUF (the LLM half) served by llama.cpp.
2. The Spark-TTS BiCodec assets (config + checkpoint) used by the vocoder.

Usage:
    python -m server.download_models --gguf bambara-tts-Q4_K_M.gguf
"""

import argparse
import os

from huggingface_hub import hf_hub_download

from maliba_ai.config.settings import Settings
from maliba_ai.models.models import download_spark_assets


def download_gguf(filename: str, repo: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=out_dir,
    )
    print(f"GGUF ready: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Bambara TTS server assets")
    parser.add_argument(
        "--gguf",
        default="bambara-tts-Q4_K_M.gguf",
        help="GGUF filename to download (e.g. bambara-tts-Q4_K_M.gguf or bambara-tts-Q8_0.gguf)",
    )
    parser.add_argument("--gguf-repo", default=Settings.gguf_repo)
    parser.add_argument("--gguf-dir", default="models")
    parser.add_argument("--spark-dir", default="Spark-TTS-0.5B")
    parser.add_argument(
        "--skip-gguf", action="store_true", help="Only download the BiCodec assets"
    )
    args = parser.parse_args()

    print("Downloading BiCodec (decode-only) assets ...")
    download_spark_assets(model_dir=args.spark_dir, detokenize_only=True)
    print(f"BiCodec ready under: {args.spark_dir}")

    if not args.skip_gguf:
        print(f"Downloading GGUF {args.gguf} from {args.gguf_repo} ...")
        download_gguf(args.gguf, args.gguf_repo, args.gguf_dir)


if __name__ == "__main__":
    main()
