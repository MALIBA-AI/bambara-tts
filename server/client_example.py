"""Minimal client for the Bambara TTS gateway.

    python -m server.client_example --text "Aw ni ce" --speaker Bourama --out hello.wav
"""

import argparse

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Bambara TTS client")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--text", required=True)
    parser.add_argument("--speaker", default="Bourama", help="id or name, e.g. SPEAKER_3 / Bourama")
    parser.add_argument("--out", default="output.wav")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    payload = {
        "text": args.text,
        "speaker": args.speaker,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
    }
    if args.seed is not None:
        payload["seed"] = args.seed

    with httpx.Client(timeout=300) as client:
        resp = client.post(f"{args.url}/tts", json=payload)
        resp.raise_for_status()
        with open(args.out, "wb") as f:
            f.write(resp.content)

    print(f"Saved {args.out} "
          f"({resp.headers.get('X-Audio-Duration', '?')}s @ "
          f"{resp.headers.get('X-Sample-Rate', '?')} Hz)")


if __name__ == "__main__":
    main()
