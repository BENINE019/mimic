#!/usr/bin/env python3
"""Stage 6a: transcribe the narration track with Muse Voice Transcribe.

Reads clips/audio/<clip>.wav (mono 16 kHz, produced by convert_clips.sh) and
writes out/<clip>/narration.json with the full transcript and timestamped
turns. Object names from names.json are passed as keyword hints so the
model is less likely to mis-hear "eraser" or "wallet".

Usage (from the repo root):
  python -m mimic.transcribe clips/c01.mp4
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests

URL = "https://api.meta.ai/v1/asr/transcribe"
MODEL = "muse-voice-transcribe-1.0"
MAX_BYTES = 32 * 1024 * 1024


def transcribe(wav: Path, keywords: list[str], mode: str = "ENDPOINTING") -> dict:
    token = os.environ.get("MODEL_API_KEY")
    if not token:
        raise RuntimeError("Set MODEL_API_KEY first.")
    if wav.stat().st_size > MAX_BYTES:
        raise RuntimeError(f"{wav} is over the 32 MB limit; shorten the clip.")
    request = {"mode": mode, "model": MODEL, "audioEncoding": "WAV"}
    if keywords:
        request["keywords"] = keywords[:20]
    with wav.open("rb") as audio:
        resp = requests.post(
            URL,
            headers={"Authorization": f"Bearer {token}"},
            files={"request": (None, json.dumps(request), "application/json"),
                   "audio": (wav.name, audio)},
            timeout=300,
        )
    if resp.status_code != 200:
        raise RuntimeError(f"Transcribe failed ({resp.status_code}): {resp.text[:300]}")
    return resp.json()


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 6a: narration to timestamped text")
    p.add_argument("video", type=Path)
    p.add_argument("--audio", type=Path, help="default: clips/audio/<clip>.wav")
    p.add_argument("--output", type=Path, help="default: out/<clip name>")
    p.add_argument("--mode", default="ENDPOINTING", choices=["PUSH_TO_TALK", "ENDPOINTING", "DIARIZATION"])
    args = p.parse_args()

    out_dir = args.output or Path("out") / args.video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = args.audio or Path("clips/audio") / f"{args.video.stem}.wav"
    if not wav.exists():
        p.error(f"{wav} not found; run scripts/convert_clips.sh (it extracts the audio)")

    keywords = []
    names = out_dir / "names.json"
    if names.exists():
        for o in json.loads(names.read_text()):
            keywords += o["phrase"].split()
        keywords = list(dict.fromkeys(keywords))

    result = transcribe(wav, keywords, args.mode)
    turns = [{"start_s": round((t.get("startMs") or 0) / 1000, 2),
              "end_s": round((t.get("endMs") or 0) / 1000, 2) if t.get("endMs") else None,
              "text": t.get("transcript", "").strip()}
             for t in result.get("turns", []) if t.get("transcript", "").strip()]
    data = {"transcript": result.get("transcript", "").strip(),
            "turns": turns,
            "audio_s": round((result.get("audioDurationMs") or 0) / 1000, 1)}
    (out_dir / "narration.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    print(f"Transcript: {data['transcript'] or '(empty)'}")
    for t in turns:
        print(f"  [{t['start_s']:>5.1f}s] {t['text']}")
    print(f"Wrote {out_dir / 'narration.json'}")


if __name__ == "__main__":
    main()
