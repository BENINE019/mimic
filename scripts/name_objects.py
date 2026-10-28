#!/usr/bin/env python3
"""Let Muse Spark name the objects, then track them with SAM 3.1.

This is the honest version of track_test.py: nobody types the object names.
Muse Spark looks at a few frames from the clip and lists the objects as short,
visually groundable noun phrases. Those phrases become the SAM 3.1 prompts.

Usage:
  export MODEL_API_KEY="..."
  # 1) just see what Muse Spark names
  python name_objects.py clips/test.mp4
  # 2) name, track, and compare against your hand-named run
  python name_objects.py clips/test.mp4 --run --compare out/test/report.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
from openai import OpenAI

BASE_URL = "https://api.meta.ai/v1"
DEFAULT_MODEL = "muse-spark-1.3"  # check the Model list in the dashboard if this errors

PROMPT = """These are {n} frames from one short video of a person moving objects on a table.

List every distinct physical object that sits on the table or gets picked up and moved
at any point in the video. For each, give a short noun phrase that a segmentation
model can find from text alone:
- 1 to 3 words, colour plus object type, e.g. "red pencil", "black wallet".
- Use the most common everyday name, not a brand or a guess at the material.
- Each phrase must pick out only one object in the scene; add a distinguishing
  colour or size word if two objects share a type.
- Do NOT list hands, arms, people, the table, the floor, walls, or the camera.
- Include objects that only appear in later frames.
"""

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "scene_objects",
        "schema": {
            "type": "object",
            "properties": {
                "objects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "phrase": {"type": "string"},
                            "first_seen_frame": {
                                "type": "integer",
                                "description": "index (1-based) of the first sampled frame showing it",
                            },
                        },
                        "required": ["phrase", "first_seen_frame"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["objects"],
            "additionalProperties": False,
        },
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Muse Spark names objects, SAM 3.1 tracks them")
    p.add_argument("video", type=Path)
    p.add_argument("--frames", type=int, default=4,
                   help="how many frames to sample evenly across the clip (default 4)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output", type=Path, default=Path("out/auto"))
    p.add_argument("--run", action="store_true", help="run track_test.py on the named objects")
    p.add_argument("--compare", type=Path, help="report.json from the hand-named run")
    return p.parse_args()


def sample_frames(video: Path, n: int, width: int = 768) -> list[tuple[int, str]]:
    """Return [(frame_index, jpeg data URL)] for n evenly spaced frames."""
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Could not read frames from {video}")
    picks = [round(i * (total - 1) / max(1, n - 1)) for i in range(n)]
    out = []
    for idx in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        if w > width:
            frame = cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        out.append((idx, "data:image/jpeg;base64," + base64.b64encode(buf).decode()))
    cap.release()
    return out


def name_objects(frames: list[tuple[int, str]], model: str) -> list[dict]:
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        raise RuntimeError("Set MODEL_API_KEY first.")
    client = OpenAI(base_url=BASE_URL, api_key=key)
    content = [{"type": "text", "text": PROMPT.format(n=len(frames))}]
    for i, (_, url) in enumerate(frames, 1):
        content.append({"type": "text", "text": f"Frame {i}:"})
        content.append({"type": "image_url", "image_url": {"url": url}})
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        response_format=SCHEMA,
    )
    data = json.loads(resp.choices[0].message.content)
    # de-duplicate, keep order
    seen, objs = set(), []
    for o in data.get("objects", []):
        phrase = o["phrase"].strip().lower()
        if phrase and phrase not in seen:
            seen.add(phrase)
            objs.append({"phrase": phrase, "first_seen_frame": o["first_seen_frame"]})
    return objs


def compare(hand_report: Path, auto_report: Path) -> None:
    hand = json.loads(hand_report.read_text())
    auto = json.loads(auto_report.read_text())

    def summary(rows):
        by = {}
        for r in rows:
            s = by.setdefault(r["concept"], {"tracks": 0, "seen": 0, "gap": 0})
            s["tracks"] += 1
            s["seen"] = max(s["seen"], r["visible_frames"])
            s["gap"] = max(s["gap"], r["longest_gap"])
        return by

    print("\n=== Hand-named vs Muse-Spark-named ===")
    for label, rows in (("hand", hand), ("auto", auto)):
        print(f"\n[{label}]  {'concept':<22}{'tracks':>7}{'seen':>7}{'gap':>6}")
        for c, s in summary(rows).items():
            print(f"        {c:<22}{s['tracks']:>7}{s['seen']:>7}{s['gap']:>6}")
    print("\nLook for: same number of objects found, one track each, similar 'seen' counts. "
          "A missing or extra object means the naming step needs work, not the tracker.")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frames = sample_frames(args.video, args.frames)
    print(f"Sampled frames {[i for i, _ in frames]} from {args.video}")
    objs = name_objects(frames, args.model)
    (args.output / "names.json").write_text(json.dumps(objs, indent=2) + "\n")

    if not objs:
        print("Muse Spark returned no objects. Check the clip or try --frames 6.")
        sys.exit(1)
    print("\nMuse Spark named:")
    for o in objs:
        print(f"  - {o['phrase']}  (first in sampled frame {o['first_seen_frame']})")

    cmd = [sys.executable, "track_test.py", str(args.video), "--output", str(args.output)]
    for o in objs:
        cmd += ["--concept", o["phrase"]]
    print("\nTracking command:\n  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))

    if args.run:
        subprocess.run(cmd, check=True)
        if args.compare:
            compare(args.compare, args.output / "report.json")


if __name__ == "__main__":
    main()
