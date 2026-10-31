#!/usr/bin/env python3
"""Stage 6b: infer the goal of the demonstration.

Muse Spark sees the first and last frame (objects boxed and labelled), the
event list from stage 5, and the narration from stage 6a, and returns one to
three candidate goals in a fixed vocabulary, each with a confidence and a
reason. If the top two are close, it also writes the clarifying question a
robot should ask before acting.

Goal vocabulary (the planner knows exactly these five):
  put_all(category, container)     all pencils in the cup
  move(object, relation, anchor)   the wallet next to the cup / to the left
  arrange(category, pattern)       the three blocks in a row
  stack(top, base)                 the small eraser on the big eraser
  swap(a, b)                       the cup and the pencil switch places

Writes out/<clip>/intent.json.

Usage (from the repo root):
  python -m mimic.intent clips/c01.mp4
  python -m mimic.intent clips/c01.mp4 --no-narration     # ablation: events only
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

import cv2
from openai import OpenAI

BASE_URL = "https://api.meta.ai/v1"
DEFAULT_MODEL = "muse-spark-1.3"
CLARIFY_MARGIN = 0.2

GOAL_TYPES = ["put_all", "move", "arrange", "stack", "swap"]

SYSTEM = """You infer what a person was trying to achieve from one demonstration, so that a robot can achieve the same thing in a different scene (objects in other positions, more or fewer of them, extra distractors).

You will get: the first and last frame of a short video with the objects boxed and labelled, a list of detected events (which object moved, when, from where to where, into or onto what), the roles of all objects, and the person's spoken narration with timestamps.

Rules:
- Describe the GOAL, not the motion. "All pencils in the cup", not "move object 2 to (26, -17)".
- Use exactly one of these goal types and fill only its arguments; leave the others as empty strings:
  put_all(category, container)  move(object, relation, anchor)  arrange(category, pattern)  stack(top, base)  swap(a, b)
- scope is "all" when every object of that kind should be handled (even ones not present in this demo), "this" when only the specific object(s) shown matter.
- order_matters is true only if doing the moves in another order would be wrong.
- Narration outranks guesses: if the person says "all the pencils", scope is "all". If narration and events disagree, trust the events for WHAT happened and the narration for WHY.
- Objects that were never touched are context, not part of the goal, unless they serve as an anchor or container.
- Use the object names as given in the data (e.g. "black pencil#0"); for categories use the plain noun ("pencil").
- Give 1 to 3 candidates sorted by confidence (confidences sum to at most 1). If the top two differ by less than 0.2, write one short clarifying question a robot should ask; otherwise leave it empty.
- Write a one-sentence task description in plain language that a person would say to a robot."""

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "demonstration_goal",
        "schema": {
            "type": "object",
            "properties": {
                "task_sentence": {"type": "string"},
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": GOAL_TYPES},
                            "category": {"type": "string"},
                            "container": {"type": "string"},
                            "object": {"type": "string"},
                            "relation": {"type": "string"},
                            "anchor": {"type": "string"},
                            "pattern": {"type": "string"},
                            "top": {"type": "string"},
                            "base": {"type": "string"},
                            "a": {"type": "string"},
                            "b": {"type": "string"},
                            "scope": {"type": "string", "enum": ["this", "all"]},
                            "order_matters": {"type": "boolean"},
                            "order": {"type": "array", "items": {"type": "string"}},
                            "confidence": {"type": "number"},
                            "reason": {"type": "string"},
                        },
                        "required": ["type", "category", "container", "object", "relation", "anchor",
                                     "pattern", "top", "base", "a", "b", "scope", "order_matters",
                                     "order", "confidence", "reason"],
                        "additionalProperties": False,
                    },
                },
                "clarifying_question": {"type": "string"},
            },
            "required": ["task_sentence", "candidates", "clarifying_question"],
            "additionalProperties": False,
        },
    },
}

ARG_NAMES = {
    "put_all": ["category", "container"],
    "move": ["object", "relation", "anchor"],
    "arrange": ["category", "pattern"],
    "stack": ["top", "base"],
    "swap": ["a", "b"],
}


def goal_string(c: dict) -> str:
    args = ", ".join(c[k] for k in ARG_NAMES[c["type"]] if c.get(k))
    return f"{c['type']}({args})"


def labelled_frame(video: Path, tracks: dict, frame_index: int, width: int = 768) -> str:
    """JPEG data URL of one frame with object boxes and keys drawn on it."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} of {video}")
    for o in tracks["objects"]:
        if o.get("is_hand") or o.get("is_marker_sheet"):
            continue
        near = min(o["frames"], key=lambda f: abs(f["f"] - frame_index), default=None)
        if near is None or abs(near["f"] - frame_index) > 15:
            continue
        x0, y0, x1, y1 = map(int, near["bbox"])
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 200, 255), 2)
        cv2.putText(frame, o["key"], (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA)
    h, w = frame.shape[:2]
    if w > width:
        frame = cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def describe(events: dict, narration: dict | None) -> str:
    lines = ["OBJECTS (role, start -> end position in cm on the table):"]
    for o in events["objects"]:
        att = f", attached to {o['attached_to']}" if o["attached_to"] else ""
        lines.append(f"- {o['key']}: {o['role']}{att}, start {o['start_xy']} -> end {o['end_xy']}")
    lines.append("\nEVENTS (in order):")
    if not events["events"]:
        lines.append("- none detected")
    for e in events["events"]:
        extra = f" together with {', '.join(e['carried'])}" if e["carried"] else ""
        where = f"into {e['in']}" if e["in"] else f"onto {e['on']}" if e.get("on") else f"to {e['place_xy']}"
        lines.append(f"- {e['grasp_s']}s to {e['release_s']}s: {e['object']}{extra} moved from {e['pick_xy']} {where}")
    lines.append("\nNARRATION:")
    if narration and narration.get("turns"):
        for t in narration["turns"]:
            lines.append(f"- [{t['start_s']}s] {t['text']}")
    elif narration and narration.get("transcript"):
        lines.append(f"- {narration['transcript']}")
    else:
        lines.append("- (none)")
    return "\n".join(lines)


def infer(video: Path, out_dir: Path, model: str, use_narration: bool) -> dict:
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        raise RuntimeError("Set MODEL_API_KEY first.")
    tracks = json.loads((out_dir / "table_tracks.json").read_text())
    events = json.loads((out_dir / "events.json").read_text())
    narration = None
    if use_narration and (out_dir / "narration.json").exists():
        narration = json.loads((out_dir / "narration.json").read_text())

    last = tracks["frames"] - 1
    first_img = labelled_frame(video, tracks, 0)
    last_img = labelled_frame(video, tracks, last)
    text = describe(events, narration)

    client = OpenAI(base_url=BASE_URL, api_key=key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": "First frame:"},
                {"type": "image_url", "image_url": {"url": first_img}},
                {"type": "text", "text": "Last frame:"},
                {"type": "image_url", "image_url": {"url": last_img}},
                {"type": "text", "text": text + "\n\nWhat was the person trying to achieve?"},
            ]},
        ],
        response_format=SCHEMA,
    )
    data = json.loads(resp.choices[0].message.content)
    cands = sorted(data.get("candidates", []), key=lambda c: -c["confidence"])
    data["candidates"] = cands
    for c in cands:
        c["goal"] = goal_string(c)
    # Enforce the clarification rule ourselves, whatever the model decided.
    if len(cands) >= 2 and cands[0]["confidence"] - cands[1]["confidence"] < CLARIFY_MARGIN:
        data["needs_clarification"] = True
        if not data.get("clarifying_question"):
            data["clarifying_question"] = (f"Did you mean {cands[0]['goal']} or {cands[1]['goal']}?")
    else:
        data["needs_clarification"] = False
        data["clarifying_question"] = ""
    data["model"] = model
    data["used_narration"] = narration is not None
    data["prompt_text"] = text
    return data


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 6b: infer the goal with Muse Spark")
    p.add_argument("video", type=Path)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--no-narration", action="store_true", help="ablation: ignore the transcript")
    p.add_argument("--output", type=Path, help="default: out/<clip name>")
    args = p.parse_args()

    out_dir = args.output or Path("out") / args.video.stem
    for need in ("table_tracks.json", "events.json"):
        if not (out_dir / need).exists():
            p.error(f"{out_dir / need} not found; run the earlier stages first")

    data = infer(args.video, out_dir, args.model, not args.no_narration)
    name = "intent.json" if not args.no_narration else "intent_no_narration.json"
    (out_dir / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    print(f"Task: {data['task_sentence']}")
    for i, c in enumerate(data["candidates"], 1):
        flags = f"scope={c['scope']}" + (", ordered" if c["order_matters"] else "")
        print(f"  {i}. {c['goal']:<40} {c['confidence']:.2f}  ({flags})  {c['reason']}")
    if data["needs_clarification"]:
        print(f"ASK: {data['clarifying_question']}")
    print(f"Wrote {out_dir / name}")


if __name__ == "__main__":
    main()
