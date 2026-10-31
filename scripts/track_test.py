#!/usr/bin/env python3
"""SAM 3.1 tracking sanity check.

Uploads one short clip to the Meta Model API, tracks each concept you name
(one request per concept, as the API requires), then:

  1. prints a continuity report per tracked object (first/last frame,
     how many frames it was visible, longest gap), and
  2. writes an overlay MP4 so you can see whether identities survive your
     hand passing over the objects.

Usage:
  export MODEL_API_KEY="..."
  python track_test.py clips/test.mp4 --concept "red cup" --concept "sock" --output out/test

The verdict you are looking for: every object has ONE track that spans the
clip with only short gaps. Many short tracks for one object means the tracker
lost it and re-detected it with a new identity.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from meta_sam_parser import decode_mask_to_raster
from sam_api import segment_video, upload_video, validate_result

# Distinct colours per object id (BGR for OpenCV).
PALETTE = [
    (80, 200, 60), (60, 120, 255), (255, 120, 60), (200, 60, 200),
    (40, 220, 220), (220, 220, 40), (120, 80, 255), (255, 80, 160),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAM 3.1 video tracking sanity check")
    p.add_argument("video", type=Path, nargs="?", help="MP4 clip, ideally 720p and under 30 s")
    p.add_argument("--file-id", help="reuse a previous upload instead of the video path")
    p.add_argument("--concept", action="append", required=True,
                   help="short noun phrase; repeat for more objects (one API call each)")
    p.add_argument("--output", type=Path, required=True, help="folder for report and overlay")
    p.add_argument("--max-gap", type=int, default=15,
                   help="frames; a gap longer than this counts as a lost track (default 15)")
    args = p.parse_args()
    if (args.video is None) == (args.file_id is None):
        p.error("give a video path or --file-id, not both")
    return args


def continuity_report(result, concept: str, max_gap: int) -> list[dict]:
    """One row per object id: where it was seen and how continuous the track is."""
    frames_by_obj: dict[str, set[int]] = defaultdict(set)
    for rec in result.records:
        if rec.kind == "mask" and rec.frame is not None:
            frames_by_obj[rec.object_id].add(rec.frame.frame_index)

    rows = []
    for obj, frames in sorted(frames_by_obj.items(), key=lambda kv: int(kv[0])):
        seq = sorted(frames)
        gaps = [b - a - 1 for a, b in zip(seq, seq[1:])]
        longest = max(gaps) if gaps else 0
        rows.append({
            "concept": concept,
            "object_id": obj,
            "first_frame": seq[0],
            "last_frame": seq[-1],
            "visible_frames": len(seq),
            "span_frames": seq[-1] - seq[0] + 1,
            "longest_gap": longest,
            "verdict": "OK" if longest <= max_gap else "LOST (gap > %d)" % max_gap,
        })
    return rows


def masks_by_frame(result) -> dict[int, list]:
    """frame_index -> list of (object_id, bounds, raster uint8 HxW)."""
    out: dict[int, list] = defaultdict(list)
    for rec in result.records:
        if rec.kind != "mask" or rec.frame is None:
            continue
        raw = decode_mask_to_raster(rec.mask)
        raster = np.frombuffer(raw, dtype=np.uint8).reshape(rec.mask.height, rec.mask.width)
        out[rec.frame.frame_index].append((rec.object_id, rec.bounds, raster))
    return out


def render_overlay(video: Path, per_concept: list[tuple[str, dict[int, list]]], out_path: Path) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    colour_key = {}  # (concept, object_id) -> colour
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        overlay = frame.copy()
        for concept, by_frame in per_concept:
            for obj, b, raster in by_frame.get(idx, []):
                key = (concept, obj)
                if key not in colour_key:
                    colour_key[key] = PALETTE[len(colour_key) % len(PALETTE)]
                colour = colour_key[key]
                x0, y0 = max(0, int(b.left)), max(0, int(b.top))
                x1, y1 = min(w, x0 + raster.shape[1]), min(h, y0 + raster.shape[0])
                if x1 <= x0 or y1 <= y0:
                    continue
                sub = raster[: y1 - y0, : x1 - x0].astype(bool)
                region = overlay[y0:y1, x0:x1]
                region[sub] = (0.45 * np.array(colour) + 0.55 * region[sub]).astype(np.uint8)
                cv2.rectangle(overlay, (x0, y0), (x1, y1), colour, 2)
                cv2.putText(overlay, f"{concept} #{obj}", (x0, max(12, y0 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
        cv2.putText(overlay, f"frame {idx}", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(overlay)
        idx += 1
    cap.release()
    writer.release()
    return idx


async def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.file_id:
        file_id = args.file_id
    else:
        if not args.video.is_file():
            raise FileNotFoundError(args.video)
        print(f"Uploading {args.video} ...")
        file_id = await upload_video(args.video)
        print(f"Uploaded as {file_id}  (reuse with --file-id {file_id})")

    rows: list[dict] = []
    per_concept: list[tuple[str, dict[int, list]]] = []
    for concept in args.concept:
        print(f"\nTracking '{concept}' ...")
        result = await segment_video(file_id, concept)
        validate_result(result)
        rows.extend(continuity_report(result, concept, args.max_gap))
        per_concept.append((concept, masks_by_frame(result)))

    # Report
    (args.output / "report.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("\n=== Track continuity ===")
    hdr = f"{'concept':<18}{'id':<5}{'first':>6}{'last':>6}{'seen':>6}{'span':>6}{'gap':>5}  verdict"
    print(hdr)
    for r in rows:
        print(f"{r['concept']:<18}{r['object_id']:<5}{r['first_frame']:>6}{r['last_frame']:>6}"
              f"{r['visible_frames']:>6}{r['span_frames']:>6}{r['longest_gap']:>5}  {r['verdict']}")
    if not rows:
        print("(no masks returned; check the concept wording or the clip)")

    # Overlay video (needs the local file; skipped when only --file-id was given)
    if args.video is not None:
        out_mp4 = args.output / "overlay.mp4"
        n = render_overlay(args.video, per_concept, out_mp4)
        print(f"\nWrote {out_mp4} ({n} frames). Open it and watch the labels through the occlusions.")
    else:
        print("\n(no local video path, so no overlay was rendered)")

    print("\nHow to read it: one object = one id for the whole clip is a PASS. "
          "An object whose id changes after your hand covers it is the failure to look for.")


if __name__ == "__main__":
    asyncio.run(main())
