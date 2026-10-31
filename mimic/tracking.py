#!/usr/bin/env python3
"""Stage 3: track named objects (and the hand) through a clip with SAM 3.1.

Saves out/<clip>/tracks.json: per object, per frame, the bounding box, mask
centroid, a "base point" (where the object meets the table), mask area and
confidence. Raw API results are cached per concept, so reruns are free.

Usage (from the repo root):
  python -m mimic.tracking clips/c01.mp4 --names out/c01/names.json
  python -m mimic.tracking clips/c01.mp4 --concept "red pencil" --concept cup
The hand is tracked by default; turn that off with --no-hand.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import re
from pathlib import Path

import cv2
import numpy as np
from meta_sam_parser import decode_mask_to_raster

from sam_api import segment_video, upload_video, validate_result

BASE_FRACTION = 0.2  # base point = centroid of the lowest 20% of mask rows


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def video_info(video: Path) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def place_raster(rec, frame_w: int, frame_h: int):
    """Return (raster bool HxW, x offset, y offset) in frame pixel coordinates.

    The API may return full-frame masks or masks local to their bounds; both
    are handled so later stages never need to care.
    """
    raw = decode_mask_to_raster(rec.mask)
    raster = np.frombuffer(raw, dtype=np.uint8).reshape(rec.mask.height, rec.mask.width) > 0
    if rec.mask.width == frame_w and rec.mask.height == frame_h:
        return raster, 0, 0
    b = rec.bounds
    return raster, int(round(b.left)), int(round(b.top))


def mask_features(raster: np.ndarray, x0: int, y0: int) -> dict | None:
    ys, xs = np.nonzero(raster)
    if len(xs) == 0:
        return None
    ys = ys + y0
    xs = xs + x0
    top, bottom = int(ys.min()), int(ys.max())
    cut = bottom - BASE_FRACTION * max(1, bottom - top)
    low = ys >= cut
    return {
        "bbox": [int(xs.min()), top, int(xs.max()), bottom],
        "centroid": [float(xs.mean()), float(ys.mean())],
        "base": [float(xs[low].mean()), float(ys[low].mean())],
        "area": int(len(xs)),
    }


async def get_file_id(video: Path, cache_dir: Path) -> str:
    cache = cache_dir / "file_id.txt"
    if cache.exists():
        return cache.read_text().strip()
    print(f"Uploading {video} ...")
    file_id = await upload_video(video)
    cache.write_text(file_id + "\n")
    return file_id


async def get_result(file_id: str, concept: str, cache_dir: Path):
    cache = cache_dir / f"sam_{slug(concept)}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    print(f"Tracking '{concept}' ...")
    result = await segment_video(file_id, concept)
    validate_result(result)
    cache.write_bytes(pickle.dumps(result))
    return result


async def track(video: Path, concepts: list[str], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    info = video_info(video)
    file_id = await get_file_id(video, cache_dir)

    objects = []
    for concept in concepts:
        result = await get_result(file_id, concept, cache_dir)
        per_obj: dict[str, list] = {}
        for rec in result.records:
            if rec.kind != "mask" or rec.frame is None:
                continue
            raster, x0, y0 = place_raster(rec, info["width"], info["height"])
            feats = mask_features(raster, x0, y0)
            if feats is None:
                continue
            feats["f"] = rec.frame.frame_index
            feats["conf"] = None if rec.confidence is None else float(rec.confidence)
            per_obj.setdefault(rec.object_id, []).append(feats)
        for obj_id, frames in sorted(per_obj.items(), key=lambda kv: int(kv[0])):
            frames.sort(key=lambda d: d["f"])
            objects.append({
                "key": f"{concept}#{obj_id}",
                "concept": concept,
                "object_id": obj_id,
                "is_hand": concept == "hand",
                "frames": frames,
            })

    data = {"video": str(video), **info, "objects": objects}
    (out_dir / "tracks.json").write_text(json.dumps(data) + "\n")
    print(f"\nWrote {out_dir / 'tracks.json'}")
    for o in objects:
        fs = o["frames"]
        print(f"  {o['key']:<26} frames {fs[0]['f']:>4} to {fs[-1]['f']:>4}, visible {len(fs)}")
    return data


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 3: SAM 3.1 tracking to tracks.json")
    p.add_argument("video", type=Path)
    p.add_argument("--names", type=Path, help="names.json from name_objects.py")
    p.add_argument("--concept", action="append", default=[])
    p.add_argument("--no-hand", action="store_true")
    p.add_argument("--output", type=Path, help="default: out/<clip name>")
    args = p.parse_args()

    concepts = list(args.concept)
    if args.names:
        concepts += [o["phrase"] for o in json.loads(args.names.read_text())]
    if not args.no_hand:
        concepts.append("hand")
    concepts = list(dict.fromkeys(c.strip().lower() for c in concepts if c.strip()))
    if not concepts or concepts == ["hand"]:
        p.error("give object names with --names or --concept")
    out_dir = args.output or Path("out") / args.video.stem
    asyncio.run(track(args.video, concepts, out_dir))


if __name__ == "__main__":
    main()
