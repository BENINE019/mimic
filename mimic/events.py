#!/usr/bin/env python3
"""Stage 5: turn table-plane tracks into events.

For every object it finds the periods where the object rests on the table
(no hand touching it), and treats a change of rest position as one move:
grasp frame, release frame, pick position, place position. On top of that:

- frames where the hand overlaps the object are ignored, which removes the
  jumps caused by a partly covered mask;
- an object released inside a container's outline is snapped to the
  container ("in": cup), which also fixes the height error of objects
  sticking out of a cup;
- objects whose moves always happen together, with the same displacement,
  are grouped (the keychain becomes attached to the wallet);
- objects that never move are scene context.

Outputs in out/<clip>/:
  events.json    objects with roles and moves, plus the ordered event list
  timeline.png   one row per object: hand contact, grasp and release

Usage (from the repo root):
  python -m mimic.events clips/c01.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CONTAINER_WORDS = ("cup", "mug", "container", "basket", "box", "bowl", "jar", "bin", "tray", "pot", "bag")

CONTACT_FRAC = 0.15    # hand box covering this share of the object box counts as contact
MOVE_CM = 3.0          # rest positions further apart than this are a move
MIN_REST = 5           # frames needed for a rest interval
SMOOTH = 5             # median window over rest frames
GROUP_FRAMES = 20      # grouped moves must start and end within this many frames
GROUP_CM = 4.0         # and have displacements this similar


def box_overlap_frac(a, b) -> float:
    """Share of box a covered by box b (boxes as x0, y0, x1, y1)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    return ix * iy / area


def is_container(concept: str) -> bool:
    return any(w in concept.split() or concept.endswith(w) for w in CONTAINER_WORDS)


def hand_boxes_by_frame(objects: list[dict]) -> dict[int, list]:
    out: dict[int, list] = {}
    for o in objects:
        if o.get("is_hand"):
            for f in o["frames"]:
                out.setdefault(f["f"], []).append(f["bbox"])
    return out


def rest_intervals(frames: list[dict], hands: dict[int, list]) -> list[dict]:
    """Consecutive runs of hand-free frames, each with a robust table position."""
    free = [f for f in frames
            if all(box_overlap_frac(f["bbox"], h) < CONTACT_FRAC for h in hands.get(f["f"], []))]
    runs, cur = [], []
    for f in free:
        if cur and f["f"] - cur[-1]["f"] > 2:  # a gap (contact or not visible) ends a run
            runs.append(cur)
            cur = []
        cur.append(f)
    if cur:
        runs.append(cur)

    out = []
    for r in runs:
        if len(r) < MIN_REST:
            continue
        xy = np.array([f["xy"] for f in r])
        k = min(SMOOTH, len(xy))
        sm = np.array([np.median(xy[max(0, i - k // 2): i + k // 2 + 1], axis=0) for i in range(len(xy))])
        out.append({
            "f0": r[0]["f"], "f1": r[-1]["f"],
            "xy": np.median(sm, axis=0),
            "bbox": np.median(np.array([f["bbox"] for f in r]), axis=0),
        })
    return out


def merge_rests(rests: list[dict]) -> list[dict]:
    """Join neighbouring rests at the same place (a hand brushing past is not a move)."""
    merged: list[dict] = []
    for r in rests:
        if merged and np.linalg.norm(r["xy"] - merged[-1]["xy"]) < MOVE_CM:
            m = merged[-1]
            m["f1"] = r["f1"]
            m["bbox"] = r["bbox"]
        else:
            merged.append(dict(r))
    return merged


def container_at(bbox, containers: list[dict]):
    """The container whose image box holds the centre of bbox, if any."""
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    best = None
    for c in containers:
        x0, y0, x1, y1 = c["bbox"]
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            if best is None or (x1 - x0) * (y1 - y0) < (best["bbox"][2] - best["bbox"][0]) * (best["bbox"][3] - best["bbox"][1]):
                best = c
    return best


def find_moves(obj: dict, hands: dict, containers: list[dict]) -> list[dict]:
    rests = merge_rests(rest_intervals(obj["frames"], hands))
    moves = []
    for a, b in zip(rests, rests[1:]):
        if np.linalg.norm(b["xy"] - a["xy"]) < MOVE_CM:
            continue
        moves.append({"grasp_f": int(a["f1"]) + 1, "release_f": int(b["f0"]),
                      "pick_xy": a["xy"], "place_xy": b["xy"], "place_bbox": b["bbox"], "in": None})

    # Object put into a container and never seen resting again (hidden inside it).
    last_seen = obj["frames"][-1]
    if rests and last_seen["f"] > rests[-1]["f1"] + 2:
        c = container_at(last_seen["bbox"], [c for c in containers if c["key"] != obj["key"]])
        if c is not None:
            moves.append({"grasp_f": int(rests[-1]["f1"]) + 1, "release_f": int(last_seen["f"]),
                          "pick_xy": rests[-1]["xy"], "place_xy": np.array(c["xy"]),
                          "place_bbox": np.array(last_seen["bbox"]), "in": c["key"]})

    for m in moves:
        if m["in"] is None:
            c = container_at(m["place_bbox"], [c for c in containers if c["key"] != obj["key"]])
            if c is not None:
                m["in"] = c["key"]
                m["place_xy"] = np.array(c["xy"])
    return moves


def same_motion(ma: list[dict], mb: list[dict]) -> bool:
    if len(ma) != len(mb) or not ma:
        return False
    for a, b in zip(ma, mb):
        if abs(a["grasp_f"] - b["grasp_f"]) > GROUP_FRAMES or abs(a["release_f"] - b["release_f"]) > GROUP_FRAMES:
            return False
        da = a["place_xy"] - a["pick_xy"]
        db = b["place_xy"] - b["pick_xy"]
        if np.linalg.norm(da - db) > GROUP_CM:
            return False
    return True


def analyse(data: dict) -> dict:
    objects = [o for o in data["objects"] if o["frames"] and not o.get("is_marker_sheet")]
    hands = hand_boxes_by_frame(objects)
    things = [o for o in objects if not o.get("is_hand")]

    containers = []
    for o in things:
        if is_container(o["concept"]):
            containers.append({"key": o["key"], "xy": o["start_xy"],
                               "bbox": np.median(np.array([f["bbox"] for f in o["frames"][:30]]), axis=0)})

    result = {}
    for o in things:
        moves = find_moves(o, hands, containers)
        area = float(np.median([f.get("area", 0) for f in o["frames"]]))
        result[o["key"]] = {"key": o["key"], "concept": o["concept"], "moves": moves, "area": area,
                            "container": is_container(o["concept"]), "attached_to": None,
                            "start_xy": o["start_xy"], "end_xy": o["end_xy"]}

    # Rigid groups: the larger object is the parent.
    moved = [r for r in result.values() if r["moves"]]
    for i, a in enumerate(moved):
        for b in moved[i + 1:]:
            if a["attached_to"] or b["attached_to"]:
                continue
            if same_motion(a["moves"], b["moves"]):
                child, parent = (a, b) if a["area"] < b["area"] else (b, a)
                child["attached_to"] = parent["key"]

    for r in result.values():
        if r["attached_to"]:
            r["role"] = "attached"
        elif r["moves"]:
            r["role"] = "moved"
        elif r["container"]:
            r["role"] = "container"
        else:
            r["role"] = "static"

    events = []
    for r in result.values():
        if r["role"] != "moved":
            continue
        carried = [c["key"] for c in result.values() if c["attached_to"] == r["key"]]
        for m in r["moves"]:
            events.append({"object": r["key"], "carried": carried,
                           "grasp_f": m["grasp_f"], "release_f": m["release_f"],
                           "grasp_s": round(m["grasp_f"] / data["fps"], 2),
                           "release_s": round(m["release_f"] / data["fps"], 2),
                           "pick_xy": np.round(m["pick_xy"], 1).tolist(),
                           "place_xy": np.round(m["place_xy"], 1).tolist(),
                           "in": m["in"]})
    events.sort(key=lambda e: e["grasp_f"])

    objs_out = []
    for r in result.values():
        objs_out.append({k: r[k] for k in ("key", "concept", "role", "attached_to", "start_xy", "end_xy")}
                        | {"moves": len(r["moves"])})
    return {"video": data["video"], "fps": data["fps"], "frames": data["frames"],
            "objects": objs_out, "events": events}


def draw_timeline(data: dict, ev: dict, path: Path) -> None:
    rows = [o for o in ev["objects"]]
    n, total = len(rows), max(1, ev["frames"])
    W, left, rowh = 1100, 230, 34
    H = 60 + rowh * n
    img = np.full((H, W, 3), 255, np.uint8)
    sx = (W - left - 30) / total
    hands = hand_boxes_by_frame([o for o in data["objects"] if not o.get("is_marker_sheet")])
    frames_by_key = {o["key"]: o["frames"] for o in data["objects"]}
    colours = {"moved": (60, 120, 255), "attached": (200, 60, 200), "container": (40, 160, 40), "static": (150, 150, 150)}

    for i, o in enumerate(rows):
        y = 40 + i * rowh
        c = colours[o["role"]]
        label = o["key"] + (f" -> {o['attached_to']}" if o["attached_to"] else "")
        cv2.putText(img, label[:30], (10, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
        cv2.line(img, (left, y + 14), (W - 30, y + 14), (225, 225, 225), 1)
        for f in frames_by_key.get(o["key"], []):  # visible frames, hand contact in orange
            x = int(left + f["f"] * sx)
            touched = any(box_overlap_frac(f["bbox"], h) >= CONTACT_FRAC for h in hands.get(f["f"], []))
            cv2.line(img, (x, y + 10), (x, y + 18), (0, 150, 255) if touched else (200, 200, 200), 1)
        for e in ev["events"]:
            if e["object"] == o["key"] or o["key"] in e["carried"]:
                g, r = int(left + e["grasp_f"] * sx), int(left + e["release_f"] * sx)
                cv2.rectangle(img, (g, y + 6), (r, y + 22), c, 2)
                cv2.putText(img, "G", (g - 4, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1, cv2.LINE_AA)
                cv2.putText(img, "R", (r - 4, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1, cv2.LINE_AA)
    cv2.putText(img, "grey = visible, orange = hand contact, box = grasp (G) to release (R)",
                (left, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 5: events from table-plane tracks")
    p.add_argument("video", type=Path)
    p.add_argument("--output", type=Path, help="default: out/<clip name>")
    args = p.parse_args()

    out_dir = args.output or Path("out") / args.video.stem
    src = out_dir / "table_tracks.json"
    if not src.exists():
        p.error(f"{src} not found; run mimic.table_map first")
    data = json.loads(src.read_text())
    ev = analyse(data)
    (out_dir / "events.json").write_text(json.dumps(ev, indent=2) + "\n")
    draw_timeline(data, ev, out_dir / "timeline.png")

    print(f"{'object':<28}{'role':<11}{'moves':>6}  attached to")
    for o in ev["objects"]:
        print(f"{o['key']:<28}{o['role']:<11}{o['moves']:>6}  {o['attached_to'] or ''}")
    print("\nEvents:")
    for e in ev["events"]:
        extra = f" + {', '.join(e['carried'])}" if e["carried"] else ""
        where = f"into {e['in']}" if e["in"] else f"to {e['place_xy']}"
        print(f"  {e['grasp_s']:>5.1f}s to {e['release_s']:>5.1f}s  {e['object']}{extra}  from {e['pick_xy']} {where}")
    print(f"\nWrote {out_dir / 'events.json'} and {out_dir / 'timeline.png'}")


if __name__ == "__main__":
    main()
