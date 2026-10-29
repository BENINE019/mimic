#!/usr/bin/env python3
"""Stage 4: map pixel tracks onto the table plane, in centimetres.

Finds the printed ArUco marker (DICT_4X4_50, id 0) in frames sampled across
the clip, takes the median corners (robust to a hand passing over it), and
builds a homography from image pixels to the table plane. Every object's base
point is then converted to (x, y) in cm.

Table frame: origin at the marker centre, x along the marker's top edge
(left to right as printed), y along its left edge (top to bottom as printed).

Outputs in out/<clip>/:
  table_tracks.json   tracks.json plus "xy" (cm) per frame and a calibration block
  topdown.png         bird's-eye plot of every trajectory, for a quick sanity check

Usage (from the repo root):
  python -m mimic.table_map clips/c01.mp4 --marker-cm 9.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DICT = cv2.aruco.DICT_4X4_50
MARKER_ID = 0
END_WINDOW = 10  # frames used for robust start / end positions

PALETTE = [
    (80, 200, 60), (60, 120, 255), (255, 120, 60), (200, 60, 200),
    (40, 220, 220), (220, 220, 40), (120, 80, 255), (255, 80, 160),
]


def detect_marker(video: Path, samples: int = 30) -> tuple[np.ndarray, int]:
    """Median marker corners (4x2, pixel) over sampled frames, and how many frames saw it."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT), params)

    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    found = []
    for idx in np.linspace(0, max(0, total - 1), samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            continue
        for c, i in zip(corners, ids.ravel()):
            if i == MARKER_ID:
                found.append(c.reshape(4, 2))
    cap.release()
    if len(found) < 3:
        raise RuntimeError(
            f"Marker id {MARKER_ID} seen in only {len(found)} of {samples} sampled frames. "
            "Check it is flat, uncovered, well lit and in frame."
        )
    return np.median(np.stack(found), axis=0), len(found)


def homography(corners_px: np.ndarray, marker_cm: float) -> tuple[np.ndarray, float]:
    h = marker_cm / 2
    world = np.array([[-h, -h], [h, -h], [h, h], [-h, h]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(corners_px.astype(np.float32), world)
    back = cv2.perspectiveTransform(corners_px.reshape(1, 4, 2).astype(np.float64), H)[0]
    rms = float(np.sqrt(np.mean(np.sum((back - world) ** 2, axis=1))))
    return H, rms


def to_table(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)


def covers_marker(obj: dict, corners_px: np.ndarray) -> bool:
    """True if the object's typical box contains the marker centre (the printed sheet itself)."""
    boxes = np.array([f["bbox"] for f in obj["frames"]], dtype=float)
    x0, y0, x1, y1 = np.median(boxes, axis=0)
    cx, cy = corners_px.mean(axis=0)
    return bool(x0 <= cx <= x1 and y0 <= cy <= y1)


def robust_end(xy: np.ndarray, start: bool) -> np.ndarray:
    win = xy[:END_WINDOW] if start else xy[-END_WINDOW:]
    return np.median(win, axis=0)


def draw_topdown(objects: list[dict], marker_cm: float, path: Path, size: int = 900) -> None:
    pts = [np.array([f["xy"] for f in o["frames"]]) for o in objects if o["frames"]]
    allp = np.vstack(pts + [np.array([[-marker_cm, -marker_cm], [marker_cm, marker_cm]])])
    lo, hi = allp.min(axis=0) - 5, allp.max(axis=0) + 5
    scale = (size - 80) / max(hi - lo)
    img = np.full((size, size, 3), 255, np.uint8)

    def px(p):
        q = (np.asarray(p) - lo) * scale + 40
        return int(q[0]), int(q[1])

    h = marker_cm / 2
    cv2.rectangle(img, px([-h, -h]), px([h, h]), (0, 0, 0), 2)
    cv2.putText(img, "marker", px([-h, -h - 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    placed: list[tuple[int, int, int, int]] = []

    def free_spot(x: int, y: int, w: int, h: int = 14) -> tuple[int, int]:
        while any(not (x + w < a or x > c or y < b or y - h > d) for a, b, c, d in placed):
            y += h + 2
        placed.append((x, y - h, x + w, y))
        return x, y

    for i, o in enumerate(objects):
        xy = np.array([f["xy"] for f in o["frames"]])
        colour = (150, 150, 150) if o["is_hand"] else PALETTE[i % len(PALETTE)]
        cv2.polylines(img, [np.array([px(p) for p in xy], np.int32)], False, colour, 1, cv2.LINE_AA)
        if o["is_hand"]:
            continue
        s, e = px(o["start_xy"]), px(o["end_xy"])
        cv2.circle(img, s, 6, colour, 2)
        cv2.rectangle(img, (e[0] - 6, e[1] - 6), (e[0] + 6, e[1] + 6), colour, -1)
        label = f"{o['key']} ({o['moved_cm']:.0f} cm)"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        lx, ly = free_spot(e[0] + 9, e[1] + 4, tw)
        if ly != e[1] + 4:
            cv2.line(img, (e[0] + 6, e[1]), (lx - 2, ly - 5), colour, 1, cv2.LINE_AA)
        cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

    # 10 cm scale bar
    x0, y0 = 40, size - 20
    cv2.line(img, (x0, y0), (x0 + int(10 * scale), y0), (0, 0, 0), 2)
    cv2.putText(img, "10 cm", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "circle = start, square = end, grey = hand", (size - 360, size - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 4: pixel tracks to table-plane centimetres")
    p.add_argument("video", type=Path)
    p.add_argument("--marker-cm", type=float, required=True,
                   help="measured side length of the black marker square, in cm")
    p.add_argument("--point", choices=["base", "centroid"], default="base",
                   help="which mask point touches the table (default base)")
    p.add_argument("--output", type=Path, help="default: out/<clip name>")
    args = p.parse_args()

    out_dir = args.output or Path("out") / args.video.stem
    tracks_path = out_dir / "tracks.json"
    if not tracks_path.exists():
        p.error(f"{tracks_path} not found; run mimic.tracking first")
    data = json.loads(tracks_path.read_text())

    corners, seen = detect_marker(args.video)
    H, rms = homography(corners, args.marker_cm)
    print(f"Marker found in {seen} sampled frames; corner fit error {rms:.3f} cm")

    rows = []
    for o in data["objects"]:
        if not o["frames"]:
            continue
        o["is_marker_sheet"] = (not o["is_hand"]) and covers_marker(o, corners)
        if o["is_marker_sheet"]:
            print(f"Ignoring {o['key']}: it covers the marker (the printed sheet)")
            continue
        xy = to_table(H, np.array([f[args.point] for f in o["frames"]]))
        for f, q in zip(o["frames"], xy):
            f["xy"] = [round(float(q[0]), 2), round(float(q[1]), 2)]
        o["start_xy"] = robust_end(xy, start=True).round(2).tolist()
        o["end_xy"] = robust_end(xy, start=False).round(2).tolist()
        o["moved_cm"] = round(float(np.linalg.norm(np.subtract(o["end_xy"], o["start_xy"]))), 1)
        rows.append(o)

    data["table"] = {
        "marker_cm": args.marker_cm,
        "marker_id": MARKER_ID,
        "dictionary": "DICT_4X4_50",
        "point": args.point,
        "H_px_to_cm": H.tolist(),
        "corners_px": corners.tolist(),
        "frames_with_marker": seen,
        "corner_fit_rms_cm": round(rms, 4),
    }
    (out_dir / "table_tracks.json").write_text(json.dumps(data) + "\n")
    draw_topdown(rows, args.marker_cm, out_dir / "topdown.png")

    print(f"\n{'object':<26}{'start (cm)':>18}{'end (cm)':>18}{'moved':>9}")
    for o in rows:
        if o["is_hand"]:
            continue
        s, e = o["start_xy"], o["end_xy"]
        print(f"{o['key']:<26}{s[0]:>9.1f}{s[1]:>9.1f}{e[0]:>9.1f}{e[1]:>9.1f}{o['moved_cm']:>8.1f}")
    print(f"\nWrote {out_dir / 'table_tracks.json'} and {out_dir / 'topdown.png'}")
    print("Sanity check: moved objects should show roughly the distance you moved them; "
          "untouched ones under about 2 cm.")


if __name__ == "__main__":
    main()
