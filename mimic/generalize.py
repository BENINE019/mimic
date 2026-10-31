#!/usr/bin/env python3
"""Stage 8: teach once, adapt to new scenes.

Takes the goal inferred for a clip, randomizes the scene (new positions, an
extra object of the same kind, a distractor), and runs two robots in each:

  goal   plans from the inferred goal in the new scene (should succeed)
  copy   replays the demonstrated pick and place positions (should fail)

Each run is checked geometrically against the goal, and every successful
goal run is saved as an episode (scene, plan, joint trajectory, final state)
for export.

Outputs in out/<clip>/generalize/:
  summary.json                success rates and per-scene verdicts
  scene_NN/goal/  copy/       result.json (+ episode.mp4 for rendered scenes)

Usage (from the repo root):
  python -m mimic.generalize clips/c01.mp4 --scenes 10 --render 2
  python -m mimic.generalize clips/c01.mp4 --scene0        # demo scene only (no shuffle), both robots
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from . import planner
from .sim import Scene, fit_to_reach, load_clip, plan_from_events, scene_objects

MIN_GAP_CM = 3.0
MARGIN_CM = 6.0


def load_goal(out_dir: Path) -> tuple[dict, dict]:
    data = json.loads((out_dir / "intent.json").read_text())
    return data["candidates"][0], data


def shuffle(objects: list[dict], rng: np.random.Generator, extra: bool, distractor: bool) -> list[dict]:
    objs = copy.deepcopy(objects)
    pts = np.array([o["xy"] for o in objs])
    lo, hi = pts.min(axis=0) - MARGIN_CM, pts.max(axis=0) + MARGIN_CM
    span = np.minimum(hi - lo, 40.0)                     # keep the layout within the arm's reach
    mid = (lo + hi) / 2
    lo, hi = mid - span / 2, mid + span / 2
    if extra:
        src = next((o for o in objs if not o["container"] and o["role"] in ("moved",)), None) or objs[0]
        dup = copy.deepcopy(src)
        dup["key"] = f"{dup['concept']}#9"
        dup["attached_to"] = None
        dup["role"] = "new"
        objs.append(dup)
    if distractor:
        objs.append({"key": "yellow block#9", "concept": "yellow block", "role": "new", "attached_to": None,
                     "xy": [0.0, 0.0], "w_cm": 4.0, "d_cm": 4.0, "h_cm": 3.0, "container": False,
                     "rgb": (0.95, 0.85, 0.2)})
    placed: list[tuple[np.ndarray, float]] = []
    for o in objs:
        if o.get("attached_to"):
            continue
        r = max(o["w_cm"], o["d_cm"]) / 2
        for _ in range(300):
            p = rng.uniform(lo, hi)
            if all(np.linalg.norm(p - q) > r + rq + MIN_GAP_CM for q, rq in placed):
                break
        o["xy"] = [round(float(p[0]), 1), round(float(p[1]), 1)]
        placed.append((p, r))
    # attached objects ride with their parent, keeping the original offset
    by_key = {o["key"]: o for o in objs}
    for o in objs:
        if o.get("attached_to") and o["attached_to"] in by_key:
            src = next(x for x in objects if x["key"] == o["key"])
            par = next(x for x in objects if x["key"] == o["attached_to"])
            off = np.subtract(src["xy"], par["xy"])
            o["xy"] = [round(float(v), 1) for v in np.add(by_key[o["attached_to"]]["xy"], off)]
    return objs


def run(objects: list[dict], steps: list[dict], goal: dict, render: bool, out_dir: Path | None) -> dict:
    scene = Scene(objects, render=render)
    res = scene.execute(steps, out_dir)
    ok, why = planner.check(goal, objects, res["final_xy_cm"])
    res["goal_satisfied"] = bool(ok)
    res["why"] = why
    if out_dir is not None:
        res_light = {k: v for k, v in res.items()}
        (out_dir / "result.json").write_text(json.dumps(res_light, indent=2) + "\n")
    return res


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 8: generalize the inferred goal to new scenes")
    p.add_argument("video", type=Path)
    p.add_argument("--scenes", type=int, default=10)
    p.add_argument("--render", type=int, default=2, help="render videos for the first N scenes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scene0", action="store_true", help="only the original scene, no shuffle")
    p.add_argument("--goal", type=Path, help="JSON file with a goal to use instead of intent.json's top candidate")
    p.add_argument("--output", type=Path, help="default: out/<clip name>")
    args = p.parse_args()

    out_dir = args.output or Path("out") / args.video.stem
    tt, ev = load_clip(out_dir)
    base_objects = scene_objects(tt, ev)
    if args.goal:
        goal, intent = json.loads(args.goal.read_text()), {}
    else:
        goal, intent = load_goal(out_dir)
    print(f"Goal: {goal.get('goal') or goal['type']}  scope={goal.get('scope')}  ({intent.get('task_sentence', '')})")

    gen_dir = out_dir / "generalize"
    gen_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []
    n_scenes = 1 if args.scene0 else args.scenes + 1
    for i in range(n_scenes):
        if i == 0:
            objects = copy.deepcopy(base_objects)
            variant = "original"
        else:
            extra = i % 2 == 0 and goal["type"] in ("put_all", "arrange")
            distractor = i % 3 == 2
            for _ in range(30):
                objects = shuffle(base_objects, rng, extra, distractor)
                already, _why = planner.check(goal, objects, {o["key"]: o["xy"] for o in objects})
                if not already:
                    break
            variant = "shuffled" + (" +extra" if extra else "") + (" +distractor" if distractor else "")
        scene_dir = gen_dir / f"scene_{i:02d}"
        scene_dir.mkdir(exist_ok=True)
        (scene_dir / "scene.json").write_text(json.dumps({"variant": variant, "goal": goal, "objects": objects}, indent=2) + "\n")
        render = i < args.render

        steps_copy = plan_from_events(ev, objects)
        k = fit_to_reach(objects, [steps_copy])   # copy positions scale with the scene; the goal plan is made after
        steps_goal, note = planner.plan(goal, objects)
        rg = run(objects, steps_goal, goal, render, scene_dir / "goal")
        rc = run(objects, steps_copy, goal, render, scene_dir / "copy")
        if k < 1.0:
            variant += f" (scaled x{k:.2f})"

        rows.append({"scene": i, "variant": variant, "goal_plan": note, "goal_steps": len(steps_goal),
                     "goal_ok": rg["goal_satisfied"], "goal_why": rg["why"],
                     "copy_ok": rc["goal_satisfied"], "copy_why": rc["why"],
                     "copy_misses": sum("missed" in s["status"] for s in rc["steps"])})
        print(f"scene {i:02d} {variant:<28} goal: {'OK ' if rg['goal_satisfied'] else 'FAIL'} ({rg['why']})"
              f"   copy: {'OK ' if rc['goal_satisfied'] else 'FAIL'} ({rc['why']}; {rows[-1]['copy_misses']} missed grasps)")

    shuffled = [r for r in rows if r["scene"] > 0]
    summary = {
        "clip": args.video.stem, "goal": goal, "task_sentence": intent.get("task_sentence", ""),
        "scenes": rows,
        "goal_success_original": rows[0]["goal_ok"],
        "copy_success_original": rows[0]["copy_ok"],
        "goal_success_shuffled": (sum(r["goal_ok"] for r in shuffled) / len(shuffled)) if shuffled else None,
        "copy_success_shuffled": (sum(r["copy_ok"] for r in shuffled) / len(shuffled)) if shuffled else None,
        "episodes_exported": sum(1 for r in rows if r["goal_ok"]),
    }
    (gen_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if shuffled:
        print(f"\nShuffled scenes: goal plan {summary['goal_success_shuffled']:.0%} success, "
              f"copy baseline {summary['copy_success_shuffled']:.0%}. "
              f"{summary['episodes_exported']} episode(s) from one demonstration.")
    print(f"Wrote {gen_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
