#!/usr/bin/env python3
"""Planner: from a goal to pick-and-place steps in ANY scene, and the matching success check.

The five goal types (see intent.py):
  put_all(category, container)   move(object, relation, anchor)   arrange(category, pattern)
  stack(top, base)               swap(a, b)

A scene is a list of objects as produced by sim.scene_objects: key, concept,
xy (cm), w_cm, d_cm, h_cm, container, role. The planner only uses geometry and
names, never the demonstration's positions, which is what lets it work after
the objects are shuffled.
"""

from __future__ import annotations

import re

import numpy as np

NEAR_CM = 4.0          # "in the container" / "on the base" / "swapped" tolerance
NEXT_TO_CM = 8.0       # gap between object and anchor for "next to", "left of", ...
LINE_TOL_CM = 3.0      # arrange: allowed deviation from the common line
GENERIC = {"object", "objects", "item", "items", "thing", "things", "everything", "all"}


def words(s: str) -> set[str]:
    return set(re.findall(r"[a-z]+", s.lower()))


def plain(s: str) -> str:
    """'black pencil#0' -> 'black pencil'."""
    return s.split("#")[0].strip().lower()


def matches_category(obj: dict, category: str) -> bool:
    cat = words(category)
    if not cat or cat & GENERIC:
        return not obj["container"]
    cw = words(obj["concept"])
    # category "pencil" matches "black pencil" but not "pen"; allow simple plurals
    return any(c in cw or c.rstrip("s") in cw or c + "s" in cw for c in cat)


def find_object(objects: list[dict], name: str) -> dict | None:
    """Resolve a name from the goal ('black wallet#0', 'wallet', 'the cup') to a scene object."""
    if not name:
        return None
    for o in objects:
        if o["key"] == name:
            return o
    target = words(name) - {"the", "a", "an", "my"}
    best, score = None, 0
    for o in objects:
        s = len(target & words(o["key"]))
        if s > score:
            best, score = o, s
    return best


def find_container(objects: list[dict], name: str) -> dict | None:
    c = find_object(objects, name)
    if c is not None and c["container"]:
        return c
    conts = [o for o in objects if o["container"]]
    return conts[0] if len(conts) == 1 else c


def half_extent(o: dict) -> float:
    return max(o["w_cm"], o["d_cm"]) / 2


def relation_target(obj: dict, relation: str, anchor: dict | None, objects: list[dict]) -> list[float]:
    r = relation.lower()
    oxy = np.array(obj["xy"], float)
    others = [o for o in objects if o["key"] != obj["key"]]
    if "leftmost" in r or ("left" in r and anchor is None and "most" in r):
        return [min(o["xy"][0] for o in others) - NEXT_TO_CM - half_extent(obj), float(np.median([o["xy"][1] for o in others]))]
    if "rightmost" in r:
        return [max(o["xy"][0] for o in others) + NEXT_TO_CM + half_extent(obj), float(np.median([o["xy"][1] for o in others]))]
    if anchor is None:
        step = NEXT_TO_CM + 2 * half_extent(obj)
        if "left" in r:
            return [oxy[0] - step * 1.5, oxy[1]]
        if "right" in r:
            return [oxy[0] + step * 1.5, oxy[1]]
        if "front" in r or "forward" in r or "down" in r:
            return [oxy[0], oxy[1] - step * 1.5]
        if "behind" in r or "back" in r or "up" in r:
            return [oxy[0], oxy[1] + step * 1.5]
        return oxy.tolist()  # "to", "over here": no better information
    axy = np.array(anchor["xy"], float)
    gap = half_extent(anchor) + half_extent(obj) + 3.0
    if "left" in r:
        return [axy[0] - gap, axy[1]]
    if "right" in r:
        return [axy[0] + gap, axy[1]]
    if "front" in r:
        return [axy[0], axy[1] - gap]
    if "behind" in r or "back" in r:
        return [axy[0], axy[1] + gap]
    # next to / beside / near: the side the object is already on, so the arm does not cross the anchor
    d = oxy - axy
    d = d / (np.linalg.norm(d) + 1e-9)
    return (axy + d * gap).tolist()


def fits_in(obj: dict, cont: dict) -> bool:
    """A long thin object stands in a cup; anything else must be narrower than the opening."""
    opening = min(cont["w_cm"], cont["d_cm"]) * 0.8
    short = min(obj["w_cm"], obj["d_cm"])
    long = max(obj["w_cm"], obj["d_cm"])
    if long > 2.5 * short:                 # pencil-like: stands upright, only the cross-section matters
        return short < opening
    return long < opening


def plan(goal: dict, objects: list[dict]) -> tuple[list[dict], str]:
    """Return (steps, note). Steps: {object, place_xy, into, onto, carried}."""
    t = goal["type"]
    steps: list[dict] = []
    carried_by = {o["attached_to"]: [] for o in objects if o.get("attached_to")}
    for o in objects:
        if o.get("attached_to"):
            carried_by[o["attached_to"]].append(o["key"])

    def step(obj, place_xy, into=None, onto=None):
        steps.append({"object": obj["key"], "place_xy": [round(float(place_xy[0]), 2), round(float(place_xy[1]), 2)],
                      "into": into, "onto": onto, "carried": carried_by.get(obj["key"], [])})

    if t == "put_all":
        cont = find_container(objects, goal.get("container", ""))
        if cont is None:
            return [], "no container in scene"
        items = [o for o in objects if not o["container"] and not o.get("attached_to")
                 and matches_category(o, goal.get("category", ""))]
        if goal.get("order"):
            rank = {k: i for i, k in enumerate(goal["order"])}
            items.sort(key=lambda o: rank.get(o["key"], len(rank)))
        skipped = []
        for o in items:
            if np.linalg.norm(np.subtract(o["xy"], cont["xy"])) <= NEAR_CM:
                continue
            if not fits_in(o, cont):
                skipped.append(o["key"])
                continue
            step(o, cont["xy"], into=cont["key"])
        note = f"{len(steps)} item(s) into {cont['key']}"
        if skipped:
            note += f"; does not fit: {', '.join(skipped)}"
        return steps, note

    if t == "move":
        obj = find_object(objects, goal.get("object", ""))
        if obj is None:
            return [], "object not found"
        anchor = find_object(objects, goal.get("anchor", "")) if goal.get("anchor") else None
        if anchor is not None and anchor["key"] == obj["key"]:
            anchor = None
        target = relation_target(obj, goal.get("relation", ""), anchor, objects)
        step(obj, target)
        return steps, f"{obj['key']} {goal.get('relation', 'to')} {anchor['key'] if anchor else target}"

    if t == "arrange":
        items = [o for o in objects if not o["container"] and not o.get("attached_to")
                 and matches_category(o, goal.get("category", ""))]
        if len(items) < 2:
            return [], "fewer than two items to arrange"
        items.sort(key=lambda o: o["xy"][0])
        y = float(np.median([o["xy"][1] for o in items]))
        pitch = max(half_extent(o) for o in items) * 2 + 3.0
        x0 = float(np.mean([o["xy"][0] for o in items])) - pitch * (len(items) - 1) / 2
        for i, o in enumerate(items):
            step(o, [x0 + i * pitch, y])
        return steps, f"{len(items)} items in a row"

    if t == "stack":
        top = find_object(objects, goal.get("top", ""))
        base = find_object(objects, goal.get("base", ""))
        if top is None or base is None or top["key"] == base["key"]:
            return [], "top or base not found"
        step(top, base["xy"], onto=base["key"])
        return steps, f"{top['key']} onto {base['key']}"

    if t == "swap":
        a = find_object(objects, goal.get("a", ""))
        b = find_object(objects, goal.get("b", ""))
        if a is None or b is None or a["key"] == b["key"]:
            return [], "a or b not found"
        axy, bxy = list(a["xy"]), list(b["xy"])
        temp = [axy[0], axy[1] - (half_extent(a) * 2 + 12.0)]
        step(a, temp)
        step(b, axy)
        step(a, bxy)
        return steps, f"{a['key']} <-> {b['key']}"

    return [], f"unknown goal type {t}"


def check(goal: dict, objects: list[dict], final_xy: dict[str, list[float]]) -> tuple[bool, str]:
    """Does the final scene satisfy the goal? Uses only geometry."""
    t = goal["type"]
    pos = {k: np.array(v, float) for k, v in final_xy.items()}
    by_key = {o["key"]: o for o in objects}

    if t == "put_all":
        cont = find_container(objects, goal.get("container", ""))
        if cont is None:
            return False, "no container"
        items = [o for o in objects if not o["container"] and not o.get("attached_to")
                 and matches_category(o, goal.get("category", ""))]
        bad = [o["key"] for o in items if np.linalg.norm(pos[o["key"]] - pos[cont["key"]]) > NEAR_CM]
        return (not bad), ("all in" if not bad else f"not in container: {', '.join(bad)}")

    if t == "move":
        obj = find_object(objects, goal.get("object", ""))
        if obj is None:
            return False, "object not found"
        anchor = find_object(objects, goal.get("anchor", "")) if goal.get("anchor") else None
        if anchor is not None and anchor["key"] == obj["key"]:
            anchor = None
        r = goal.get("relation", "").lower()
        p = pos[obj["key"]]
        others = [k for k in pos if k != obj["key"]]
        moved_others = [k for k in others if np.linalg.norm(pos[k] - np.array(by_key[k]["xy"])) > NEAR_CM
                        and k not in (by_key[obj["key"]].get("carried") or [])
                        and by_key[k].get("attached_to") != obj["key"]]
        if moved_others:
            return False, f"disturbed: {', '.join(moved_others)}"
        if "leftmost" in r:
            ok = all(p[0] < pos[k][0] for k in others)
            return ok, "leftmost" if ok else "not leftmost"
        if "rightmost" in r:
            ok = all(p[0] > pos[k][0] for k in others)
            return ok, "rightmost" if ok else "not rightmost"
        if anchor is None:
            start = np.array(obj["xy"])
            d = p - start
            if "left" in r:
                return d[0] < -5, "moved left" if d[0] < -5 else "did not move left"
            if "right" in r:
                return d[0] > 5, "moved right" if d[0] > 5 else "did not move right"
            if "front" in r or "down" in r:
                return d[1] < -5, "moved toward front"
            if "behind" in r or "back" in r or "up" in r:
                return d[1] > 5, "moved back"
            return np.linalg.norm(d) > 5, "moved" if np.linalg.norm(d) > 5 else "did not move"
        a = pos[anchor["key"]]
        gap = half_extent(anchor) + half_extent(obj)
        dist = np.linalg.norm(p - a)
        if "left" in r:
            ok = p[0] < a[0] - gap * 0.5 and abs(p[1] - a[1]) < gap + NEXT_TO_CM
        elif "right" in r:
            ok = p[0] > a[0] + gap * 0.5 and abs(p[1] - a[1]) < gap + NEXT_TO_CM
        elif "front" in r:
            ok = p[1] < a[1] - gap * 0.5 and abs(p[0] - a[0]) < gap + NEXT_TO_CM
        elif "behind" in r or "back" in r:
            ok = p[1] > a[1] + gap * 0.5 and abs(p[0] - a[0]) < gap + NEXT_TO_CM
        else:
            ok = gap * 0.5 < dist < gap + NEXT_TO_CM + NEAR_CM
        return ok, f"{dist:.1f} cm from {anchor['key']}"

    if t == "arrange":
        items = [o for o in objects if not o["container"] and not o.get("attached_to")
                 and matches_category(o, goal.get("category", ""))]
        if len(items) < 2:
            return False, "fewer than two items"
        pts = np.array([pos[o["key"]] for o in items])
        # fit a line, measure the spread perpendicular to it
        c = pts.mean(axis=0)
        u, s, vt = np.linalg.svd(pts - c)
        perp = np.abs((pts - c) @ vt[1])
        along = np.sort((pts - c) @ vt[0])
        gaps = np.diff(along)
        ok = perp.max() < LINE_TOL_CM and gaps.min() > 2.0
        return ok, f"line deviation {perp.max():.1f} cm, min gap {gaps.min():.1f} cm"

    if t == "stack":
        top = find_object(objects, goal.get("top", ""))
        base = find_object(objects, goal.get("base", ""))
        if top is None or base is None:
            return False, "top or base not found"
        d = np.linalg.norm(pos[top["key"]] - pos[base["key"]])
        return d < NEAR_CM, f"{d:.1f} cm off the base"

    if t == "swap":
        a = find_object(objects, goal.get("a", ""))
        b = find_object(objects, goal.get("b", ""))
        if a is None or b is None:
            return False, "a or b not found"
        da = np.linalg.norm(pos[a["key"]] - np.array(b["xy"]))
        db = np.linalg.norm(pos[b["key"]] - np.array(a["xy"]))
        ok = da < NEAR_CM and db < NEAR_CM
        return ok, f"a {da:.1f} cm from b's spot, b {db:.1f} cm from a's spot"

    return False, f"unknown goal type {t}"
