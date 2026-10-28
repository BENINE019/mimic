# Mimic: teach once, adapt to new scenes

Film yourself doing a task once on your phone. Mimic works out what you were
trying to do, then does it in scenes it has never seen and exports every
success as robot training data.

Built solo for the Meta Global AI Developer Hackathon (Oct 27 to Nov 4, 2026),
indie developer track, Push the Frontier.

## Status

Day 2. Perception core tested; intent layer in progress. See "Built during the
hackathon" below for what exists today.

## How it works

1. **Name objects** with Muse Spark from a few sampled frames.
2. **Track** every object and the hand through the clip with SAM 3.1.
3. **Map** pixels to table coordinates using one printed ArUco marker.
4. **Find events**: which objects moved, which move together, grasp and release.
5. **Infer the goal** from events and your spoken narration (Muse Spark, Muse Voice Transcribe); ask if ambiguous.
6. **Replay and plan** in MuJoCo: a copy of the demonstration, and a plan from the goal.
7. **Generalize**: randomized scenes, verified with SAM 3.1, exported as LeRobot episodes.

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export MODEL_API_KEY="..."   # never commit this
python track_test.py clips/test.mp4 --concept "red pencil" --output out/test
python name_objects.py clips/test.mp4 --run --compare out/test/report.json
```

## Built during the hackathon

Everything in this repository was written between Oct 27 and Nov 4, 2026.

| Date | Component |
| --- | --- |
| Oct 27 | SAM 3.1 video tracking test (`track_test.py`), Muse Spark auto-naming (`name_objects.py`) |

`sam_api.py` is Meta's helper from the Model API cookbook (MIT licence),
included unchanged with its header.

## Related work

Learning manipulation from human video is an active research area (for
example ORION, OKAMI, Track2Act, MimicPlay). Mimic's difference is setup and
scope: a phone and one printed marker, no task-specific training, narration
as the disambiguating signal, and goal-level generalization in simulation.
It is a tool built on hosted foundation models, not a new method.

## Limitations

Positions on a flat table plane only (no depth, no orientation); scripted
grasp; rigid objects, with garments handled only as whole items; simulated
execution.

## Licence

MIT
