# Auto-fix example: a bad framing mark

This folder holds a real run of the local web app on a stock clip of a creator at her desk (Pexels video 9032610, 1920×1080, 6.5 seconds, 9:16 output).

A framing mark was placed at 0:03 with its crop centred on the camera (`center_x` 0.75). The mark scope was "This shot". That pushed the crop away from the person.

| File | What it shows |
|---|---|
| `src.jpg` | Source frame at 0:04 |
| `r0.jpg` | Round 0 output at 0:04. The crop shows only the camera. |
| `r2.jpg` | Round 2 output at 0:04. A context and detail layout keeps the person in view. |
| `framing_corrections.json` | The framing mark sent by the web app |
| `auto_fix_trace.json` | The full decision record written by the pipeline |

## What the trace records

| Round | Score | P10 coverage | Context and detail | Issues |
|---|---:|---:|---:|---|
| 0 | 42.2 | 0.199 | 0.00 | low content coverage, low visual confidence |
| 1 | 42.2 | 0.199 | 0.00 | low content coverage, low visual confidence |
| 2 | 74.0 | 0.740 | 1.00 | low visual confidence |

The content coverage gate failed in round 0. Its action is to make unsafe crops switch to a split or context and detail layout sooner. The controller raised `fallback_coverage_threshold` from 0.46 to 0.505 in round 1, which was not enough. It raised it to 0.55 in round 2, and the planner switched layout. The controller kept round 2.

Low visual confidence was not repaired. It stays in `remaining_issues` so a person can review it. The video was analysed once. Rounds 1 and 2 only reran the planner on the cached OpenCV evidence.

Reproduce it with the web app: open the clip, press 3, click on the camera at 0:03, and press Reframe. The same files appear under `web-runs/<job>/output/`.
