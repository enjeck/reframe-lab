# Evaluation results

These are the video tests I ran while building Reframe Lab. Each folder holds the comparison videos and, where I saved them, the plans and metrics. The numbers come from the same OpenCV evidence the planner uses, so they are diagnostic measures and not human ground truth.

The benchmark clips are a Chromium timeline screencast, the KristenAndSara and FourPeople test sequences, a 4K moving car clip and the Big Buck Bunny trailer. The wildlife and deer clips come from a Creative Commons wildlife compilation.

## Summary

| Test | Baseline | Reframe Lab | Folder |
|---|---:|---:|---|
| Important content in view, wildlife sample | 0.291 (centre crop) | 0.791 (Smart) | [`deploy/aws/sample/wild-animals`](../deploy/aws/sample/wild-animals/) |
| Text kept whole, screen recording | 0.289 (centre crop) | 0.998 | [`text`](text/) |
| Text visible somewhere, screen recording | 0.467 (centre crop) | 1.000 | [`text`](text/) |
| Text kept whole, Big Buck Bunny | | 0.975 | [`text`](text/) |
| Mean camera jerk, car clip | 1.961 (direct follow) | 0.178 | [`camera`](camera/) |
| 95th percentile camera jerk, car clip | 5.136 | 0.288 | [`camera`](camera/) |
| Largest single-frame crop jump, Big Buck Bunny | 0.417 viewports | 0.073 viewports | [`camera`](camera/) |
| Largest single-frame crop jump, car clip | 0.058 viewports | 0.009 viewports | [`camera`](camera/) |
| Split layout kept for two and four people | | 100% of the clip | [`people`](people/) |
| Opening crop spread, controlled repeat test | 0.075 crop widths | 0.003 crop widths | [`repeated-shots/controlled`](repeated-shots/controlled/) |
| Opening crop spread, deer edit | 0.411 crop widths | 0.028 crop widths | [`repeated-shots/deer`](repeated-shots/deer/) |
| Review loop score, framing mark on the wrong subject | 42 | 74 | [`review-loop`](review-loop/) |
| Run time, 45 s screen recording | 17.11 s | 11.99 s | [`speed`](speed/) |

## Text

`chromium_text.mp4` is a 45 second screen recording with dense interface text. `bunny_text.mp4` runs the same rule on the title cards of Big Buck Bunny. Green boxes stay whole and red boxes cross the crop edge.

| Screen recording | Text kept whole | Text visible somewhere | Text cut | Split layout |
|---|---:|---:|---:|---:|
| Centre crop | 0.289 | 0.467 | 0.282 | 0% |
| One moving crop | 0.615 | 0.663 | 0.187 | 0% |
| Smart | 0.998 | 1.000 | 0.107 | 100% |

On Big Buck Bunny, Smart keeps 0.975 of the text whole and 0.987 visible somewhere.

The text metrics count the text regions the detector found. On the screen recording, the lower view of the split still cuts some body text at its left edge, so this layout is not perfect for dense pages.

## Camera

`car_centre_vs_reframe.mp4` puts the centre crop next to the guided camera. The centre crop starts with the car half cut off and loses it by the end. `car_naive_guided.mp4` puts the direct follower on the left and the guided camera on the right. `mask_minimum_jerk.mp4` compares the older online controller with the minimum-jerk planner. `car_tracking_preview.mp4` shows the tracker's box and how much it trusts each match.

| Car clip | Target context kept | Path length | Mean acceleration | Mean jerk | 95th percentile jerk |
|---|---:|---:|---:|---:|---:|
| Direct follow | 1.000 | 1.280 | 0.671 | | |
| Guided | 1.000 | 0.960 | 0.317 | | |
| Online controller | | | | 1.961 | 5.136 |
| Minimum jerk | | | | 0.178 | 0.288 |

Initially, still crops and moving crops used separate render code. This caused jumps when the plan switched between them. After moving them into one viewport model, the largest single-frame jump fell from 0.417 to 0.073 viewports on Big Buck Bunny and from 0.058 to 0.009 on the car clip.

## People

`two_people_comparison.mp4` and `four_people_comparison.mp4` show the source next to narrow crops and the split layout. The split layout kept everyone in view for the whole of both clips and never fell back to a small full-frame inset.

## Repeated shots

`controlled/framing_lock_compare.mp4` is a synthetic edit that returns to the same angle. The opening crop spread fell from 0.075 to 0.003 crop widths.

`deer/` is an edit that cuts back to one deer shot three times, with two other shots in between. I ran the *Follow* edit with the framing lock off and on.

- `edit.mp4` is the input.
- `lock-off.mp4`, `lock-on.mp4` and `compare.mp4` are the outputs.
- `opening-frames.jpg` shows the first frame of each return, with the lock off on top and on below.
- The plans and metrics are saved for both runs.

Without the lock, the three openings sat at 0.21, 0.35 and 0.57 of the frame width. With it, they sat at 0.34, 0.35 and 0.40.

## Manual correction

`manual_correction_compare.mp4` shows the automatic result on the left and the corrected framing on the right. One correction changed 32 analysis samples across matching shots without running any detector again.

## Active speaker

`active_speaker_compare.mp4` is a controlled two-person test with an added audio track, since the public clip has no audio. It produced one speaker switch, and the active panel grew to an average of 56.6% of the split. This only checks that the decision path works.

## Review loop

`auto-fix-before-after.mp4` shows round 0 losing the person and round 2 keeping her. The trace and frames for this run are in [`docs/evidence/auto-fix-example`](../docs/evidence/auto-fix-example/). A framing mark put the crop on a camera instead of the person. Round 0 scored 42 with a 10th-percentile coverage of 0.20. After two rounds, the planner switched to a context and detail layout, coverage rose to 0.74 and the score rose to 74.

## Speed

To speed it up, the first round sampled and resized frames before Python, refined only difficult intervals, ran detectors less often and removed a second encode. `speed_people_compare.mp4` and `speed_car_compare.mp4` are from that round. Total time fell by 25.4% on FourPeople and 19.5% on the 4K car.

The second round added static-scene reuse, persistent models, DNN batching and frame queues. `v012_benchmark.json` holds the full timings, and `v012_ui_compare.mp4` and `v012_car_compare.mp4` show that the outputs did not change.

| Clip and mode | Analysis | Render | Total | Change |
|---|---:|---:|---:|---:|
| Screen recording, first round only | 10.60 s | 5.93 s | 17.11 s | |
| Screen recording, static reuse | 5.73 s | 6.19 s | 12.40 s | 27.5% faster |
| Screen recording, queues | 8.48 s | 5.38 s | 14.38 s | 15.9% faster |
| Screen recording, both rounds | 4.98 s | 6.48 s | 11.99 s | 29.9% faster |
| Car, first round only | 3.46 s | 11.44 s | 15.99 s | |
| Car, static reuse | 3.08 s | 11.95 s | 16.10 s | 0.7% slower |
| Car, queues | 2.95 s | 11.06 s | 15.08 s | 5.7% faster |
| Car, both rounds | 3.26 s | 11.20 s | 15.55 s | 2.7% faster |

Static reuse skipped full analysis on 155 of 193 screen-recording samples and on none of the car samples, since every car frame had real motion. The quality metrics stayed the same.

## Experiments

These larger models stayed behind an experimental flag, since they did not improve the real clips enough.

| Experiment | Files | Result |
|---|---|---|
| Mask instead of a box for the car | `car_box.jpg`, `car_mask.jpg`, `car_mask_compact.mp4` | The mask covered 70.9% of the box area, but the crop came out the same. |
| Dense flow on a scene with few corners | `dense_source.jpg`, `dense_sparse.jpg`, `dense_result.jpg` | The object-to-background motion ratio rose from 0.47 with sparse flow to 26.19 with dense flow. |
| Dense flow on the real car clip | `car_motion_compact.mp4` | Content kept fell from 0.352 with sparse flow to 0.323 with ungated dense flow, and 0.341 with gating. |
| Depth on a controlled scene | `depth_scene.jpg`, `depth_map.jpg` | The planner responded to depth on the test scene. The learned depth model was not validated on real video. |

## Failures

- `failure_center.jpg`, `failure_chase.jpg` and `failure_shrink.jpg` show the three simple answers this project started from. They are a centre crop that cuts the subject, a crop that chases one box and a full frame shrunk onto a background.
- `cursor_failure.jpg` shows a small cursor in a busy interface jumping to a similar shape. The tracker now flags those frames for review.
