# vision-assist-v2 — Design

**Date:** 2026-09-06
**Status:** Approved

## Context

Version 1 (bachelor's thesis) segments the pavement ahead with a YOLO model,
lays a 20px grid over the mask, scores each cell by how far off-centre it sits
within its row/column run, finds contour peaks as candidate destinations, runs
A* to each, and reduces the result to one of three spoken instructions:
`move_left`, `move_right`, `continue_forward`.

It never ran on the phone. An Android app streamed base64 JPEG frames over a
websocket to a laptop running FastAPI, which ran the pipeline and returned the
instruction.

### Why it never ran on-device

The model is not the bottleneck. From v1's own profiling, per frame:

| Stage | Avg (s) |
|---|---|
| yolo_prediction | 0.04 |
| grid_extraction | 0.04 |
| protrusion_detection | 0.04 |
| graph_creation | 0.03 |
| path_finding | 0.05 |
| penalty_calculation | 0.02 |
| blurry_frame_check | 0.02 |
| **Total** | **~0.24 (≈4fps)** |

Inference is ~17% of the frame budget. The other 83% is ~1,300 lines of
Python/NumPy/OpenCV that allocates a Pydantic object per grid cell. Core ML
will run the model on the Neural Engine in single-digit milliseconds; the
post-processing has to be rewritten in a compiled language. That rewrite is the
project.

There was also never an iOS build: no `ios/` directory, and the frame→base64
camera plugin was Android-only Java.

## Goals

- The full pipeline runs on-device on an iPhone 15 (A16, no LiDAR)
- The model's output is visible on screen in real time
- The Swift port is provably equivalent to the Python reference, stage by stage
- Native SwiftUI, so VoiceOver and Dynamic Type come for free

## Non-goals

- Depth sensing. The target device has no LiDAR.
- Keeping the React Native app or the FastAPI backend as products. Python
  survives only as a test oracle.
- Retraining the model, unless the Core ML spike (below) forces the question.

## Architecture

```
vision-assist-v2/
├── ios/VisionAssist/       SwiftUI app: camera, Core ML, overlay rendering
├── core/VisionAssistCore/  Swift package: the pipeline
├── reference/              Python v1, cleaned — correctness oracle
├── model/                  train.py, export_coreml.py, weights/
├── fixtures/               golden per-stage outputs
└── tools/                  fixture generation, benchmarks
```

`VisionAssistCore` takes a binary mask and returns an instruction plus the
intermediate state needed to draw the debug overlay. It depends on nothing
beyond Foundation and Accelerate — no Core ML, no UIKit — so the parity tests
run under `swift test` on the Mac in seconds without a simulator.

The app owns everything platform-specific: capture session, Core ML request,
mask decoding, and rendering.

### Data representation

The two structural changes from v1, both of which shape the module boundaries:

- **Flat buffers, not objects.** Grid occupancy as `[UInt8]` and penalties as
  `[Float]`, indexed `row * cols + col`. v1 allocates a Pydantic `Grid` per
  cell plus a `(x, y) -> Grid` dictionary, every frame.
- **No materialised adjacency graph.** v1's `_create_graph()` builds a
  `defaultdict` of neighbours each frame, but 4-neighbour adjacency is already
  fully determined by the occupancy buffer.

### Verification

`tools/` runs the Python reference over a fixed set of frames and dumps each
stage's output to JSON in `fixtures/`. Swift tests assert the port reproduces
them stage by stage. Parity is established per stage as it is ported, not at
the end.

Bugs in the reference must be fixed before it can serve as an oracle,
otherwise the port would faithfully reproduce wrong behaviour.

**Fixed.** *Angle-cache unit mismatch* (`PathFinder._angle_between_grids`).
Cache misses appended degrees while the value stored was radians. Since the
cache is deliberately never cleared between frames, the A* smoothness penalty
collapsed to roughly 1/57th of its intended value for any vector pair seen
before. Covered by `test_angle_is_stable_across_repeated_calls`.

**Fixed, but smaller than expected.** The `came_from` chain was re-walked for
every neighbour of every expanded node, which is identical work for all four.
Lifting it out of the loop is safe — any ancestor of the current node is
already closed, so it cannot be rewritten mid-loop — but it turned out to be
worth well under 1% on a representative 40x25 obstructed field (34.1ms to
33.9ms per call).

**Open, needs a decision.** Profiling shows `_angle_between_grids` is ~70% of
`find_path`. It rescans the path from the start on every neighbour expansion
and returns the `max` angle change over the whole history, so cost grows with
path length and a single sharp turn penalises every subsequent expansion for
the rest of the route. Restricting the window to the tail of the path would
make it O(1) per expansion, but it changes the cost function and therefore the
routes chosen. That is a design decision for the port, not a refactor, and is
deferred to the pathfinding milestone.

## Milestone 1

**Camera → Core ML → sidewalk mask on screen. Nothing else.**

1. Repo and skeleton
2. Export `train11/best.pt` to Core ML; inspect the output tensors
3. SwiftUI app: capture session plus a Core ML request on the video buffer
4. Draw the mask as an overlay, with a live ms/frame readout

**Done when:** it runs on the iPhone 15, the mask tracks the pavement in real
time, and inference cost is visible on screen.

Explicitly not in scope: pathfinding, audio, settings, accessibility. Those are
later milestones.

### Open decision: model choice

Step 2 is a spike, not a commitment. YOLO-seg exports to Core ML as raw tensors
— boxes, mask coefficients, and 32 prototype masks — so decoding in Swift needs
NMS, a coefficient×proto matmul, sigmoid, box-crop and upsample. That is a few
hundred lines of plumbing unrelated to the project's actual contribution.

The pipeline only ever uses the single largest mask. A single-class semantic
segmentation model would export as image-in → mask-out with no decoding at all.

The decision is deferred until the exported model has been inspected. If the
decoding is manageable, YOLO stays.

## Later milestones

Sketched only, to be designed when reached:

2. Port grid extraction and penalty calculation, with parity tests
3. Port protrusion detection
4. Port A* and path analysis — the pipeline runs end-to-end on-device
5. Debug overlay: grid, penalties, candidate paths
6. Accessibility and audio: VoiceOver, Dynamic Type, spoken instructions
