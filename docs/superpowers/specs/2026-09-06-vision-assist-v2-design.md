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

Two bugs in the reference must be fixed before it can serve as an oracle,
otherwise the port would faithfully reproduce wrong behaviour:

1. **Angle-cache unit mismatch** (`PathFinder.py:118-121`). Cache misses append
   degrees; the value stored is radians. Since the cache is deliberately never
   cleared between frames, the A* smoothness penalty collapses to near-zero for
   any vector pair seen before.
2. **O(n²) A* reconstruction** (`PathFinder.py:170-175`). The entire `came_from`
   chain is re-walked for every neighbour of every expanded node — identical
   work for all four neighbours. Tracking direction-of-arrival in a parallel
   buffer makes the angle penalty O(1) per expansion.

Both are fixed when that stage is reached, not up front.

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
