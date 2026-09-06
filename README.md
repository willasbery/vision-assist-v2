# vision-assist-v2

Real-time pavement pathfinding for blind and low-vision users, running entirely
on-device on iPhone.

A rework of a university bachelor's thesis project. Version 1 ran a YOLO
segmentation model plus a grid/A* pathfinding pipeline on a laptop, with an
Android app streaming camera frames to it over a websocket. Version 2 runs the
whole thing natively on the phone.

## Layout

| Path | What |
|---|---|
| `ios/VisionAssist` | SwiftUI app |
| `core/VisionAssistCore` | Swift package: the pipeline. No CoreML or UIKit dependencies, so it tests on the Mac. |
| `reference/` | The Python v1 pipeline, kept as the correctness oracle for the Swift port |
| `model/` | Training and Core ML export |
| `fixtures/` | Golden per-stage outputs, used for Swift/Python parity tests |
| `tools/` | Fixture generation, benchmarks |

## Version 1

Archived, unchanged:

- [willasbery/vision-assist](https://github.com/willasbery/vision-assist) — the CV pipeline and model
- [willasbery/vision-assist-app](https://github.com/willasbery/vision-assist-app) — the React Native app and FastAPI backend

## Design

See [docs/superpowers/specs](docs/superpowers/specs).
