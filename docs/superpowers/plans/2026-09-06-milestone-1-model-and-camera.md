# Milestone 1 (Phases A & B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get the v1 segmentation model exported to Core ML and verified, and get a native SwiftUI app running a live camera preview on the iPhone 15 — stopping at the decision gate that determines how the mask gets decoded.

**Architecture:** Two independent phases. Phase A is Python: copy the v1 weights in, export to Core ML, and prove the export still produces the same masks as the PyTorch checkpoint. Phase B is Swift: an XcodeGen-generated app with an `AVCaptureVideoPreviewLayer` feed. Neither depends on the other, so they can be done in either order.

**Tech Stack:** Python 3.13, ultralytics 8.4, coremltools 9.0, torch 2.14 (export only). Swift 5, SwiftUI, AVFoundation, XcodeGen. Target iOS 17+, device iPhone 15 (A16).

---

## Scope

This plan covers Milestone 1 **up to the decision gate, not through it**.

The spec defers one decision: whether to decode YOLO-seg's raw tensors in Swift, or retrain a single-class semantic segmentation model that exports as image-in → mask-out. Phase A Task 4 is what answers that question. Phase C — wiring the model into the app and drawing the mask — cannot be written honestly until we have seen the actual output tensors, so it gets planned once Task 4 reports.

At the end of this plan you will have: a verified Core ML model on disk, a recorded answer to the decoding question, and an app showing live camera on your phone. You will not yet have the mask on screen.

## File structure

| File | Responsibility |
|---|---|
| `model/requirements.txt` | Pinned export toolchain |
| `model/README.md` | Weight provenance and how to re-export |
| `model/weights/*.pt` | v1 checkpoints, copied from the archived repo |
| `model/export_coreml.py` | Checkpoint → `.mlpackage` |
| `model/inspect_coreml.py` | Prints the export's input/output tensor contract |
| `model/tests/test_coreml_parity.py` | Asserts the export matches the checkpoint |
| `fixtures/frames/` | Five test images, reused later for pipeline parity |
| `ios/project.yml` | XcodeGen project definition |
| `ios/Local.xcconfig.example` | Signing team template (real one is gitignored) |
| `ios/VisionAssist/VisionAssistApp.swift` | App entry point |
| `ios/VisionAssist/ContentView.swift` | Root view |
| `ios/VisionAssist/Camera/CameraController.swift` | Session lifecycle and authorisation |
| `ios/VisionAssist/Camera/CameraPreviewView.swift` | SwiftUI wrapper over the preview layer |

---

# Phase A — Model export and verification

## Task 1: Export toolchain

**Files:**
- Create: `model/requirements.txt`
- Create: `model/README.md`
- Modify: `.gitignore`

- [ ] **Step 1: Pin the toolchain**

`model/requirements.txt`:

```
ultralytics==8.4.142
coremltools==9.0
torch==2.14.0
torchvision==0.29.0
numpy==2.5.2
pillow==12.3.0
pytest==9.1.1
```

- [ ] **Step 2: Write the README**

`model/README.md`:

````markdown
# Model

The segmentation model and its Core ML export.

## Setup

Python 3.13 — not 3.14, which torch does not yet support.

```sh
cd model
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Weights

See "Provenance" below. Weights are committed; the training dataset is not.

## Re-exporting

```sh
.venv/bin/python export_coreml.py
.venv/bin/python -m pytest tests/
```

## Provenance

Filled in by Task 2.
````

- [ ] **Step 3: Ignore the venv and dataset**

Append to `.gitignore`:

```
model/.venv/
```

- [ ] **Step 4: Create the venv and install**

Run:
```sh
cd model && python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
```
Expected: installs without resolver errors. Takes a few minutes — torch is large.

- [ ] **Step 5: Verify the install**

Run:
```sh
model/.venv/bin/python -c "import ultralytics, coremltools, torch; print(ultralytics.__version__, coremltools.__version__, torch.__version__)"
```
Expected: `8.4.142 9.0 2.14.0`

- [ ] **Step 6: Commit**

```sh
git add model/requirements.txt model/README.md .gitignore
git commit -m "Add model export toolchain"
```

---

## Task 2: Bring the v1 weights in

The archived v1 repo has nine training runs. Only two matter: `train11` is what the v1 backend actually loaded, and `train16` is the newer but weaker YOLO11 run. Both come across so the comparison is reproducible.

**Files:**
- Create: `model/weights/train11-yolov8n-seg-640.pt`
- Create: `model/weights/train16-yolo11n-seg-240.pt`
- Modify: `model/README.md`

- [ ] **Step 1: Copy the checkpoints**

Run:
```sh
mkdir -p model/weights
cp ../vision-assist/model/runs/segment/train11/weights/best.pt model/weights/train11-yolov8n-seg-640.pt
cp ../vision-assist/model/runs/segment/train16/weights/best.pt model/weights/train16-yolo11n-seg-240.pt
```

- [ ] **Step 2: Verify they load**

Run:
```sh
model/.venv/bin/python -c "
from ultralytics import YOLO
for w in ['model/weights/train11-yolov8n-seg-640.pt', 'model/weights/train16-yolo11n-seg-240.pt']:
    m = YOLO(w)
    print(w, m.task, m.model.yaml.get('nc'), sum(p.numel() for p in m.model.parameters()))
"
```
Expected: both print `segment 1` and a parameter count of roughly 3.2M. If ultralytics 8.4 refuses a checkpoint written by 8.3, that is the one real risk in this task — record the error and stop rather than working around it.

- [ ] **Step 3: Record provenance**

Replace the `## Provenance` section of `model/README.md` with:

````markdown
## Provenance

Both checkpoints come from the archived v1 repo,
[willasbery/vision-assist](https://github.com/willasbery/vision-assist),
under `model/runs/segment/`. Single class: `sidewalk`. Dataset was
[sidewalk-dlu6l](https://universe.roboflow.com/projects-5k1o6/sidewalk-dlu6l/dataset/1)
on Roboflow, 5,296 images, public domain.

| File | Run | Base | imgsz | Epochs | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---|---|---|---|
| `train11-yolov8n-seg-640.pt` | train11 | yolov8n-seg | 640 | 100 | 0.859 | 0.734 |
| `train16-yolo11n-seg-240.pt` | train16 | yolo11n-seg | 240 | 150 | 0.810 | 0.689 |

`train11` is what the v1 backend loaded in production, and is the default for
export despite being the older architecture — it is the stronger model.
````

- [ ] **Step 4: Commit**

```sh
git add model/weights model/README.md
git commit -m "Add v1 segmentation checkpoints"
```

---

## Task 3: Export to Core ML

**Files:**
- Create: `model/export_coreml.py`
- Create: `model/weights/train11-yolov8n-seg-640.mlpackage/` (generated)

- [ ] **Step 1: Write the export script**

`model/export_coreml.py`:

```python
"""Export a trained YOLO segmentation checkpoint to Core ML.

Run from the model/ directory:
    .venv/bin/python export_coreml.py
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "train11-yolov8n-seg-640.pt"


def export(weights: Path, imgsz: int) -> Path:
    """Export `weights` to a .mlpackage sitting alongside it."""
    model = YOLO(weights)
    exported = model.export(format="coreml", imgsz=imgsz, half=True)
    return Path(exported)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    output = export(args.weights, args.imgsz)
    print(f"Exported to {output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the export**

Run:
```sh
cd model && .venv/bin/python export_coreml.py
```
Expected: prints `Exported to .../weights/train11-yolov8n-seg-640.mlpackage`. Ultralytics prints its own conversion log first; coremltools warnings about unsupported ops are normal as long as it completes.

If it fails or warns that `half` is unsupported, re-run with `half=False` in
`export_coreml.py` and note it — the export will be roughly double the size but
is otherwise equivalent, and the parity test in Task 4 will confirm that.

- [ ] **Step 3: Confirm the package exists and check its size**

Run:
```sh
du -sh model/weights/train11-yolov8n-seg-640.mlpackage
```
Expected: roughly 6-8MB at `half=True`. If it is over 50MB something has gone wrong with the fp16 conversion — stop and investigate rather than committing it.

- [ ] **Step 4: Commit**

The `.mlpackage` is committed deliberately: the app needs to build without anyone re-running the export.

```sh
git add model/export_coreml.py model/weights/train11-yolov8n-seg-640.mlpackage
git commit -m "Add Core ML export script and exported model"
```

---

## Task 4: Verify the export and answer the decoding question

**This is the decision gate.** Two separate questions: does the export still produce the right masks, and what shape is its output contract.

**Files:**
- Create: `fixtures/frames/` (five images)
- Create: `model/inspect_coreml.py`
- Create: `model/tests/test_coreml_parity.py`
- Create: `docs/superpowers/notes/2026-09-06-coreml-export-findings.md`

- [ ] **Step 1: Copy five test frames in**

These come from the v1 test split — images the model was never trained on. They get reused later as pipeline parity fixtures, which is why they live in `fixtures/` rather than under `model/`.

Run:
```sh
mkdir -p fixtures/frames
for f in $(ls ../vision-assist/model/test/images | sort | head -5); do
  cp "../vision-assist/model/test/images/$f" fixtures/frames/
done
ls fixtures/frames | wc -l
```
Expected: `5`

- [ ] **Step 2: Write the parity test**

Ultralytics can run inference through an exported `.mlpackage` directly, doing the decoding itself. That lets us check the export numerically without having written a Swift decoder yet.

`model/tests/test_coreml_parity.py`:

```python
"""The Core ML export must produce the same masks as the PyTorch checkpoint.

fp16 conversion moves values slightly, so this asserts a high IoU rather than
equality.
"""
from pathlib import Path

import numpy as np
import pytest
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[2]
FRAMES = sorted((REPO / "fixtures" / "frames").glob("*.jpg"))
CHECKPOINT = REPO / "model" / "weights" / "train11-yolov8n-seg-640.pt"
EXPORT = REPO / "model" / "weights" / "train11-yolov8n-seg-640.mlpackage"

MIN_IOU = 0.95


def largest_mask(model: YOLO, frame: Path) -> np.ndarray | None:
    """The pipeline only ever uses the biggest mask, so that is what we compare."""
    result = model.predict(frame, conf=0.5, verbose=False)[0]
    if result.masks is None or len(result.masks.data) == 0:
        return None
    masks = result.masks.data.cpu().numpy() > 0.5
    return masks[masks.sum(axis=(1, 2)).argmax()]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


@pytest.fixture(scope="module")
def models() -> tuple[YOLO, YOLO]:
    return YOLO(CHECKPOINT), YOLO(EXPORT)


def test_fixture_frames_exist():
    assert len(FRAMES) == 5


@pytest.mark.parametrize("frame", FRAMES, ids=lambda p: p.name[:20])
def test_export_matches_checkpoint(models, frame):
    checkpoint, export = models
    expected, actual = largest_mask(checkpoint, frame), largest_mask(export, frame)

    if expected is None and actual is None:
        pytest.skip("neither model detected a pavement in this frame")
    assert expected is not None and actual is not None, "one model detected nothing"

    assert iou(expected, actual) >= MIN_IOU
```

- [ ] **Step 3: Run the parity test**

Run:
```sh
cd model && .venv/bin/python -m pytest tests/ -v
```
Expected: 6 passed (or some skipped if a frame has no pavement). **Record the actual IoU values** — if any frame lands between 0.80 and 0.95 the export is usable but lossy, and that is worth knowing before building on it. If IoUs are near zero, the export is broken and everything downstream is invalid: stop here.

- [ ] **Step 4: Write the inspection script**

`model/inspect_coreml.py`:

```python
"""Print the input/output tensor contract of an exported Core ML model.

This is what determines how much decoding the Swift side has to do.

    .venv/bin/python inspect_coreml.py
"""
import argparse
from pathlib import Path

import coremltools as ct

DEFAULT_MODEL = Path(__file__).parent / "weights" / "train11-yolov8n-seg-640.mlpackage"


def describe(model_path: Path) -> None:
    spec = ct.models.MLModel(str(model_path)).get_spec()

    print(f"{model_path.name}\n")
    print(f"spec version: {spec.specificationVersion}")

    for label, features in (("INPUTS", spec.description.input),
                            ("OUTPUTS", spec.description.output)):
        print(f"\n{label}")
        for feature in features:
            kind = feature.type.WhichOneof("Type")
            print(f"  {feature.name}  ({kind})")
            if kind == "multiArrayType":
                array = feature.type.multiArrayType
                print(f"      shape: {list(array.shape)}")
                print(f"      dtype: {array.dataType}")
            elif kind == "imageType":
                image = feature.type.imageType
                print(f"      {image.width}x{image.height}, colorspace {image.colorSpace}")

    if spec.description.metadata.userDefined:
        print("\nMETADATA")
        for key, value in spec.description.metadata.userDefined.items():
            print(f"  {key}: {value[:200]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    describe(parser.parse_args().model)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run it and read the output**

Run:
```sh
cd model && .venv/bin/python inspect_coreml.py
```
Expected: an image input around 640x640, and two outputs — a detection tensor roughly `[1, 37, 8400]` and a mask-prototype tensor roughly `[1, 32, 160, 160]`. The exact names and shapes are what we need.

- [ ] **Step 6: Record the findings and make the call**

Create `docs/superpowers/notes/2026-09-06-coreml-export-findings.md` with the real numbers — parity IoUs, the tensor contract from Step 5, and the `.mlpackage` size. Then answer, in writing:

> To turn these outputs into one binary mask in Swift, what has to happen?

Decode the detection tensor, run NMS, take the winning box's 32 mask coefficients, matrix-multiply against the prototypes, sigmoid, crop to the box, threshold, and upsample. Estimate the Swift work honestly.

**The decision:** if that is a contained amount of Accelerate/Metal code, keep YOLO and Phase C writes the decoder. If it looks like several hundred lines of fiddly plumbing, the alternative is retraining a single-class semantic segmentation model that exports image-in → mask-out with no decoding at all — the pipeline only ever uses the largest mask, so nothing of value is lost.

**Do not start writing a decoder as part of this task.** Write the finding down and stop.

- [ ] **Step 7: Commit**

```sh
git add fixtures/frames model/inspect_coreml.py model/tests docs/superpowers/notes
git commit -m "Verify Core ML export against checkpoint and record tensor contract"
```

---

# Phase B — iOS app with live camera

Independent of Phase A. Nothing here loads a model.

## Task 5: Generate the Xcode project

The `.xcodeproj` is generated from `project.yml` and gitignored, so the repo has no giant unmergeable project file.

**Files:**
- Create: `ios/project.yml`
- Create: `ios/Local.xcconfig.example`
- Create: `ios/VisionAssist/VisionAssistApp.swift`
- Create: `ios/VisionAssist/ContentView.swift`
- Modify: `.gitignore`

- [ ] **Step 1: Write the project definition**

`ios/project.yml`:

```yaml
name: VisionAssist
options:
  bundleIdPrefix: com.willasbery
  deploymentTarget:
    iOS: "17.0"
  createIntermediateGroups: true

configs:
  Debug: debug
  Release: release

targets:
  VisionAssist:
    type: application
    platform: iOS
    sources:
      - path: VisionAssist
    configFiles:
      Debug: Local.xcconfig
      Release: Local.xcconfig
    settings:
      base:
        PRODUCT_BUNDLE_IDENTIFIER: com.willasbery.visionassist
        MARKETING_VERSION: "2.0"
        CURRENT_PROJECT_VERSION: "1"
        TARGETED_DEVICE_FAMILY: "1"
        SWIFT_VERSION: "5.0"
        GENERATE_INFOPLIST_FILE: YES
        INFOPLIST_KEY_UILaunchScreen_Generation: YES
        INFOPLIST_KEY_UISupportedInterfaceOrientations: UIInterfaceOrientationPortrait
        INFOPLIST_KEY_NSCameraUsageDescription: Vision Assist uses the camera to find the pavement ahead of you.
```

- [ ] **Step 2: Write the signing template**

Signing to a real device needs your Apple developer team ID, which does not belong in a public repo.

`ios/Local.xcconfig.example`:

```
// Copy to Local.xcconfig and fill in your team ID.
// Find it in Xcode: Settings > Accounts > your Apple ID > Manage Certificates,
// or run:  security find-identity -v -p codesigning
DEVELOPMENT_TEAM = ABCDE12345
CODE_SIGN_STYLE = Automatic
```

- [ ] **Step 3: Ignore generated and local files**

Append to `.gitignore`:

```
ios/VisionAssist.xcodeproj/
ios/Local.xcconfig
```

- [ ] **Step 4: Write the app entry point**

`ios/VisionAssist/VisionAssistApp.swift`:

```swift
import SwiftUI

@main
struct VisionAssistApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
```

- [ ] **Step 5: Write a placeholder root view**

`ios/VisionAssist/ContentView.swift`:

```swift
import SwiftUI

struct ContentView: View {
    var body: some View {
        Text("Vision Assist")
            .font(.largeTitle)
    }
}

#Preview {
    ContentView()
}
```

- [ ] **Step 6: Create your local config and generate**

Run:
```sh
cd ios && cp Local.xcconfig.example Local.xcconfig
```
Then edit `ios/Local.xcconfig` and replace `ABCDE12345` with your real team ID. Then:
```sh
cd ios && xcodegen generate
```
Expected: `Created project at .../ios/VisionAssist.xcodeproj`

- [ ] **Step 7: Build for the simulator**

Run:
```sh
xcodebuild -project ios/VisionAssist.xcodeproj -scheme VisionAssist \
  -destination 'generic/platform=iOS Simulator' build 2>&1 | tail -5
```
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 8: Commit**

```sh
git add ios/project.yml ios/Local.xcconfig.example ios/VisionAssist .gitignore
git commit -m "Add SwiftUI app skeleton generated by XcodeGen"
```

---

## Task 6: Live camera preview

`AVCaptureVideoPreviewLayer` renders the feed directly, so no frames pass through SwiftUI. Later milestones draw the mask as a separate overlay on top rather than compositing into the video, which keeps rendering cheap.

**Files:**
- Create: `ios/VisionAssist/Camera/CameraController.swift`
- Create: `ios/VisionAssist/Camera/CameraPreviewView.swift`
- Modify: `ios/VisionAssist/ContentView.swift`

- [ ] **Step 1: Write the session controller**

`ios/VisionAssist/Camera/CameraController.swift`:

```swift
import AVFoundation

/// Owns the capture session and its lifecycle.
///
/// The session is configured and started off the main thread — `startRunning()`
/// blocks, and doing it on the main thread stutters the UI on launch.
@MainActor
final class CameraController: ObservableObject {

    enum State: Equatable {
        case idle
        case running
        case denied
        case failed(String)
    }

    @Published private(set) var state: State = .idle

    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "com.willasbery.visionassist.session")
    private var isConfigured = false

    func start() async {
        guard await isAuthorised() else {
            state = .denied
            return
        }

        if !isConfigured {
            do {
                try configure()
                isConfigured = true
            } catch {
                state = .failed(String(describing: error))
                return
            }
        }

        let session = self.session
        queue.async { 
            if !session.isRunning { session.startRunning() }
        }
        state = .running
    }

    func stop() {
        let session = self.session
        queue.async {
            if session.isRunning { session.stopRunning() }
        }
        state = .idle
    }

    private func isAuthorised() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return true
        case .notDetermined:
            return await AVCaptureDevice.requestAccess(for: .video)
        default:
            return false
        }
    }

    private enum ConfigurationError: Error {
        case noCamera
        case cannotAddInput
    }

    private func configure() throws {
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        session.sessionPreset = .hd1280x720

        guard let camera = AVCaptureDevice.default(
            .builtInWideAngleCamera, for: .video, position: .back
        ) else {
            throw ConfigurationError.noCamera
        }

        let input = try AVCaptureDeviceInput(device: camera)
        guard session.canAddInput(input) else {
            throw ConfigurationError.cannotAddInput
        }
        session.addInput(input)
    }
}
```

- [ ] **Step 2: Write the SwiftUI wrapper**

`ios/VisionAssist/Camera/CameraPreviewView.swift`:

```swift
import AVFoundation
import SwiftUI
import UIKit

/// A `UIView` backed directly by an `AVCaptureVideoPreviewLayer`, so the layer
/// resizes with the view for free.
final class PreviewUIView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

    var previewLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }
}

struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ view: PreviewUIView, context: Context) {
        if view.previewLayer.session !== session {
            view.previewLayer.session = session
        }
    }
}
```

- [ ] **Step 3: Wire it into the root view**

Replace `ios/VisionAssist/ContentView.swift` entirely:

```swift
import SwiftUI

struct ContentView: View {
    @StateObject private var camera = CameraController()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            switch camera.state {
            case .running:
                CameraPreviewView(session: camera.session)
                    .ignoresSafeArea()
            case .denied:
                message("Camera access is off. Turn it on in Settings to use Vision Assist.")
            case .failed(let reason):
                message("The camera could not start.\n\(reason)")
            case .idle:
                ProgressView()
            }
        }
        .task { await camera.start() }
        .onDisappear { camera.stop() }
    }

    private func message(_ text: String) -> some View {
        Text(text)
            .font(.headline)
            .foregroundStyle(.white)
            .multilineTextAlignment(.center)
            .padding()
    }
}
```

- [ ] **Step 4: Regenerate the project**

New files were added, so XcodeGen has to pick them up.

Run:
```sh
cd ios && xcodegen generate
```
Expected: `Created project at .../ios/VisionAssist.xcodeproj`

- [ ] **Step 5: Build for the simulator**

Run:
```sh
xcodebuild -project ios/VisionAssist.xcodeproj -scheme VisionAssist \
  -destination 'generic/platform=iOS Simulator' build 2>&1 | tail -5
```
Expected: `** BUILD SUCCEEDED **`. The simulator has no camera, so this only proves it compiles.

- [ ] **Step 6: Run it on the phone**

Plug in the iPhone 15, unlock it, and open `ios/VisionAssist.xcodeproj` in Xcode. Select the device and hit Run. The first run needs "Trust This Computer" on the phone, and possibly Settings > General > VPN & Device Management to trust the developer certificate.

Expected: a permission prompt on first launch, then a live camera feed filling the screen in portrait.

- [ ] **Step 7: Check the failure paths by hand**

There is no meaningful unit test for camera permissions, so verify manually:

1. Deny the permission on first launch → the "Camera access is off" message shows, no crash.
2. Grant it in Settings, relaunch → the feed appears.
3. Background the app and return → the feed resumes.

- [ ] **Step 8: Commit**

```sh
git add ios/VisionAssist
git commit -m "Add live camera preview"
```

---

## Done when

- `model/.venv/bin/python -m pytest model/tests/ -v` passes, with the export matching the checkpoint at IoU ≥ 0.95
- `docs/superpowers/notes/2026-09-06-coreml-export-findings.md` records the real tensor contract and a written decision on YOLO-decode vs semantic segmentation
- The app runs on the iPhone 15 showing a live camera feed
- Permission-denied and camera-failure paths both show a message rather than crashing

## Next

Phase C — Core ML inference on the video frames, the mask overlay, and the ms/frame readout — gets planned once Task 4 has reported. That plan depends on the decision made there.
