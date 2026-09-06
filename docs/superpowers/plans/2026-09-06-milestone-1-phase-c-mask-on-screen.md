# Milestone 1, Phase C — Mask on screen

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the Core ML model on live camera frames and draw the sidewalk mask over the preview, with a frame-time readout — completing Milestone 1.

**Architecture:** The mask decoder is pure arithmetic, so it lives in a Swift package and is tested on the Mac against a golden fixture captured from a real Core ML inference. The app supplies frames, runs Core ML, hands the raw tensors to the decoder, and draws the result as a layer above the existing preview.

**Tech Stack:** Swift 5, Swift Package Manager, Accelerate (`vDSP`), Vision, Core ML, AVFoundation, SwiftUI.

---

## What Phase A established

These are measured facts, not assumptions. Everything below depends on them.

`train11-yolov8n-seg-640.mlpackage`, exported with `nms=True`:

```
INPUT   image      640x640 RGB

OUTPUT  [1, 300, 38]       NMS-resolved detections, zero-padded
                           col 0..3  box x1, y1, x2, y2 in 640-pixel space
                           col 4     confidence
                           col 5     class index (always 0, "sidewalk")
                           col 6..37 32 mask coefficients
        [1, 32, 160, 160]  mask prototypes
```

Output tensor names are converter-generated and change between exports. **Bind by shape, never by name.**

Mask assembly, verified in Python at IoU 0.9941 against ultralytics' own output:

```
logits = coefficients (32) · prototypes (32 x 25600)  ->  reshape 160 x 160
mask   = sigmoid(logits) > 0.5, restricted to the box scaled by 1/4
```

**Preprocessing is a stretch, not a letterbox.** The Roboflow dataset was
exported with "Resize to 640x640 (Stretch)", so the model was trained on
stretched images. Vision's `.scaleFill` matches that, and it makes coordinate
mapping a plain linear scale with no padding offsets to unwind.

## File structure

| File | Responsibility |
|---|---|
| `tools/dump_decoder_fixture.py` | Captures one real inference to disk as a golden fixture |
| `fixtures/decoder/` | That fixture — raw tensors plus the expected mask |
| `core/VisionAssistCore/Package.swift` | Swift package definition |
| `.../Sources/Masking/Mask.swift` | The `Mask` value type. Model-agnostic. |
| `.../Sources/YOLODecoding/YOLODetection.swift` | A parsed detection row |
| `.../Sources/YOLODecoding/YOLOMaskDecoder.swift` | Tensors → `Mask` |
| `.../Tests/YOLODecodingTests/` | Decoder tests against the fixture |
| `ios/VisionAssist/Camera/FrameStream.swift` | Video output delegate with backpressure |
| `ios/VisionAssist/Segmentation/PavementSegmenter.swift` | The protocol — the model seam |
| `ios/VisionAssist/Segmentation/YOLOSegmenter.swift` | Core ML + Vision implementation |
| `ios/VisionAssist/Overlay/MaskOverlayView.swift` | Draws the mask above the preview |

---

## Task 1: Golden decoder fixture

The Swift decoder must be testable on the Mac with no camera, no device, and no
Core ML. That means capturing one real inference to disk.

**Files:**
- Create: `tools/dump_decoder_fixture.py`
- Create: `fixtures/decoder/manifest.json`, `detections.bin`, `prototypes.bin`, `expected_mask.bin`

- [ ] **Step 1: Write the dump tool**

`tools/dump_decoder_fixture.py`:

```python
"""Capture one real Core ML inference as a golden fixture for the Swift decoder.

Writes raw little-endian float32 tensors plus the expected mask, so the Swift
tests can verify the decoder without Core ML or a device.

    ../model/.venv/bin/python tools/dump_decoder_fixture.py
"""
import json
from pathlib import Path

import coremltools as ct
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "model" / "weights" / "train11-yolov8n-seg-640.mlpackage"
OUT = REPO / "fixtures" / "decoder"

CONF_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5


def main() -> None:
    frame = sorted((REPO / "fixtures" / "frames").glob("image_*.jpg"))[1]
    image = Image.open(frame).convert("RGB").resize((640, 640))

    outputs = ct.models.MLModel(str(MODEL)).predict({"image": image})
    detections = next(v for v in outputs.values() if v.ndim == 3 and v.shape[-1] == 38)[0]
    prototypes = next(v for v in outputs.values() if v.ndim == 4)[0]

    rows = detections[detections[:, 4] > CONF_THRESHOLD]
    assert len(rows) >= 1, "fixture frame produced no detection above threshold"

    # Largest box wins, matching what the pipeline does with the mask.
    areas = (rows[:, 2] - rows[:, 0]) * (rows[:, 3] - rows[:, 1])
    row = rows[areas.argmax()]
    x1, y1, x2, y2, conf = (float(row[i]) for i in range(5))

    logits = (row[6:] @ prototypes.reshape(32, -1)).reshape(160, 160)
    inside = np.zeros_like(logits, dtype=bool)
    inside[int(y1 / 4):int(np.ceil(y2 / 4)), int(x1 / 4):int(np.ceil(x2 / 4))] = True
    mask = ((1.0 / (1.0 + np.exp(-logits))) > MASK_THRESHOLD) & inside

    OUT.mkdir(parents=True, exist_ok=True)
    detections.astype("<f4").tofile(OUT / "detections.bin")
    prototypes.astype("<f4").tofile(OUT / "prototypes.bin")
    mask.astype(np.uint8).tofile(OUT / "expected_mask.bin")

    (OUT / "manifest.json").write_text(json.dumps({
        "source_frame": frame.name,
        "model": MODEL.name,
        "detections": {"file": "detections.bin", "rows": int(detections.shape[0]),
                       "columns": int(detections.shape[1])},
        "prototypes": {"file": "prototypes.bin", "count": int(prototypes.shape[0]),
                       "size": int(prototypes.shape[1])},
        "expected_mask": {"file": "expected_mask.bin", "size": 160,
                          "filled": int(mask.sum())},
        "confidence_threshold": CONF_THRESHOLD,
        "mask_threshold": MASK_THRESHOLD,
        "expected_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "expected_confidence": conf,
    }, indent=2) + "\n")

    print(f"wrote fixture from {frame.name}: box=({x1:.1f},{y1:.1f})-({x2:.1f},{y2:.1f}) "
          f"conf={conf:.4f} filled={int(mask.sum())}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run:
```sh
model/.venv/bin/python tools/dump_decoder_fixture.py
```
Expected: `wrote fixture from image_0142...: box=(123.2,351.8)-(639.0,639.5) conf=0.9556 filled=6218`

- [ ] **Step 3: Check the sizes**

Run:
```sh
ls -l fixtures/decoder/
```
Expected: `prototypes.bin` about 3.3MB, `detections.bin` about 45KB, `expected_mask.bin` 25,600 bytes exactly.

- [ ] **Step 4: Commit**

```sh
git add tools/dump_decoder_fixture.py fixtures/decoder
git commit -m "Add golden decoder fixture from a real Core ML inference"
```

---

## Task 2: The `Mask` type

**Files:**
- Create: `core/VisionAssistCore/Package.swift`
- Create: `core/VisionAssistCore/Sources/Masking/Mask.swift`
- Create: `core/VisionAssistCore/Tests/MaskingTests/MaskTests.swift`

- [ ] **Step 1: Write the package definition**

Two targets, because they have genuinely different lifetimes: `Masking` is
model-agnostic and will be what the pathfinding pipeline consumes, while
`YOLODecoding` is specific to the current model and is the part that gets
replaced if we ever switch.

`core/VisionAssistCore/Package.swift`:

```swift
// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "VisionAssistCore",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "Masking", targets: ["Masking"]),
        .library(name: "YOLODecoding", targets: ["YOLODecoding"]),
    ],
    targets: [
        .target(name: "Masking"),
        .target(name: "YOLODecoding", dependencies: ["Masking"]),
        .testTarget(name: "MaskingTests", dependencies: ["Masking"]),
        .testTarget(name: "YOLODecodingTests", dependencies: ["YOLODecoding"]),
    ]
)
```

- [ ] **Step 2: Write the `Mask` type**

`core/VisionAssistCore/Sources/Masking/Mask.swift`:

```swift
import Foundation

/// A binary mask in its own coordinate space.
///
/// This is the entire contract between the segmentation model and everything
/// downstream. Nothing here knows how the mask was produced, which is what
/// makes swapping the model a contained change.
public struct Mask: Sendable, Equatable {
    public let width: Int
    public let height: Int

    /// Row-major, one byte per cell, 0 or 1.
    public let values: [UInt8]

    public init(width: Int, height: Int, values: [UInt8]) {
        precondition(width > 0 && height > 0, "mask must have positive extent")
        precondition(values.count == width * height,
                     "expected \(width * height) values, got \(values.count)")
        self.width = width
        self.height = height
        self.values = values
    }

    public subscript(x: Int, y: Int) -> Bool {
        precondition(x >= 0 && x < width && y >= 0 && y < height, "out of bounds")
        return values[y * width + x] != 0
    }

    /// How many cells are set. Used to pick between candidate masks.
    public var filledCount: Int {
        values.reduce(into: 0) { $0 += Int($1) }
    }

    public var isEmpty: Bool { filledCount == 0 }
}
```

- [ ] **Step 3: Write the tests**

`core/VisionAssistCore/Tests/MaskingTests/MaskTests.swift`:

```swift
import XCTest
@testable import Masking

final class MaskTests: XCTestCase {
    func testSubscriptReadsRowMajor() {
        // 3 wide, 2 tall, with only the middle of the bottom row set.
        let mask = Mask(width: 3, height: 2, values: [0, 0, 0,
                                                      0, 1, 0])
        XCTAssertTrue(mask[1, 1])
        XCTAssertFalse(mask[0, 0])
        XCTAssertFalse(mask[1, 0])
    }

    func testFilledCount() {
        XCTAssertEqual(Mask(width: 2, height: 2, values: [1, 0, 1, 1]).filledCount, 3)
        XCTAssertTrue(Mask(width: 2, height: 2, values: [0, 0, 0, 0]).isEmpty)
    }
}
```

- [ ] **Step 4: Run them**

Run:
```sh
cd core/VisionAssistCore && swift test --filter MaskingTests
```
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```sh
git add core/VisionAssistCore
git commit -m "Add Mask type"
```

---

## Task 3: The YOLO mask decoder

**Files:**
- Create: `core/VisionAssistCore/Sources/YOLODecoding/YOLODetection.swift`
- Create: `core/VisionAssistCore/Sources/YOLODecoding/YOLOMaskDecoder.swift`
- Create: `core/VisionAssistCore/Tests/YOLODecodingTests/YOLOMaskDecoderTests.swift`

- [ ] **Step 1: Write the detection types**

`core/VisionAssistCore/Sources/YOLODecoding/YOLODetection.swift`:

```swift
import Foundation

/// A box in the model's 640x640 input space.
public struct BoundingBox: Sendable, Equatable {
    public let x1: Float
    public let y1: Float
    public let x2: Float
    public let y2: Float

    public init(x1: Float, y1: Float, x2: Float, y2: Float) {
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    }

    public var area: Float {
        max(0, x2 - x1) * max(0, y2 - y1)
    }
}

/// One row of the NMS-resolved detection tensor.
public struct YOLODetection: Sendable, Equatable {
    public let box: BoundingBox
    public let confidence: Float

    /// The 32 mask coefficients, combined with the prototypes to form a mask.
    public let coefficients: [Float]
}
```

- [ ] **Step 2: Write the failing decoder test**

The fixture lives outside the package, so the path is derived from `#filePath`
rather than a bundle resource — this keeps a 3.3MB binary out of the built
product.

`core/VisionAssistCore/Tests/YOLODecodingTests/YOLOMaskDecoderTests.swift`:

```swift
import XCTest
@testable import Masking
@testable import YOLODecoding

/// Verifies the decoder against a real Core ML inference captured by
/// `tools/dump_decoder_fixture.py`. The expected mask in that fixture was
/// itself checked against ultralytics' own output at IoU 0.9941.
final class YOLOMaskDecoderTests: XCTestCase {

    struct Manifest: Decodable {
        struct Detections: Decodable { let rows: Int; let columns: Int }
        struct Prototypes: Decodable { let count: Int; let size: Int }
        struct ExpectedMask: Decodable { let size: Int; let filled: Int }
        struct Box: Decodable { let x1: Float; let y1: Float; let x2: Float; let y2: Float }

        let detections: Detections
        let prototypes: Prototypes
        let expectedMask: ExpectedMask
        let expectedBox: Box
        let expectedConfidence: Float

        enum CodingKeys: String, CodingKey {
            case detections, prototypes
            case expectedMask = "expected_mask"
            case expectedBox = "expected_box"
            case expectedConfidence = "expected_confidence"
        }
    }

    /// core/VisionAssistCore/Tests/YOLODecodingTests/<file> -> repo root
    static let fixtures = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()   // YOLODecodingTests
        .deletingLastPathComponent()   // Tests
        .deletingLastPathComponent()   // VisionAssistCore
        .deletingLastPathComponent()   // core
        .deletingLastPathComponent()   // repo root
        .appendingPathComponent("fixtures/decoder")

    private func floats(_ name: String) throws -> [Float] {
        let data = try Data(contentsOf: Self.fixtures.appendingPathComponent(name))
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    private func manifest() throws -> Manifest {
        let data = try Data(contentsOf: Self.fixtures.appendingPathComponent("manifest.json"))
        return try JSONDecoder().decode(Manifest.self, from: data)
    }

    func testParsesTheDetectionRow() throws {
        let manifest = try manifest()
        let decoder = YOLOMaskDecoder()

        let detections = decoder.detections(
            from: try floats("detections.bin"),
            rows: manifest.detections.rows,
            columns: manifest.detections.columns
        )

        XCTAssertEqual(detections.count, 1, "fixture has one detection above threshold")
        let detection = try XCTUnwrap(detections.first)
        XCTAssertEqual(detection.confidence, manifest.expectedConfidence, accuracy: 1e-4)
        XCTAssertEqual(detection.box.x1, manifest.expectedBox.x1, accuracy: 1e-3)
        XCTAssertEqual(detection.box.y2, manifest.expectedBox.y2, accuracy: 1e-3)
        XCTAssertEqual(detection.coefficients.count, 32)
    }

    func testProducesTheExpectedMask() throws {
        let manifest = try manifest()
        let decoder = YOLOMaskDecoder()

        let mask = try XCTUnwrap(decoder.decode(
            detections: try floats("detections.bin"),
            rows: manifest.detections.rows,
            columns: manifest.detections.columns,
            prototypes: try floats("prototypes.bin")
        ))

        XCTAssertEqual(mask.width, manifest.expectedMask.size)
        XCTAssertEqual(mask.height, manifest.expectedMask.size)
        XCTAssertEqual(mask.filledCount, manifest.expectedMask.filled)

        let expected = try Data(contentsOf: Self.fixtures.appendingPathComponent("expected_mask.bin"))
        XCTAssertEqual(mask.values, Array(expected), "mask differs from the captured inference")
    }

    func testReturnsNilWhenNothingClearsTheThreshold() throws {
        let manifest = try manifest()
        let decoder = YOLOMaskDecoder()

        let mask = decoder.decode(
            detections: [Float](repeating: 0, count: manifest.detections.rows * manifest.detections.columns),
            rows: manifest.detections.rows,
            columns: manifest.detections.columns,
            prototypes: try floats("prototypes.bin")
        )

        XCTAssertNil(mask)
    }
}
```

- [ ] **Step 3: Run it to watch it fail**

Run:
```sh
cd core/VisionAssistCore && swift test --filter YOLODecodingTests
```
Expected: compile failure — `YOLOMaskDecoder` does not exist yet.

- [ ] **Step 4: Write the decoder**

Note the sigmoid: thresholding `sigmoid(x) > t` is the same as `x > ln(t/(1-t))`,
so 25,600 exponentials per frame are replaced by one logarithm and a comparison.

`core/VisionAssistCore/Sources/YOLODecoding/YOLOMaskDecoder.swift`:

```swift
import Accelerate
import Foundation
import Masking

/// Turns the two raw Core ML output tensors into a single binary mask.
///
/// Expects the `nms=True` export, whose detection tensor is already resolved:
///   detections  [rows, columns]  columns = 4 box + 1 confidence + 1 class + 32 coefficients
///   prototypes  [32, 160, 160]
///
/// Unused rows are zero-padded, so rows are filtered by confidence rather than
/// counted.
public struct YOLOMaskDecoder: Sendable {

    public struct Configuration: Sendable {
        public var confidenceThreshold: Float = 0.5
        public var maskThreshold: Float = 0.5
        public var coefficientCount = 32
        public var prototypeSize = 160
        public var inputSize = 640

        public init() {}
    }

    public let configuration: Configuration

    public init(configuration: Configuration = Configuration()) {
        self.configuration = configuration
    }

    private var boxColumns: Int { 4 }
    private var confidenceColumn: Int { 4 }
    private var coefficientOffset: Int { 6 }

    /// Parse every row whose confidence clears the threshold.
    public func detections(from raw: [Float], rows: Int, columns: Int) -> [YOLODetection] {
        precondition(raw.count >= rows * columns, "detection buffer is too small")
        precondition(columns >= coefficientOffset + configuration.coefficientCount,
                     "detection rows are too narrow")

        var parsed: [YOLODetection] = []
        for row in 0..<rows {
            let base = row * columns
            let confidence = raw[base + confidenceColumn]
            guard confidence > configuration.confidenceThreshold else { continue }

            parsed.append(YOLODetection(
                box: BoundingBox(x1: raw[base], y1: raw[base + 1],
                                 x2: raw[base + 2], y2: raw[base + 3]),
                confidence: confidence,
                coefficients: Array(raw[(base + coefficientOffset)
                                        ..< (base + coefficientOffset + configuration.coefficientCount)])
            ))
        }
        return parsed
    }

    /// Combine one detection's coefficients with the prototypes.
    public func mask(for detection: YOLODetection, prototypes: [Float]) -> Mask {
        let size = configuration.prototypeSize
        let pixels = size * size
        let count = configuration.coefficientCount
        precondition(prototypes.count >= pixels * count, "prototype buffer is too small")

        // (1 x 32) * (32 x 25600) -> (1 x 25600)
        var logits = [Float](repeating: 0, count: pixels)
        detection.coefficients.withUnsafeBufferPointer { coefficients in
            prototypes.withUnsafeBufferPointer { prototypes in
                vDSP_mmul(coefficients.baseAddress!, 1,
                          prototypes.baseAddress!, 1,
                          &logits, 1,
                          1, vDSP_Length(pixels), vDSP_Length(count))
            }
        }

        // sigmoid(x) > t  <=>  x > ln(t / (1 - t))
        let t = configuration.maskThreshold
        let cutoff = log(t / (1 - t))

        // The box is in input space; the prototypes are 1/scale of that.
        let scale = Float(configuration.inputSize / size)
        let minX = max(0, Int(detection.box.x1 / scale))
        let minY = max(0, Int(detection.box.y1 / scale))
        let maxX = min(size, Int((detection.box.x2 / scale).rounded(.up)))
        let maxY = min(size, Int((detection.box.y2 / scale).rounded(.up)))

        var values = [UInt8](repeating: 0, count: pixels)
        guard minX < maxX, minY < maxY else {
            return Mask(width: size, height: size, values: values)
        }

        for y in minY..<maxY {
            let row = y * size
            for x in minX..<maxX where logits[row + x] > cutoff {
                values[row + x] = 1
            }
        }

        return Mask(width: size, height: size, values: values)
    }

    /// The mask for the largest detection, or nil if nothing clears the
    /// threshold. Largest rather than most confident, because that is what the
    /// pipeline downstream cares about.
    public func decode(detections raw: [Float], rows: Int, columns: Int,
                       prototypes: [Float]) -> Mask? {
        let candidates = detections(from: raw, rows: rows, columns: columns)
        guard let largest = candidates.max(by: { $0.box.area < $1.box.area }) else {
            return nil
        }
        return mask(for: largest, prototypes: prototypes)
    }
}
```

- [ ] **Step 5: Run the tests**

Run:
```sh
cd core/VisionAssistCore && swift test
```
Expected: all 5 tests pass. `testProducesTheExpectedMask` comparing byte-for-byte against the captured inference is the one that matters.

- [ ] **Step 6: Commit**

```sh
git add core/VisionAssistCore
git commit -m "Add YOLO mask decoder verified against a captured inference"
```

---

## Task 4: Deliver camera frames

**Files:**
- Create: `ios/VisionAssist/Camera/FrameStream.swift`
- Modify: `ios/VisionAssist/Camera/CameraController.swift`

- [ ] **Step 1: Write the frame stream**

Frames arrive faster than inference can consume them. Rather than queueing,
drop any frame that arrives while one is in flight — for navigation the newest
frame is the only one worth having.

`ios/VisionAssist/Camera/FrameStream.swift`:

```swift
@preconcurrency import AVFoundation
import Foundation
import os

/// Receives camera frames and forwards them one at a time.
///
/// Anything arriving while the handler is still working is dropped. The
/// handler runs on the video queue, not the main thread.
final class FrameStream: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {

    private let handler: (CVPixelBuffer) -> Void
    private let isBusy = OSAllocatedUnfairLock(initialState: false)

    init(handler: @escaping (CVPixelBuffer) -> Void) {
        self.handler = handler
    }

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let shouldRun = isBusy.withLock { busy -> Bool in
            if busy { return false }
            busy = true
            return true
        }
        guard shouldRun else { return }

        handler(buffer)
        isBusy.withLock { $0 = false }
    }
}
```

- [ ] **Step 2: Add the video output to the session**

In `ios/VisionAssist/Camera/CameraController.swift`, add these properties
alongside the existing `session` and `queue`:

```swift
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "com.willasbery.visionassist.video")
    private var frameStream: FrameStream?

    /// Set before `start()`. Called on the video queue, never the main thread.
    var onFrame: ((CVPixelBuffer) -> Void)?
```

Add a case to `ConfigurationError`:

```swift
    private enum ConfigurationError: Error {
        case noCamera
        case cannotAddInput
        case cannotAddOutput
    }
```

Then append this to the end of `configure()`, after `session.addInput(input)`:

```swift
        let stream = FrameStream { [weak self] buffer in
            self?.onFrame?(buffer)
        }
        frameStream = stream

        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.setSampleBufferDelegate(stream, queue: videoQueue)

        guard session.canAddOutput(videoOutput) else {
            throw ConfigurationError.cannotAddOutput
        }
        session.addOutput(videoOutput)

        // Buffers arrive in the sensor's landscape orientation; the app is
        // portrait-only, so rotate them to match what the preview shows.
        if let connection = videoOutput.connection(with: .video),
           connection.isVideoRotationAngleSupported(90) {
            connection.videoRotationAngle = 90
        }
```

- [ ] **Step 3: Prove frames arrive**

Temporarily add this to `ContentView`'s `.task`, before `await camera.start()`:

```swift
            camera.onFrame = { buffer in
                print("frame \(CVPixelBufferGetWidth(buffer))x\(CVPixelBufferGetHeight(buffer))")
            }
```

Build and run on the device, then read the log:
```sh
xcrun devicectl device process launch --console \
  --device 8459A37A-E0BE-5233-8F7B-C9246AEBAB6D com.willasbery.visionassist
```
Expected: a stream of `frame 720x1280` lines. Remove the temporary block once
confirmed.

- [ ] **Step 4: Commit**

```sh
git add ios/VisionAssist/Camera
git commit -m "Deliver camera frames with newest-frame-wins backpressure"
```

---

## Task 5: Run Core ML on the frames

**Files:**
- Create: `ios/VisionAssist/Segmentation/PavementSegmenter.swift`
- Create: `ios/VisionAssist/Segmentation/YOLOSegmenter.swift`
- Modify: `ios/project.yml`

- [ ] **Step 1: Bundle the model**

In `ios/project.yml`, replace the `sources:` block of the `VisionAssist` target with:

```yaml
    sources:
      - path: VisionAssist
      - path: ../model/weights/train11-yolov8n-seg-640.mlpackage
```

And add the package dependency, at the same indentation as `sources:`:

```yaml
    dependencies:
      - package: VisionAssistCore
        product: Masking
      - package: VisionAssistCore
        product: YOLODecoding
```

Then add this at the top level of the file, alongside `targets:`:

```yaml
packages:
  VisionAssistCore:
    path: ../core/VisionAssistCore
```

- [ ] **Step 2: Write the seam**

This protocol is the whole reason swapping the model later is a contained
change. Everything downstream depends on `Mask`, not on YOLO.

`ios/VisionAssist/Segmentation/PavementSegmenter.swift`:

```swift
import CoreVideo
import Masking

/// Produces a pavement mask from a camera frame.
///
/// The only contract between the model and the rest of the app. Swapping
/// YOLO for a semantic segmentation model means writing another conformance
/// and nothing else.
protocol PavementSegmenter {
    /// The mask for this frame, or nil if no pavement was found.
    /// Coordinates are the mask's own space, not the frame's.
    func mask(from frame: CVPixelBuffer) throws -> Mask?
}
```

- [ ] **Step 3: Write the Core ML implementation**

`ios/VisionAssist/Segmentation/YOLOSegmenter.swift`:

```swift
import CoreML
import CoreVideo
import Foundation
import Masking
import Vision
import YOLODecoding

/// Runs the exported YOLO segmentation model and decodes its output.
final class YOLOSegmenter: PavementSegmenter {

    enum SegmenterError: Error {
        case modelNotFound
        case unexpectedOutputs
        case unsupportedTensorType
    }

    private let model: VNCoreMLModel
    private let decoder: YOLOMaskDecoder

    init(modelName: String = "train11-yolov8n-seg-640",
         decoder: YOLOMaskDecoder = YOLOMaskDecoder()) throws {
        guard let url = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") else {
            throw SegmenterError.modelNotFound
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        self.model = try VNCoreMLModel(for: MLModel(contentsOf: url, configuration: configuration))
        self.decoder = decoder
    }

    func mask(from frame: CVPixelBuffer) throws -> Mask? {
        let request = VNCoreMLRequest(model: model)
        // The training data was resized to 640x640 by stretching, so match it.
        // This also makes the mapping back to frame space a plain linear scale.
        request.imageCropAndScaleOption = .scaleFill

        try VNImageRequestHandler(cvPixelBuffer: frame, orientation: .up).perform([request])

        guard let observations = request.results as? [VNCoreMLFeatureValueObservation] else {
            throw SegmenterError.unexpectedOutputs
        }

        let arrays = observations.compactMap(\.featureValue.multiArrayValue)

        // Bind by shape. Output names are generated by the converter and change
        // between exports.
        guard let detections = arrays.first(where: { $0.shape.count == 3 }),
              let prototypes = arrays.first(where: { $0.shape.count == 4 }) else {
            throw SegmenterError.unexpectedOutputs
        }

        return decoder.decode(
            detections: try floats(from: detections),
            rows: detections.shape[1].intValue,
            columns: detections.shape[2].intValue,
            prototypes: try floats(from: prototypes)
        )
    }

    private func floats(from array: MLMultiArray) throws -> [Float] {
        guard array.dataType == .float32 else {
            throw SegmenterError.unsupportedTensorType
        }
        return array.withUnsafeBufferPointer(ofType: Float.self) { Array($0) }
    }
}
```

- [ ] **Step 4: Regenerate and build**

Run:
```sh
cd ios && xcodegen generate && cd .. && xcodebuild \
  -project ios/VisionAssist.xcodeproj -scheme VisionAssist \
  -destination 'generic/platform=iOS Simulator' build 2>&1 | tail -4
```
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 5: Commit**

```sh
git add ios
git commit -m "Run Core ML segmentation on camera frames"
```

---

## Task 6: Draw the mask and show the frame time

**Files:**
- Create: `ios/VisionAssist/Overlay/MaskOverlayView.swift`
- Create: `ios/VisionAssist/SegmentationViewModel.swift`
- Modify: `ios/VisionAssist/ContentView.swift`

- [ ] **Step 1: Write the overlay**

The mask is 160x160 and the preview fills the screen, so a `CGImage` stretched
over the view is both correct and cheap — no per-pixel work on the main thread.

`ios/VisionAssist/Overlay/MaskOverlayView.swift`:

```swift
import Foundation
import Masking
import SwiftUI

/// Draws a mask as a translucent tint over whatever is behind it.
struct MaskOverlayView: View {
    let mask: Mask?
    var tint: Color = .green

    var body: some View {
        GeometryReader { geometry in
            if let image = mask.flatMap(Self.image(from:)) {
                Image(decorative: image, scale: 1)
                    .resizable()
                    .interpolation(.none)
                    .frame(width: geometry.size.width, height: geometry.size.height)
                    .colorMultiply(tint)
                    .allowsHitTesting(false)
            }
        }
    }

    /// Premultiplied RGBA, white where the mask is set and fully transparent
    /// where it is not. White so that `.colorMultiply` can tint it.
    ///
    /// The pixel buffer is wrapped in `Data` and handed to `CGDataProvider`,
    /// which retains it — pointing a provider at a local buffer would leave
    /// the image referencing freed memory.
    private static func image(from mask: Mask) -> CGImage? {
        let opacity: UInt8 = 180
        var pixels = [UInt8](repeating: 0, count: mask.values.count * 4)
        for (index, value) in mask.values.enumerated() where value != 0 {
            let offset = index * 4
            pixels[offset] = opacity      // premultiplied red
            pixels[offset + 1] = opacity  // premultiplied green
            pixels[offset + 2] = opacity  // premultiplied blue
            pixels[offset + 3] = opacity  // alpha
        }

        guard let provider = CGDataProvider(data: Data(pixels) as CFData) else {
            return nil
        }

        return CGImage(
            width: mask.width,
            height: mask.height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: mask.width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )
    }
}
```

- [ ] **Step 2: Write the view model**

`ios/VisionAssist/SegmentationViewModel.swift`:

```swift
import CoreVideo
import Foundation
import Masking

/// Runs segmentation on incoming frames and publishes the latest result.
@MainActor
final class SegmentationViewModel: ObservableObject {

    @Published private(set) var mask: Mask?
    @Published private(set) var milliseconds: Double = 0
    @Published private(set) var failure: String?

    private let segmenter: PavementSegmenter?

    init() {
        do {
            segmenter = try YOLOSegmenter()
        } catch {
            segmenter = nil
            failure = String(describing: error)
        }
    }

    /// Called on the video queue.
    nonisolated func process(_ frame: CVPixelBuffer) {
        guard let segmenter else { return }

        let start = CFAbsoluteTimeGetCurrent()
        let result = try? segmenter.mask(from: frame)
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000

        Task { @MainActor in
            self.mask = result
            // Smoothed, so the number is readable rather than flickering.
            self.milliseconds = self.milliseconds == 0
                ? elapsed
                : self.milliseconds * 0.9 + elapsed * 0.1
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
    @StateObject private var segmentation = SegmentationViewModel()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            switch camera.state {
            case .running:
                CameraPreviewView(session: camera.session)
                    .overlay(MaskOverlayView(mask: segmentation.mask))
                    .overlay(alignment: .top) { readout }
                    .ignoresSafeArea()
            case .denied:
                message("Camera access is off. Turn it on in Settings to use Vision Assist.")
            case .failed(let reason):
                message("The camera could not start.\n\(reason)")
            case .idle:
                ProgressView()
            }
        }
        .task {
            camera.onFrame = { [weak segmentation] frame in
                segmentation?.process(frame)
            }
            await camera.start()
        }
        .onDisappear { camera.stop() }
    }

    private var readout: some View {
        Text(segmentation.failure ?? String(format: "%.0f ms", segmentation.milliseconds))
            .font(.system(.caption, design: .monospaced))
            .foregroundStyle(.white)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(.black.opacity(0.6), in: Capsule())
            .padding(.top, 60)
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

- [ ] **Step 4: Build and run on the phone**

Run:
```sh
cd ios && xcodegen generate && cd .. && xcodebuild \
  -project ios/VisionAssist.xcodeproj -scheme VisionAssist \
  -destination 'platform=iOS,id=00008120-00112CC81E01A01E' \
  -allowProvisioningUpdates -derivedDataPath /tmp/va-build build 2>&1 | tail -3
xcrun devicectl device install app --device 8459A37A-E0BE-5233-8F7B-C9246AEBAB6D \
  /tmp/va-build/Build/Products/Debug-iphoneos/VisionAssist.app
xcrun devicectl device process launch --device 8459A37A-E0BE-5233-8F7B-C9246AEBAB6D \
  com.willasbery.visionassist
```

- [ ] **Step 5: Check it by hand**

Point the phone at a pavement. Expected:

1. A green tint over the pavement, tracking as you move
2. A millisecond readout that settles to a stable number
3. The tint disappears when pointed away from pavement, rather than sticking

Record the observed frame time — it is the first real measurement of on-device
inference cost, and Phase A's laptop figure of 40ms is the thing to beat.

- [ ] **Step 6: Commit**

```sh
git add ios
git commit -m "Draw the segmentation mask over the camera preview"
```

---

## Done when

- `swift test` passes in `core/VisionAssistCore`, including the byte-for-byte mask comparison against the captured inference
- The app draws a live mask over the pavement on the iPhone 15
- The frame-time readout shows a real on-device number

## Deliberately not in this plan

- **Mask alignment is unverified beyond eyeballing it.** `.scaleFill` plus a 90-degree rotation should line up, but the first honest check is whether the tint sits on the pavement. If it is offset or mirrored, that is a real bug to fix, not a tuning exercise.
- **The `conf=0.5` question from Phase A.** Watching the mask drop out live is the evidence needed to settle it. Gather that observation here; decide at the pathfinding milestone.
- **Frame rate control.** Every frame that is not dropped gets processed. Whether that is wasteful depends on the measured cost, which we do not have yet.
