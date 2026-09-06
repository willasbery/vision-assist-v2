"""Capture one real Core ML inference as a golden fixture for the Swift decoder.

Writes raw little-endian float32 tensors plus the expected mask, so the Swift
tests can verify the decoder without Core ML or a device.

    model/.venv/bin/python tools/dump_decoder_fixture.py
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
