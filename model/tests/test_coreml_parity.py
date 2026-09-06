"""The Core ML export must produce the same masks as the PyTorch checkpoint.

fp16 conversion moves values slightly, so this asserts a high IoU rather than
equality.

The fixture frames are spread evenly across the v1 test split rather than taken
from the front of it — the split is sorted, and its first entries all come from
a single source video.
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
    assert len(FRAMES) == 8


@pytest.mark.parametrize("frame", FRAMES, ids=lambda p: p.name[:20])
def test_export_matches_checkpoint(models, frame):
    checkpoint, export = models
    expected, actual = largest_mask(checkpoint, frame), largest_mask(export, frame)

    if expected is None and actual is None:
        pytest.skip("neither model detected a pavement in this frame")
    assert expected is not None and actual is not None, "one model detected nothing"

    score = iou(expected, actual)
    print(f"\n{frame.name[:20]}  IoU = {score:.4f}")
    assert score >= MIN_IOU
