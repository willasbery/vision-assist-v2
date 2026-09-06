"""Export a trained YOLO segmentation checkpoint to Core ML.

Run from the model/ directory:
    .venv/bin/python export_coreml.py
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "train11-yolov8n-seg-640.pt"


def export(weights: Path, imgsz: int, half: bool, nms: bool) -> Path:
    """Export `weights` to a .mlpackage sitting alongside it.

    `nms=True` bakes non-maximum suppression into the Core ML graph, which
    changes the detection output from [1, 37, 8400] raw anchors to [1, 300, 38]
    resolved detections. That removes the anchor decode and NMS from the Swift
    side entirely, and costs nothing in accuracy.
    """
    model = YOLO(weights)
    exported = model.export(format="coreml", imgsz=imgsz, half=half, nms=nms)
    return Path(exported)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--no-half", dest="half", action="store_false",
                        help="export at fp32 instead of fp16")
    parser.add_argument("--no-nms", dest="nms", action="store_false",
                        help="emit raw anchors instead of NMS-resolved detections")
    args = parser.parse_args()

    output = export(args.weights, args.imgsz, args.half, args.nms)
    print(f"Exported to {output}")


if __name__ == "__main__":
    main()
