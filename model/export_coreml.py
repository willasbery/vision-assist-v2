"""Export a trained YOLO segmentation checkpoint to Core ML.

Run from the model/ directory:
    .venv/bin/python export_coreml.py
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "train11-yolov8n-seg-640.pt"


def export(weights: Path, imgsz: int, half: bool) -> Path:
    """Export `weights` to a .mlpackage sitting alongside it."""
    model = YOLO(weights)
    exported = model.export(format="coreml", imgsz=imgsz, half=half)
    return Path(exported)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--no-half", dest="half", action="store_false",
                        help="export at fp32 instead of fp16")
    args = parser.parse_args()

    output = export(args.weights, args.imgsz, args.half)
    print(f"Exported to {output}")


if __name__ == "__main__":
    main()
