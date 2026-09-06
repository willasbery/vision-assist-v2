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
