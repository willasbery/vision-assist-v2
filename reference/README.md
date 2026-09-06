# Python reference implementation

The v1 pipeline, kept as the correctness oracle for the Swift port. This is not
a product — it exists so that `VisionAssistCore` can be shown to reproduce it
stage by stage.

Only the pipeline modules were carried over. The v1 repo also holds the
training runs, datasets, profiling scripts and various abandoned optimisation
attempts; those stay archived at
[willasbery/vision-assist](https://github.com/willasbery/vision-assist).

## Running the tests

```sh
python3 -m venv .venv
.venv/bin/pip install numpy pydantic pytest
.venv/bin/python -m pytest tests/
```

The full pipeline additionally needs `ultralytics`, `torch` and `opencv-python`,
but the tests deliberately avoid those so they stay fast.

## Divergence from v1

This copy is not byte-identical to v1. See the design spec for the bug fixes
applied and why they were needed before it could serve as an oracle.
