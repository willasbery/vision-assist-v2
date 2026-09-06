# Why the mask never appeared

**Date:** 2026-09-06
**Symptom:** On device, Core ML ran happily at 12-15ms per frame but the mask
overlay drew essentially never — perhaps one frame in a long recording.

## Root cause

The app fed Vision `.scaleFill`, which squashes the camera frame into the
model's square input. A 720x1280 portrait frame squashed to 640x640 is a far
more violent distortion than anything in the training set, and the model's
confidence collapses below the 0.5 threshold on nearly every frame.

## Why the wrong choice was made

The Roboflow dataset README says images were preprocessed with "Resize to
640x640 (Stretch)". From that it seemed to follow that the model expects
stretched input, so `.scaleFill` should match training.

The reasoning was plausible and the conclusion was wrong. Whatever the dataset
preparation did to its source images, the model in practice performs far better
on letterboxed input. The lesson is narrow and worth keeping: a README
describing dataset preparation is not a measurement of what the trained model
prefers at inference.

## The measurement

Fourteen frames pulled from a screen recording taken on the device, run through
the exported Core ML model at both preprocessings. Best detection confidence:

| frame | stretch | letterbox |
|---|---|---|
| a_04 | 0.274 | 0.869 |
| a_07 | 0.295 | 0.807 |
| a_08 | 0.425 | 0.687 |
| a_09 | 0.000 | 0.697 |
| b_02 | 0.000 | 0.270 |
| b_03 | 0.008 | 0.845 |
| b_04 | 0.000 | 0.598 |
| b_05 | 0.003 | 0.584 |

Frames scoring zero under both are genuine misses — mid-turn motion blur, or
walls and ceiling filling the view.

Padding colour barely matters. Ultralytics pads grey (114,114,114) and Vision
pads black; black scores slightly lower but comparably (0.772 versus 0.845 on
b_03), so `.scaleFit` is good enough and no manual preprocessing is needed.

## Why the threshold was not the fix

Lowering the confidence threshold was the obvious candidate, and Phase A had
already flagged the model's weak recall on video frames. But it would have been
a fix to the symptom: dropping to 0.25 would have scraped a few frames through
at poor quality while b_03, at 0.008, still failed. Letterboxing takes that same
frame to 0.845.

The `conf=0.5` question from Phase A is still open, but it should be decided
against letterboxed input, not against a distortion artefact.

## The fix

`.scaleFill` becomes `.scaleFit`, and the letterbox padding is then cropped out
of the mask so it lines up with the camera frame rather than the model's square
input. `LetterboxGeometry` computes that content rectangle; `Mask.cropped(to:)`
applies it. Both are covered by unit tests.

The overlay also had to change: the preview layer uses `.resizeAspectFill`, so
the mask now needs the same aspect-fill crop or the two would not align. This
is the coordinate mapping that choosing `.scaleFill` had originally avoided.
