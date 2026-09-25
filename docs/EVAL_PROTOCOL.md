# Evaluation protocol

Status: **draft**, fixed before the first full benchmark run. Changes after that are logged
at the end of this file with the reason.

## Data

- LM-O, BOP'19 test subset: 200 RGB-D images (Primesense, 640×480) of one cluttered scene,
  8 objects, 1445 target instances (`test_targets_bop19.json`, one instance per object and image).
- Object models: BOP `models/` (millimetres). Symmetries from `models_info.json` (objects 10
  "eggbox" and 11 "glue" are symmetric).
- No training on LM-O test images. Nothing is tuned on the test set: parameters are set on a
  development subset (every 10th image, 20 images) and then frozen; those images are reported
  separately and the headline uses all 200 images.

## Detection conditions (inputs to pose estimation, used as given)

| Condition | Source | Output | Targets covered |
|---|---|---|---:|
| `gt` | ground-truth visible masks | mask | 1445 |
| `gdrnpp` | BOP'23 default detections, GDRNPP detector trained on PBR images | box only | 1445 |
| `cnos` | BOP'23 default detections, CNOS (zero-shot, FastSAM + DINOv2) | mask | 1367 |

For each target, the top-`inst_count` detections of that object by score are used, as the BOP
localisation task allows. A target without a detection counts as a miss.

## Pose estimators

Every estimator gets the same inputs: RGB-D image, camera intrinsics, the detection's mask (or
box), and the object's CAD model. Each returns one pose and a confidence per detection.

## Metrics

- BOP: MSSD, MSPD and VSD with the official thresholds → AR_MSSD, AR_MSPD, AR_VSD, and
  AR = their mean (the BOP Challenge score). Symmetries as in `models_info.json`.
- ADD(-S) < 10 % of the object diameter (ADD-S for symmetric objects), for readability.
- Latency per instance (pose estimation only; detector time is reported separately as given).
- Grasp success: definition fixed before the grasp experiment.
