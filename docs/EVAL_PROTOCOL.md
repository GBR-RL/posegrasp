# Evaluation protocol

Status: **frozen on 2026-10-02**, after the development runs below and before the first full
benchmark run and the full grasp experiment. Changes after that are logged at the end of
this file with the reason.

## Data

- LM-O, BOP'19 test subset: 200 RGB-D images (Primesense, 640×480) of one cluttered scene,
  8 objects, 1445 target instances (`test_targets_bop19.json`, one instance per object and image).
- Object models: BOP `models/` (millimetres) for pose estimation and grasp synthesis;
  `models_eval/` for the pose errors, as the official toolkit uses. Symmetries from
  `models_info.json` (objects 10 "eggbox" and 11 "glue" are symmetric).
- No training on LM-O test images. Nothing is tuned on the full set: settings were chosen on a
  development subset (every 10th targeted image: 20 images, 144 instances) and then frozen. The
  headline uses all 200 images; the development images are among them.

## Detection conditions (inputs to pose estimation, used as given)

| Condition | Source | Output | Targets covered |
|---|---|---|---:|
| `gt` | ground-truth visible masks | mask | 1445 |
| `gdrnpp` | BOP'23 default detections, GDRNPP detector trained on PBR images | box only | 1445 |
| `cnos` | BOP'23 default detections, CNOS (zero-shot, FastSAM + DINOv2) | mask | 1367 |

For each target, the top-`inst_count` detections of that object by score are used, as the BOP
localisation task allows. A target without a detection counts as a miss. A box-only detection is
cut down to the depth points within 0.6 diameters of the surface point at the box centre.

## Pose estimators (frozen settings)

Every estimator gets the same inputs: the depth image, camera intrinsics, the detection's mask
(or box) and the object's CAD model, and returns one pose per detection. Both share the same
segment preparation (statistical outlier removal) and the same point-to-plane ICP refinement
(scene segment onto the full model, coarse to fine at 3, 1.5 and 0.75 voxels of 2 % of the
diameter, 30 iterations each).

| Method | Hypotheses | Selection |
|---|---|---|
| `fpfh-verify-r5` | FPFH (voxel 4 % of the diameter, radius 5 voxels) + RANSAC (100 000 iterations, mutual filter), 5 restarts with different seeds, each refined by ICP | depth verification |
| `ppf-score` | PPF (OpenCV `ppf_match_3d`; sampling 4 % and distance step 5 % of the diameter, every 5th scene point as reference), the 5 best pose clusters, each refined by ICP | ICP inlier fraction |

Depth verification renders the model at each hypothesis and scores the fraction of the segment
whose depth it explains (within 5 % of the diameter), times one minus the fraction of the rendered
silhouette lying in front of observed free space.

### Development runs that fixed these settings (dev subset, AR)

| Configuration | `gt` | `gdrnpp` | `cnos` |
|---|---:|---:|---:|
| FPFH, ICP score, 1 restart | 0.335 | | |
| FPFH, ICP score, 5 restarts | 0.549 | | |
| **FPFH, depth verification, 5 restarts** | **0.556** | 0.452 | 0.477 |
| **PPF, ICP score** | **0.756** | 0.646 | 0.610 |
| PPF, depth verification | 0.756 | 0.648 | 0.599 |

FPFH gains mostly from restarts and slightly from verification. For PPF, verification is level
with the ICP score on average (mean AR 0.668 vs 0.671), so the cheaper ICP score is kept.
Latencies of these runs were measured with several runs sharing the CPU and are not reported.

## Metrics

- BOP: VSD, MSSD and MSPD with the official settings → AR_VSD, AR_MSSD, AR_MSPD, and AR = their
  mean (the BOP Challenge score). VSD: δ = 15 mm, τ = 0.05…0.5 of the diameter (step 0.05), BOP'19
  visibility (missing depth counts as visible), step pixel cost, correct below 0.05…0.5. MSSD:
  correct below 0.05…0.5 of the diameter. MSPD: correct below 5…50 px (× width / 640). Recall is
  over all target instances; AR is the mean recall over the thresholds (and over τ for VSD).
- Each error is checked against the official `bop_toolkit` (pinned commit `cea62d6`) in the unit
  tests; VSD with the toolkit's own function on our renderings.
- ADD(-S) < 10 % of the object diameter (ADD-S for symmetric objects), for readability.
- Latency per instance: segment preparation, hypotheses, ICP and selection (detector time is
  reported separately, as given). Latencies are compared within one job on one machine.

## Grasp success

The gripper is a Franka Hand modelled as three boxes (two 20 × 12 × 50 mm fingers, a
60 × 200 × 70 mm palm), 80 mm stroke, finger pads 18 mm square with their centre 9 mm behind the
fingertip, Coulomb friction μ = 0.5 (friction cone half-angle 26.6°).

1. **Grasps per object** (offline): antipodal pairs from 1500 surface samples, each a ray into the
   object along the inward normal to the opposite wall; the object width must leave 5 mm on
   each side within the stroke and both normals must lie inside the friction cone; 12 approach
   directions around each closing axis; up to 3000 grasps kept that pass the test in step 3 on
   the model itself.
2. **Planning** from a pose estimate: the object's grasps are moved to the estimated pose. A grasp
   must approach within 60° of the camera ray. Scene points (every 2nd pixel) further than 8 mm
   from the posed model are obstacles; at most 3 may lie inside the open gripper. Feasible grasps
   are ranked by 0.5 × approach alignment + 0.3 × clearance (capped at 20 mm, normalised) +
   0.2 × friction margin; the best one is executed. No feasible grasp counts as a failure.
3. **Judging** on the true pose, which success means:
   - the open gripper does not collide with the object (its surface samples) nor with more than
     3 points of the rest of the scene (depth points outside the object's true visible mask),
   - closing the fingers along the closing axis, both pads touch the object (rays from a 3 × 3
     grid on each pad; the first hit per finger is its contact), and
   - both contact normals lie inside the friction cone around the closing axis.
4. **Oracle**: planning from the true pose. It measures the planner and the scene, not pose
   estimation, and bounds what any estimator can reach.

Reported per method and condition: success rate over all targets (a missing pose is a failure),
the share of targets with a feasible plan, failure reasons, and success by the MSSD of the pose
used (bins of 0.05, 0.1, 0.2 and 0.5 of the diameter).

## Change log

- 2026-10-02: frozen (this version). Earlier draft: AR over MSSD and MSPD only, errors on
  `models/`, ADD-S measured from the estimate to the true pose; all changed to match the official
  toolkit before any full run.
