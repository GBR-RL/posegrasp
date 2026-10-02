"""Hypothesis verification: which candidate pose explains the observed depth best?

The model is rendered at each candidate pose by ray casting the CAD mesh (posegrasp.render) over
the pixels around the detection, and compared with the measured depth:

- explained: segmented object pixels where rendered and measured depth agree within `tau`
- violation: pixels where the model would be in front of free space (measured depth is behind the
  rendered surface by more than `tau`), i.e. the camera sees through the hypothesised object
- occlusion (measured depth in front of the model) is not penalised: other objects may cover it

score = explained fraction of the segment x (1 - violation fraction of the rendered silhouette)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from posegrasp.data.bop import Pose
from posegrasp.render import render_depth

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DepthVerifier:
    tau_ratio: float = 0.05  # depth agreement, as a fraction of the object diameter
    margin: int = 16  # pixels around the segment's box

    def score(
        self,
        mesh_path: Path,
        diameter: float,
        pose: Pose,
        *,
        depth_mm: FloatArray,
        mask: NDArray[np.bool_],
        K: FloatArray,
    ) -> float:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0 or pose.t[2] <= 0:
            return 0.0
        h, w = depth_mm.shape
        roi = (
            max(0, int(xs.min()) - self.margin),
            max(0, int(ys.min()) - self.margin),
            min(w, int(xs.max()) + 1 + self.margin),
            min(h, int(ys.max()) + 1 + self.margin),
        )
        x0, y0, x1, y1 = roi
        rendered = render_depth(mesh_path, pose, K, roi)
        observed = depth_mm[y0:y1, x0:x1]
        segment = mask[y0:y1, x0:x1] & (observed > 0)
        tau = self.tau_ratio * diameter
        valid = (rendered > 0) & (observed > 0)
        agree = valid & (np.abs(rendered - observed) < tau)
        violation = valid & (observed > rendered + tau)
        segment_count = int(segment.sum())
        silhouette = int(valid.sum())
        explained = int((agree & segment).sum()) / max(1, segment_count)
        free_space = int(violation.sum()) / silhouette if silhouette else 1.0
        return explained * (1.0 - free_space)
