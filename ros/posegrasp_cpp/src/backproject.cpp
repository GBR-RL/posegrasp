#include "posegrasp_cpp/backproject.hpp"

#include <cmath>
#include <stdexcept>

namespace posegrasp_cpp
{

std::vector<LabelledPoint> backproject(
  const std::vector<float> & depth_m, int width, int height, const Intrinsics & K, int stride,
  const std::vector<uint16_t> * labels)
{
  const auto pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
  if (depth_m.size() != pixels || (labels != nullptr && labels->size() != pixels)) {
    throw std::invalid_argument("depth and labels must have width * height values");
  }
  if (stride < 1 || K.fx <= 0.0 || K.fy <= 0.0) {
    throw std::invalid_argument("stride and focal lengths must be positive");
  }
  std::vector<LabelledPoint> points;
  points.reserve(pixels / static_cast<size_t>(stride * stride));
  for (int v = 0; v < height; v += stride) {
    for (int u = 0; u < width; u += stride) {
      const size_t i = static_cast<size_t>(v) * static_cast<size_t>(width) +
        static_cast<size_t>(u);
      const float z = depth_m[i];
      if (!(z > 0.0F) || !std::isfinite(z)) {
        continue;
      }
      const uint16_t label = labels == nullptr ? 0 : (*labels)[i];
      if (labels != nullptr && label == 0) {
        continue;
      }
      points.push_back(
        {static_cast<float>((u - K.cx) * z / K.fx), static_cast<float>((v - K.cy) * z / K.fy), z,
          label});
    }
  }
  return points;
}

}  // namespace posegrasp_cpp
