// Depth image -> camera-frame points, with the pinhole model of sensor_msgs/CameraInfo.
#pragma once

#include <cstdint>
#include <vector>

namespace posegrasp_cpp
{

struct Intrinsics
{
  double fx;
  double fy;
  double cx;
  double cy;
};

struct LabelledPoint
{
  float x;
  float y;
  float z;
  uint16_t label;
};

// Points (metres) of the pixels with a valid depth (> 0 and finite), visiting every `stride`-th
// pixel in both directions. With `labels` (one value per pixel), only pixels with a non-zero
// label are kept, and the point carries it.
std::vector<LabelledPoint> backproject(
  const std::vector<float> & depth_m, int width, int height, const Intrinsics & K, int stride,
  const std::vector<uint16_t> * labels = nullptr);

}  // namespace posegrasp_cpp
