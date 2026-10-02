#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "posegrasp_cpp/backproject.hpp"

using posegrasp_cpp::backproject;
using posegrasp_cpp::Intrinsics;

TEST(Backproject, PinholeModel)
{
  const Intrinsics K{500.0, 400.0, 2.0, 1.0};
  std::vector<float> depth(4 * 3, 0.0F);
  depth[1 * 4 + 2] = 2.0F;  // the principal point (u = 2, v = 1)
  depth[2 * 4 + 3] = 1.0F;  // one pixel right of and below it
  const auto points = backproject(depth, 4, 3, K, 1);
  ASSERT_EQ(points.size(), 2U);
  EXPECT_FLOAT_EQ(points[0].x, 0.0F);
  EXPECT_FLOAT_EQ(points[0].y, 0.0F);
  EXPECT_FLOAT_EQ(points[0].z, 2.0F);
  EXPECT_FLOAT_EQ(points[1].x, 1.0F / 500.0F);
  EXPECT_FLOAT_EQ(points[1].y, 1.0F / 400.0F);
}

TEST(Backproject, SkipsInvalidDepthAndHonoursStride)
{
  const Intrinsics K{1.0, 1.0, 0.0, 0.0};
  std::vector<float> depth(4 * 4, 1.0F);
  depth[0] = 0.0F;
  depth[2] = std::numeric_limits<float>::quiet_NaN();
  // stride 2 visits (0,0) (2,0) (0,2) (2,2); the first two are invalid
  EXPECT_EQ(backproject(depth, 4, 4, K, 2).size(), 2U);
  EXPECT_EQ(backproject(depth, 4, 4, K, 1).size(), 14U);
}

TEST(Backproject, KeepsOnlyLabelledPixels)
{
  const Intrinsics K{1.0, 1.0, 0.0, 0.0};
  const std::vector<float> depth(3 * 2, 1.0F);
  const std::vector<uint16_t> labels{0, 3, 0, 0, 7, 0};
  const auto points = backproject(depth, 3, 2, K, 1, &labels);
  ASSERT_EQ(points.size(), 2U);
  EXPECT_EQ(points[0].label, 3);
  EXPECT_EQ(points[1].label, 7);
}

TEST(Backproject, RejectsMismatchedSizes)
{
  const Intrinsics K{1.0, 1.0, 0.0, 0.0};
  const std::vector<float> depth(5, 1.0F);
  EXPECT_THROW(backproject(depth, 3, 2, K, 1), std::invalid_argument);
}
