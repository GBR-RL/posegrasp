// Composable node: depth + camera info + detection label image -> point clouds.
//
// Publishes, with the depth image's header:
//   segments      x, y, z (float32, metres) and label (uint16) of every labelled pixel
//   scene/points  x, y, z of every `scene_stride`-th pixel: the obstacles for grasp planning
// Clouds are published as unique_ptr, so they are passed without a copy to subscribers in the
// same process when intra-process communication is on.

#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "posegrasp_cpp/backproject.hpp"

namespace posegrasp_cpp
{

using sensor_msgs::msg::CameraInfo;
using sensor_msgs::msg::Image;
using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

namespace
{

// One value per pixel from an image row buffer (rows may be padded: `step` bytes each).
template<typename T>
std::vector<T> pixels(const Image & msg)
{
  if (msg.is_bigendian) {
    throw std::runtime_error("big-endian images are not supported");
  }
  std::vector<T> out(static_cast<size_t>(msg.width) * msg.height);
  for (uint32_t v = 0; v < msg.height; ++v) {
    std::memcpy(
      out.data() + static_cast<size_t>(v) * msg.width,
      msg.data.data() + static_cast<size_t>(v) * msg.step, msg.width * sizeof(T));
  }
  return out;
}

std::vector<float> depth_metres(const Image & msg)
{
  if (msg.encoding == "16UC1" || msg.encoding == "mono16") {  // millimetres (REP 118)
    const auto mm = pixels<uint16_t>(msg);
    std::vector<float> out(mm.size());
    for (size_t i = 0; i < mm.size(); ++i) {
      out[i] = static_cast<float>(mm[i]) * 0.001F;
    }
    return out;
  }
  if (msg.encoding == "32FC1") {  // metres
    return pixels<float>(msg);
  }
  throw std::runtime_error("unsupported depth encoding " + msg.encoding);
}

std::unique_ptr<PointCloud2> to_cloud(
  const std::vector<LabelledPoint> & points, const std_msgs::msg::Header & header,
  bool with_label)
{
  auto cloud = std::make_unique<PointCloud2>();
  cloud->header = header;
  sensor_msgs::PointCloud2Modifier modifier(*cloud);
  if (with_label) {
    modifier.setPointCloud2Fields(
      4, "x", 1, PointField::FLOAT32, "y", 1, PointField::FLOAT32, "z", 1, PointField::FLOAT32,
      "label", 1, PointField::UINT16);
  } else {
    modifier.setPointCloud2FieldsByString(1, "xyz");
  }
  modifier.resize(points.size());
  cloud->is_dense = true;
  sensor_msgs::PointCloud2Iterator<float> x(*cloud, "x");
  sensor_msgs::PointCloud2Iterator<float> y(*cloud, "y");
  sensor_msgs::PointCloud2Iterator<float> z(*cloud, "z");
  for (const auto & p : points) {
    *x = p.x;
    *y = p.y;
    *z = p.z;
    ++x;
    ++y;
    ++z;
  }
  if (with_label) {
    sensor_msgs::PointCloud2Iterator<uint16_t> label(*cloud, "label");
    for (const auto & p : points) {
      *label = p.label;
      ++label;
    }
  }
  return cloud;
}

}  // namespace

class SegmentCloud : public rclcpp::Node
{
public:
  explicit SegmentCloud(const rclcpp::NodeOptions & options)
  : Node("segment_cloud", options),
    scene_stride_(static_cast<int>(declare_parameter<int64_t>("scene_stride", 2)))
  {
    segments_pub_ = create_publisher<PointCloud2>("segments", 5);
    scene_pub_ = create_publisher<PointCloud2>("scene/points", 5);
    depth_sub_.subscribe(this, "camera/depth/image_raw");
    info_sub_.subscribe(this, "camera/camera_info");
    labels_sub_.subscribe(this, "detections/labels");
    sync_ = std::make_shared<Sync>(Policy(10), depth_sub_, info_sub_, labels_sub_);
    sync_->registerCallback(
      std::bind(
        &SegmentCloud::on_frame, this, std::placeholders::_1, std::placeholders::_2,
        std::placeholders::_3));
    RCLCPP_INFO(
      get_logger(), "segment_cloud ready (intra-process: %s)",
      options.use_intra_process_comms() ? "on" : "off");
  }

private:
  using Policy = message_filters::sync_policies::ExactTime<Image, CameraInfo, Image>;
  using Sync = message_filters::Synchronizer<Policy>;

  void on_frame(
    const Image::ConstSharedPtr & depth, const CameraInfo::ConstSharedPtr & info,
    const Image::ConstSharedPtr & labels)
  {
    if (depth->width != labels->width || depth->height != labels->height) {
      RCLCPP_WARN(get_logger(), "depth and label images differ in size; frame skipped");
      return;
    }
    const Intrinsics K{info->k[0], info->k[4], info->k[2], info->k[5]};
    const auto metres = depth_metres(*depth);
    const auto label_values = pixels<uint16_t>(*labels);
    const int w = static_cast<int>(depth->width);
    const int h = static_cast<int>(depth->height);
    const auto segments = backproject(metres, w, h, K, 1, &label_values);
    const auto scene = backproject(metres, w, h, K, scene_stride_);
    segments_pub_->publish(to_cloud(segments, depth->header, true));
    scene_pub_->publish(to_cloud(scene, depth->header, false));
    RCLCPP_DEBUG(
      get_logger(), "%zu segment points, %zu scene points", segments.size(), scene.size());
  }

  int scene_stride_;
  rclcpp::Publisher<PointCloud2>::SharedPtr segments_pub_;
  rclcpp::Publisher<PointCloud2>::SharedPtr scene_pub_;
  message_filters::Subscriber<Image> depth_sub_;
  message_filters::Subscriber<CameraInfo> info_sub_;
  message_filters::Subscriber<Image> labels_sub_;
  std::shared_ptr<Sync> sync_;
};

}  // namespace posegrasp_cpp

RCLCPP_COMPONENTS_REGISTER_NODE(posegrasp_cpp::SegmentCloud)
