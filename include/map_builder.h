#ifndef MAP_BUILDER_H_
#define MAP_BUILDER_H_

#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
#include <ros/ros.h>

#include "super_glue.h"
#include "read_configs.h"
#include "imu.h"
#include "dataset.h"
#include "camera.h"
#include "frame.h"
#include "point_matcher.h"
#include "line_processor.h"
#include "feature_detector.h"
#include "map.h"
#include "ros_publisher.h"
#include "g2o_optimization/types.h"
#include "ekf_optimization/ekf_estimator.h"

struct InputData{
  size_t index;
  double time;
  cv::Mat image_left;
  cv::Mat image_right;
  ImuDataList batch_imu_data;

  InputData() {}
  InputData& operator =(InputData& other){
		index = other.index;
		time = other.time;
		image_left = other.image_left.clone();
		image_right = other.image_right.clone();
		return *this;
	}
};
typedef std::shared_ptr<InputData> InputDataPtr;

enum FrameType {
  NormalFrame = 0,
  KeyFrame = 1,
  InitializationFrame = 2,
};

struct TrackingData{
  FramePtr frame;
  FrameType frame_type;
  FramePtr ref_keyframe;
  std::vector<cv::DMatch> matches;
  InputDataPtr input_data;

  TrackingData() {}
  TrackingData& operator =(TrackingData& other){
		frame = other.frame;
		ref_keyframe = other.ref_keyframe;
		matches = other.matches;
		input_data = other.input_data;
		return *this;
	}
};
typedef std::shared_ptr<TrackingData> TrackingDataPtr;

namespace AirSLAM {

class MapBuilder {
public:
  MapBuilder(VisualOdometryConfigs& configs, ros::NodeHandle nh);
  ~MapBuilder();

  bool UseIMU();
  void AddInput(InputDataPtr data);
  void ExtractFeatureThread();
  void TrackingThread();

  int TrackFrame(FramePtr ref_frame, FramePtr current_frame, std::vector<cv::DMatch>& matches, Preinteration& _preinteration);

  int FramePoseOptimization(FramePtr frame0, FramePtr frame, std::vector<MappointPtr>& mappoints, std::vector<int>& inliers, 
      Preinteration& preinteration);
  int AddKeyframeCheck(FramePtr ref_frame, FramePtr current_frame, std::vector<cv::DMatch>& matches);
  void InsertKeyframe(FramePtr frame);

  void PublishFrame(FramePtr frame, cv::Mat& image, FrameType frame_type, std::vector<cv::DMatch>& matches);
  void SaveTrajectory();
  void SaveTrajectory(std::string file_path);
  void SaveMap(const std::string& map_root);

  void Stop();
  bool IsStopped();

private:
  void MatchLines(std::vector<std::map<int, double>>& ref_points_on_lines, 
                 std::vector<std::map<int, double>>& current_points_on_lines,
                 std::vector<cv::DMatch>& point_matches,
                 int ref_num_points,
                 int current_num_points,
                 std::vector<int>& line_matches);

private:
  std::atomic<bool> _shutdown;
  std::atomic<bool> _feature_thread_stop;
  std::atomic<bool> _tracking_trhead_stop;
  std::atomic<bool> _init;
  std::atomic<bool> _insert_next_keyframe;
  int _track_id;
  int _line_track_id;

  VisualOdometryConfigs& _configs;
  std::shared_ptr<Camera> _camera;
  std::shared_ptr<PointMatcher> _point_matcher;
  std::shared_ptr<FeatureDetector> _feature_detector;
  std::shared_ptr<RosPublisher> _ros_publisher;
  std::shared_ptr<Map> _map;

  // EKF状态估计器
  std::shared_ptr<EKFEstimator> _ekf_estimator;
  bool _use_ekf;

  std::queue<InputDataPtr> _data_buffer;
  std::mutex _buffer_mutex;

  std::queue<TrackingDataPtr> _tracking_data_buffer;
  std::mutex _tracking_mutex;

  std::mutex _stop_mutex;

  std::thread _feature_thread;
  std::thread _tracking_thread;

  FramePtr _last_keyframe_feature;
  FramePtr _last_keyframe_tracking;
  FramePtr _last_tracked_frame;
  cv::Mat _last_keyimage;

  // for publishing
  cv::Mat key_image_pub;
  int key_image_id_pub;
  std::vector<cv::KeyPoint> keyframe_keypoints_pub;

  Preinteration _preinteration_keyframe;
};

} // namespace AirSLAM

#endif  // MAP_BUILDER_H_