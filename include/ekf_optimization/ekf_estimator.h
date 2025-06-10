#pragma once

#include "ekf_state.h"
#include <Eigen/Dense>
#include <memory>
#include <vector>
#include "imu.h"

// 前向声明
class Frame;
struct ImuData;
typedef std::shared_ptr<Frame> FramePtr;

namespace AirSLAM {

class EKFEstimator {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // 构造函数
    EKFEstimator();
    
    // 初始化EKF
    bool initialize(const FramePtr& frame, const ImuData& imu_data);
    
    // 预测步骤（使用IMU数据）
    void predict(const ImuData& imu_data,EKFState& state_);

    void updateCovariance(double dt, const ImuData& imu_data, EKFState& state_);

    Eigen::Matrix<double, 15, 15> computeStateTransitionMatrix(
    double dt, const ImuData& imu_data, const EKFState& state_) ;

    Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v);
    
    // 更新步骤（使用视觉特征）
    void update(const FramePtr& frame,EKFState& state_);
    
    // 获取当前状态
    EKFState& getState(){ return state_; }

    // 设置当前状态
    void setState(EKFState& state) { state_ = state; }
    
    // 设置过程噪声
    void setProcessNoise(const Eigen::Matrix<double, EKFState::ERROR_STATE_DIM, EKFState::ERROR_STATE_DIM>& Q);
    
    // 设置测量噪声
    void setMeasurementNoise(const Eigen::Matrix<double, 6, 6>& R);  // 假设视觉测量维度为6（位置和姿态）

private:
    // 状态
    EKFState state_;
    
    // 过程噪声协方差矩阵
    Eigen::Matrix<double, EKFState::ERROR_STATE_DIM, EKFState::ERROR_STATE_DIM> Q_;
    
    // 测量噪声协方差矩阵
    Eigen::Matrix<double, 6, 6> R_;
    
    // 上一次IMU时间戳
    double last_imu_time_;

    // 预积分对象
    Preinteration preintegration_;  // 如果需要使用IMU预积分，可以添加此成员变量
    
    // 计算状态转移矩阵
    Eigen::Matrix<double, EKFState::ERROR_STATE_DIM, EKFState::ERROR_STATE_DIM> computeStateTransitionMatrix(
        const ImuData& imu_data, double dt);
    
    // 计算过程噪声矩阵
    Eigen::Matrix<double, EKFState::ERROR_STATE_DIM, EKFState::ERROR_STATE_DIM> computeProcessNoiseMatrix(
        const ImuData& imu_data, double dt);
    
    // 计算测量雅可比矩阵
    Eigen::Matrix<double, 6, EKFState::ERROR_STATE_DIM> computeMeasurementJacobian(
        const FramePtr& frame);
    
    // 计算测量残差
    Eigen::Matrix<double, 6, 1> computeMeasurementResidual(
        const FramePtr& frame);
};

// 辅助函数声明
Eigen::Matrix3d skew(const Eigen::Vector3d& v);
Eigen::Vector3d quaternionError(const Eigen::Quaterniond& q1, const Eigen::Quaterniond& q2);

} // namespace AirSLAM 