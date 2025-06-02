#include "ekf_optimization/ekf_estimator.h"
#include "frame.h"
#include "imu.h"
#include <iostream>

namespace AirSLAM {

EKFEstimator::EKFEstimator() : last_imu_time_(0.0) {
    // 初始化过程噪声矩阵
    Q_.setIdentity();
    Q_ *= 1e-4;  // 设置一个较小的初始过程噪声
    
    // 初始化测量噪声矩阵
    R_.setIdentity();
    R_ *= 1e-2;  // 设置一个较小的初始测量噪声
}

bool EKFEstimator::initialize(const FramePtr& frame, const ImuData& imu_data) {
    if (!frame) {
        std::cerr << "Invalid frame for EKF initialization" << std::endl;
        return false;
    }
    
    // 从第一帧初始化状态
    state_.setPosition(frame->GetPose().block<3,1>(0,3));
    state_.setVelocity(Eigen::Vector3d::Zero());  // 初始速度设为0
    
    // 从位姿矩阵提取四元数
    Eigen::Matrix3d R = frame->GetPose().block<3,3>(0,0);
    Eigen::Quaterniond q(R);
    state_.setAttitude(q);
    
    state_.setAccBias(Eigen::Vector3d::Zero());   // 初始偏置设为0
    state_.setGyroBias(Eigen::Vector3d::Zero());
    
    last_imu_time_ = imu_data.timestamp;
    
    return true;
}

void EKFEstimator::predict(const ImuData& imu_data) {
    // 计算时间间隔
    double dt = imu_data.timestamp - last_imu_time_;
    last_imu_time_ = imu_data.timestamp;
    
    if (dt <= 0 || dt > 1.0) {
        // 时间间隔无效，跳过预测
        return;
    }
    
    // 计算状态转移矩阵
    Eigen::Matrix<double, EKFState::STATE_DIM, EKFState::STATE_DIM> F = 
        computeStateTransitionMatrix(imu_data, dt);
    
    // 计算过程噪声矩阵
    Eigen::Matrix<double, EKFState::STATE_DIM, EKFState::STATE_DIM> Q = 
        computeProcessNoiseMatrix(imu_data, dt);
    
    // 预测状态
    state_.x = F * state_.x;
    
    // 预测协方差
    state_.P = F * state_.P * F.transpose() + Q;
}

void EKFEstimator::update(const FramePtr& frame) {
    if (!frame) {
        std::cerr << "Invalid frame for EKF update" << std::endl;
        return;
    }
    
    // 计算测量雅可比矩阵
    Eigen::Matrix<double, 6, EKFState::STATE_DIM> H = 
        computeMeasurementJacobian(frame);
    
    // 计算测量残差
    Eigen::Matrix<double, 6, 1> y = computeMeasurementResidual(frame);
    
    // 计算卡尔曼增益
    Eigen::Matrix<double, EKFState::STATE_DIM, 6> K = 
        state_.P * H.transpose() * (H * state_.P * H.transpose() + R_).inverse();
    
    // 更新状态
    state_.x = state_.x + K * y;
    
    // 归一化四元数部分
    Eigen::Vector4d quat = state_.x.segment<4>(6);
    quat.normalize();
    state_.x.segment<4>(6) = quat;
    
    // 更新协方差
    Eigen::Matrix<double, EKFState::STATE_DIM, EKFState::STATE_DIM> I = 
        Eigen::Matrix<double, EKFState::STATE_DIM, EKFState::STATE_DIM>::Identity();
    state_.P = (I - K * H) * state_.P;
}

void EKFEstimator::setProcessNoise(
    const Eigen::Matrix<double, EKFState::STATE_DIM, EKFState::STATE_DIM>& Q) {
    Q_ = Q;
}

void EKFEstimator::setMeasurementNoise(const Eigen::Matrix<double, 6, 6>& R) {
    R_ = R;
}

Eigen::Matrix<double, EKFState::STATE_DIM, EKFState::STATE_DIM> 
EKFEstimator::computeStateTransitionMatrix(const ImuData& imu_data, double dt) {
    Eigen::Matrix<double, EKFState::STATE_DIM, EKFState::STATE_DIM> F = 
        Eigen::Matrix<double, EKFState::STATE_DIM, EKFState::STATE_DIM>::Identity();
    
    // 填充状态转移矩阵
    // 位置对速度的导数
    F.block<3,3>(0,3) = Eigen::Matrix3d::Identity() * dt;
    
    // 速度对姿态的导数（考虑重力）
    Eigen::Matrix3d R = state_.getAttitude().toRotationMatrix();
    F.block<3,3>(3,6) = -R * skew(imu_data.acc) * dt;
    
    return F;
}

Eigen::Matrix<double, EKFState::STATE_DIM, EKFState::STATE_DIM> 
EKFEstimator::computeProcessNoiseMatrix(const ImuData& imu_data, double dt) {
    Eigen::Matrix<double, EKFState::STATE_DIM, EKFState::STATE_DIM> Q = Q_;
    Q *= dt;  // 时间相关的噪声
    return Q;
}

Eigen::Matrix<double, 6, EKFState::STATE_DIM> 
EKFEstimator::computeMeasurementJacobian(const FramePtr& frame) {
    Eigen::Matrix<double, 6, EKFState::STATE_DIM> H = 
        Eigen::Matrix<double, 6, EKFState::STATE_DIM>::Zero();
    
    // 位置测量对状态的雅可比
    H.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
    
    // 姿态测量对状态的雅可比
    H.block<3,3>(3,6) = Eigen::Matrix3d::Identity();
    
    return H;
}

Eigen::Matrix<double, 6, 1> 
EKFEstimator::computeMeasurementResidual(const FramePtr& frame) {
    Eigen::Matrix<double, 6, 1> y;
    
    // 计算位置残差
    Eigen::Vector3d frame_position = frame->GetPose().block<3,1>(0,3);
    y.segment<3>(0) = frame_position - state_.getPosition();
    
    // 计算姿态残差
    Eigen::Matrix3d frame_rotation = frame->GetPose().block<3,3>(0,0);
    Eigen::Quaterniond q_meas(frame_rotation);
    Eigen::Quaterniond q_pred = state_.getAttitude();
    y.segment<3>(3) = quaternionError(q_meas, q_pred);
    
    return y;
}

// 辅助函数：计算反对称矩阵
Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d S;
    S << 0, -v(2), v(1),
         v(2), 0, -v(0),
         -v(1), v(0), 0;
    return S;
}

// 辅助函数：计算四元数误差
Eigen::Vector3d quaternionError(const Eigen::Quaterniond& q1, const Eigen::Quaterniond& q2) {
    Eigen::Quaterniond q_error = q1 * q2.inverse();
    return 2.0 * q_error.vec() / q_error.w();
}

} // namespace AirSLAM 