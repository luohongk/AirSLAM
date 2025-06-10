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

void EKFEstimator::predict(const ImuData& imu_data, EKFState& state_) {
 // 计算时间间隔
    double dt = imu_data.timestamp - last_imu_time_;
    last_imu_time_ = imu_data.timestamp;
    
    if (dt <= 0 || dt > 1.0) {
        // 时间间隔无效，跳过预测
        return;
    }

    // 从状态向量中提取当前状态
    Eigen::Vector3d position = state_.x.segment<3>(0);  // 位置 (x,y,z)
    Eigen::Vector3d velocity = state_.x.segment<3>(3);  // 速度 (vx,vy,vz)
    Eigen::Quaterniond orientation(
        state_.x(6),  // w
        state_.x(7),  // x
        state_.x(8),  // y
        state_.x(9)   // z
    );
    Eigen::Vector3d acc_bias = state_.x.segment<3>(10);  // 加速度计偏置
    Eigen::Vector3d gyro_bias = state_.x.segment<3>(13); // 陀螺仪偏置

    // 构造当前位姿变换矩阵 Twb0
    Eigen::Matrix4d Twb0 = Eigen::Matrix4d::Identity();
    Twb0.block<3, 3>(0, 0) = orientation.toRotationMatrix();
    Twb0.block<3, 1>(0, 3) = position;

    // 速度向量
    Eigen::Vector3d vwb0 = velocity;

    // 输出变量
    Eigen::Matrix4d Twb1;
    Eigen::Vector3d vwb1;

    // 调用预积分预测函数（假设已正确实现）
    preintegration_.Predict(Twb0, vwb0, Twb1, vwb1);

    // 从预测结果中提取新的位置和姿态
    Eigen::Vector3d new_position = Twb1.block<3, 1>(0, 3);
    Eigen::Matrix3d new_rotation = Twb1.block<3, 3>(0, 0);
    Eigen::Quaterniond new_orientation(new_rotation);

    // 更新状态向量
    state_.x.segment<3>(0) = new_position;      // 更新位置
    state_.x.segment<3>(3) = vwb1;              // 更新速度
    
    // 更新姿态四元数（确保归一化）
    new_orientation.normalize();
    state_.x(6) = new_orientation.w();
    state_.x(7) = new_orientation.x();
    state_.x(8) = new_orientation.y();
    state_.x(9) = new_orientation.z();

    // 更新协方差矩阵
    updateCovariance(dt, imu_data, state_);

    // 设置状态
    setState(state_);
}

void EKFEstimator::updateCovariance(double dt, const ImuData& imu_data, EKFState& state_) {
    // 计算状态转移矩阵 F
    Eigen::Matrix<double, 15, 15> F = computeStateTransitionMatrix(dt, imu_data, state_);
    
    // 计算过程噪声矩阵 Q
    Eigen::Matrix<double, 15, 15> Q = computeProcessNoiseMatrix(imu_data,dt);
    
    // 更新协方差矩阵 (P = F * P * F^T + Q)
    state_.P = F * state_.P * F.transpose() + Q;
}

Eigen::Matrix<double, 15, 15> EKFEstimator::computeStateTransitionMatrix(
    double dt, const ImuData& imu_data, const EKFState& state_) {
    
    // 提取当前状态
    Eigen::Quaterniond q(
        state_.x(6), state_.x(7), state_.x(8), state_.x(9)
    );
    Eigen::Matrix3d R = q.toRotationMatrix();
    
    // 创建状态转移矩阵 (15x15)
    Eigen::Matrix<double, 15, 15> F = Eigen::Matrix<double, 15, 15>::Identity();
    
    // 位置部分: ∂p/∂p = I, ∂p/∂v = I*dt
    F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;
    
    // 速度部分: ∂v/∂θ = -R*(a - ba)^× * dt
    Eigen::Vector3d acc = imu_data.acc;
    Eigen::Vector3d acc_bias = state_.x.segment<3>(10);
    Eigen::Matrix3d acc_skew = skewSymmetric(acc - acc_bias);
    F.block<3, 3>(3, 6) = -R * acc_skew * dt;
    
    // 速度部分: ∂v/∂ba = -R*dt
    F.block<3, 3>(3, 10) = -R * dt;
    
    // 姿态部分: ∂θ/∂θ = I - (ω - bg)^× * dt
    Eigen::Vector3d gyro = imu_data.gyr;
    Eigen::Vector3d gyro_bias = state_.x.segment<3>(13);
    Eigen::Matrix3d gyro_skew = skewSymmetric(gyro - gyro_bias);
    F.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() - gyro_skew * dt;
    
    // 姿态部分: ∂θ/∂bg = -I*dt
    F.block<3, 3>(6, 13) = -Eigen::Matrix3d::Identity() * dt;
    
    return F;
}

// 辅助函数：计算向量的反对称矩阵
Eigen::Matrix3d EKFEstimator::skewSymmetric(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m << 0, -v.z(), v.y(),
         v.z(), 0, -v.x(),
         -v.y(), v.x(), 0;
    return m;
}

void EKFEstimator::update(const FramePtr& frame,EKFState& state_) {
    if (!frame) {
        std::cerr << "Invalid frame for EKF update" << std::endl;
        return;
    }
    
    Eigen::Matrix<double, 6, 15> H = computeMeasurementJacobian(frame);
    Eigen::Matrix<double, 6, 1> y = computeMeasurementResidual(frame);
    Eigen::Matrix<double, 15, 6> K = state_.P * H.transpose() * (H * state_.P * H.transpose() + R_).inverse();
    Eigen::Matrix<double, 15, 1> dx = K * y;

    // 状态更新（将误差状态映射回真实状态）
    state_.x.segment<3>(0) += dx.segment<3>(0);   // 位置
    state_.x.segment<3>(3) += dx.segment<3>(3);   // 速度

    // 姿态更新
    Eigen::Vector3d dtheta = dx.segment<3>(6);
    Eigen::Quaterniond dq(1, dtheta(0)/2, dtheta(1)/2, dtheta(2)/2);  // 小角度四元数
    dq.normalize();
    Eigen::Quaterniond q(state_.x.segment<4>(6));
    q = dq * q;
    q.normalize();
    state_.x.segment<4>(6) << q.w(), q.x(), q.y(), q.z();

    // 偏置
    state_.x.segment<3>(10) += dx.segment<3>(9);   // ba
    state_.x.segment<3>(13) += dx.segment<3>(12);  // bg

    // 协方差更新
    state_.P = (Eigen::Matrix<double, 15, 15>::Identity() - K * H) * state_.P;

    // 设置状态
     setState(state_);

}

void EKFEstimator::setProcessNoise(
    const Eigen::Matrix<double, EKFState::ERROR_STATE_DIM, EKFState::ERROR_STATE_DIM>& Q) {
    Q_ = Q;
}

void EKFEstimator::setMeasurementNoise(const Eigen::Matrix<double, 6, 6>& R) {
    R_ = R;
}

Eigen::Matrix<double, EKFState::ERROR_STATE_DIM, EKFState::ERROR_STATE_DIM> 
EKFEstimator::computeStateTransitionMatrix(const ImuData& imu_data, double dt) {
    Eigen::Matrix<double, EKFState::ERROR_STATE_DIM, EKFState::ERROR_STATE_DIM> F = 
        Eigen::Matrix<double, EKFState::ERROR_STATE_DIM, EKFState::ERROR_STATE_DIM>::Identity();
    
    // 填充状态转移矩阵
    // 位置对速度的导数
    F.block<3,3>(0,3) = Eigen::Matrix3d::Identity() * dt;
    
    // 速度对姿态的导数（考虑重力）
    Eigen::Matrix3d R = state_.getAttitude().toRotationMatrix();
    F.block<3,3>(3,6) = -R * skew(imu_data.acc) * dt;
    
    return F;
}

Eigen::Matrix<double, EKFState::ERROR_STATE_DIM, EKFState::ERROR_STATE_DIM> 
EKFEstimator::computeProcessNoiseMatrix(const ImuData& imu_data, double dt) {
    Eigen::Matrix<double, EKFState::ERROR_STATE_DIM, EKFState::ERROR_STATE_DIM> Q = Q_;
    Q *= dt;  // 时间相关的噪声
    return Q;
}

Eigen::Matrix<double, 6, EKFState::ERROR_STATE_DIM> 
EKFEstimator::computeMeasurementJacobian(const FramePtr& frame) {
    Eigen::Matrix<double, 6, EKFState::ERROR_STATE_DIM> H = 
        Eigen::Matrix<double, 6, EKFState::ERROR_STATE_DIM>::Zero();
    
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