#include "ekf_optimization/ekf_state.h"

namespace AirSLAM {

EKFState::EKFState() {
    // 初始化状态向量为零向量
    x.setZero();
    
    // 初始化协方差矩阵为单位矩阵
    P.setIdentity();
    P *= 1e-6;  // 设置一个小的初始不确定性
    
    // 设置初始四元数为单位四元数 (x, y, z, w)
    x.segment<4>(6) << 0, 0, 0, 1;
}

Eigen::Vector3d EKFState::getPosition() const {
    return x.segment<3>(0);
}

Eigen::Vector3d EKFState::getVelocity() const {
    return x.segment<3>(3);
}

Eigen::Quaterniond EKFState::getAttitude() const {
    // 四元数存储顺序: x, y, z, w
    Eigen::Quaterniond q(x(9), x(6), x(7), x(8));
    q.normalize();  // 确保四元数归一化
    return q;
}

Eigen::Vector3d EKFState::getAccBias() const {
    return x.segment<3>(10);
}

Eigen::Vector3d EKFState::getGyroBias() const {
    return x.segment<3>(13);
}

void EKFState::setPosition(const Eigen::Vector3d& position) {
    x.segment<3>(0) = position;
}

void EKFState::setVelocity(const Eigen::Vector3d& velocity) {
    x.segment<3>(3) = velocity;
}

void EKFState::setAttitude(const Eigen::Quaterniond& attitude) {
    // 确保输入的四元数是归一化的
    Eigen::Quaterniond q_normalized = attitude.normalized();
    
    // 四元数存储顺序: x, y, z, w
    x(6) = q_normalized.x();
    x(7) = q_normalized.y();
    x(8) = q_normalized.z();
    x(9) = q_normalized.w();
}

void EKFState::setAccBias(const Eigen::Vector3d& acc_bias) {
    x.segment<3>(10) = acc_bias;
}

void EKFState::setGyroBias(const Eigen::Vector3d& gyro_bias) {
    x.segment<3>(13) = gyro_bias;
}

} // namespace AirSLAM 