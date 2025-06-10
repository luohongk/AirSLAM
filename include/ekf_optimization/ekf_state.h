#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace AirSLAM {

class EKFState {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // 状态向量维度
    static constexpr int STATE_DIM = 16;  // 位置(3) + 速度(3) + 四元数(4) + 加速度计偏置(3) + 陀螺仪偏置(3)
    
    // 构造函数
    EKFState();
    
    // 协方差矩阵维度
    static constexpr int ERROR_STATE_DIM = 15;

    // 状态向量 (真实状态 16维)
    Eigen::Matrix<double, STATE_DIM, 1> x;  // [p v q ba bg]

    // 协方差矩阵 (误差状态 15维)
    Eigen::Matrix<double, ERROR_STATE_DIM, ERROR_STATE_DIM> P;  // [δp δv δθ δba δbg]
    
    // 获取位置
    Eigen::Vector3d getPosition() const;
    
    // 获取速度
    Eigen::Vector3d getVelocity() const;
    
    // 获取姿态（四元数）
    Eigen::Quaterniond getAttitude() const;
    
    // 获取加速度计偏置
    Eigen::Vector3d getAccBias() const;
    
    // 获取陀螺仪偏置
    Eigen::Vector3d getGyroBias() const;
    
    // 设置位置
    void setPosition(const Eigen::Vector3d& position);
    
    // 设置速度
    void setVelocity(const Eigen::Vector3d& velocity);
    
    // 设置姿态
    void setAttitude(const Eigen::Quaterniond& attitude);
    
    // 设置加速度计偏置
    void setAccBias(const Eigen::Vector3d& acc_bias);
    
    // 设置陀螺仪偏置
    void setGyroBias(const Eigen::Vector3d& gyro_bias);
};

} // namespace AirSLAM 