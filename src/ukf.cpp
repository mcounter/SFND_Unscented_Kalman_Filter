#include "ukf.h"
#include "Eigen/Dense"

#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

const double pi2 = M_PI + M_PI;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // Will be initialized during first measurement
  is_initialized_ = false;

  // Will be initialized during first prediction
  is_sigma_points_predicted_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // State dimension
  n_x_ = 5;

  // state vector (just initialize object)
  x_ = VectorXd(n_x_);

  // covariance matrix (just initialize object)
  P_ = MatrixXd(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 2;

  // Standard deviation for initial speed guess
  std_v_ = 10;

  // Standard deviation for initial angle guess
  std_fi_ = 3;

  // Standard deviation for initial angle velocity guess
  std_dfi_ = 3;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  // Augmented state dimension
  n_aug_ = 7;

  // Number of sigma points
  n_sigma_ = 2 * n_aug_ + 1;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // predicted sigma points matrix (just initialize object)
  Xsig_pred_ = MatrixXd(n_x_, n_sigma_);

  // Weights of sigma points
  weights_ = VectorXd(n_sigma_);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  weights_.tail(n_sigma_ - 1).setConstant(0.5 / (lambda_ + n_aug_));
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (is_initialized_ &&
      meas_package.timestamp_ > time_us_)
  {
    double delta_t = double(meas_package.timestamp_ - time_us_) / 1000000;
    
    Prediction(delta_t);
  }

  time_us_ = meas_package.timestamp_;
  
  if (meas_package.sensor_type_ == MeasurementPackage::LASER)
  {
    UpdateLidar(meas_package);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
  {
    UpdateRadar(meas_package);
  }
}

double normalizeFi(double fi)
{
  if (fi > 100 || fi < -100)
  {
    // Can be undefined
    return 0;
  }
  
  double mult = trunc(fi / pi2);
  fi -= mult * pi2;

  if (fi > M_PI)
  {
    fi -= pi2;
  }

  if (fi < -M_PI)
  {
    fi += pi2;
  }

  return fi;
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

  if (!is_initialized_ ||
      (is_sigma_points_predicted_ && delta_t <= 0))
  {
    return;
  }

  std::cout << "Predicted:" << std::endl;

  VectorXd x_aug(n_aug_);
  x_aug.head(n_x_) = x_;
  x_aug.tail(n_aug_ - n_x_).setZero();

  MatrixXd P_aug(n_aug_, n_aug_);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.block(0, n_x_, n_x_, n_aug_ - n_x_).setZero();
  P_aug.block(n_x_, 0, n_aug_ - n_x_, n_x_).setZero();
  P_aug.block(n_x_, n_x_, n_aug_ - n_x_, n_aug_ - n_x_) <<
    std_a_ * std_a_, 0,
    0, std_yawdd_ * std_yawdd_;

  MatrixXd P_aug_sqrt = P_aug.llt().matrixL();

  P_aug_sqrt *= sqrt(lambda_ + n_aug_);

  MatrixXd X_sigma_aug(n_aug_, n_sigma_);
  X_sigma_aug.col(0) = x_aug;
  X_sigma_aug.block(0, 1, n_aug_, n_aug_) = P_aug_sqrt.colwise() + x_aug;
  X_sigma_aug.block(0, n_aug_ + 1, n_aug_, n_aug_) = (-P_aug_sqrt).colwise() + x_aug;

  double& dt = delta_t;
  double dt2 = dt * dt;
  
  for (int i = 0; i < n_sigma_; ++i)
  {
      double& px = X_sigma_aug(0, i);
      double& py = X_sigma_aug(1, i);
      double& v = X_sigma_aug(2, i);
      double& fi = X_sigma_aug(3, i);
      double& dfi = X_sigma_aug(4, i);
      double& va = X_sigma_aug(5, i);
      double& vfi = X_sigma_aug(6, i);
      
      double cosfi = cos(fi);
      double sinfi = sin(fi);

      Xsig_pred_(0, i) = px + 0.5 * va * dt2 * cosfi;
      Xsig_pred_(1, i) = py + 0.5 * va * dt2 * sinfi;
          
      if (fabs(dfi) < 0.001)
      {
          Xsig_pred_(0, i) += v * cosfi * dt;
          Xsig_pred_(1, i) += v * sinfi * dt;
      }
      else
      {
          Xsig_pred_(0, i) += v / dfi * (sin(fi + dfi * dt) - sinfi);
          Xsig_pred_(1, i) += v / dfi * (cosfi - cos(fi + dfi * dt));
      }
      
      Xsig_pred_(2, i) = v + va * dt;
      Xsig_pred_(3, i) = normalizeFi(fi + dfi * dt + 0.5 * vfi * dt2);
      Xsig_pred_(4, i) = dfi + vfi * dt;
  }

  x_ = Xsig_pred_ * weights_;
  x_(3) = normalizeFi(x_(3));
  P_.setZero();

  for (int i = 0; i < n_sigma_; ++i)
  {
    VectorXd x_res = Xsig_pred_.col(i) - x_;
    x_res(3) = normalizeFi(x_res(3));
    P_ += weights_(i) * x_res * x_res.transpose();
  }

  is_sigma_points_predicted_ = true;

  std::cout << x_ << std::endl;
  std::cout << P_ << std::endl;
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  if (!use_laser_)
  {
    return;
  }

  if (!is_initialized_)
  {
    double pos_x = meas_package.raw_measurements_[0];
    double pos_y = meas_package.raw_measurements_[1];
    
    x_ << pos_x, pos_y, 0, 0, 0;

    P_ <<
      std_laspx_ * std_laspx_, 0, 0, 0, 0,
      0, std_laspy_ * std_laspy_, 0, 0, 0,
      0, 0, std_v_ * std_v_, 0, 0,
      0, 0, 0, std_fi_ * std_fi_ , 0,
      0, 0, 0, 0, std_dfi_ * std_dfi_;

    is_initialized_ = true;

    std::cout << "Init LIDAR" << std::endl;
    std::cout << x_ << std::endl;
    std::cout << P_ << std::endl;

    return;
  }

  if (!is_sigma_points_predicted_)
  {
    Prediction(0);
  }

  std::cout << "Update LIDAR" << std::endl;

  // Create predicted mesarement sigma points
  int n_z = 2;
  MatrixXd Zsig(n_z, n_sigma_);

  for (int i = 0; i < n_sigma_; ++i)
  {
    double& px = Xsig_pred_(0, i);
    double& py = Xsig_pred_(1, i);

    Zsig(0, i) = px;
    Zsig(1, i) = py;
  }

  // Calculate predicted measurement and measurement covariance matrix
  VectorXd z_pred = Zsig * weights_;
  MatrixXd S = MatrixXd(n_z, n_z).setZero();

  for (int i = 0; i < n_sigma_; ++i)
  {
    VectorXd z_res = Zsig.col(i) - z_pred;
    S += weights_(i) * z_res * z_res.transpose();
  }

  VectorXd r = VectorXd(n_z);
  r << std_laspx_ * std_laspx_, std_laspy_ * std_laspy_;
  S += r.asDiagonal();

  // Calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z).setZero();

  for (int i = 0; i < n_sigma_; ++i)
  {
    VectorXd x_res = Xsig_pred_.col(i) - x_;
    VectorXd z_res = Zsig.col(i) - z_pred;

    Tc += weights_(i) * x_res * z_res.transpose();
  }
  
  // Calculate Kalman gain K
  MatrixXd K = Tc * S.inverse();
  
  //update state mean and covariance matrix
  VectorXd y = meas_package.raw_measurements_ - z_pred;
  
  x_ += K * y;
  x_(3) = normalizeFi(x_(3));
  P_ -= K * S * K.transpose();

  std::cout << K << std::endl;
  std::cout << x_ << std::endl;
  std::cout << P_ << std::endl;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  if (!use_radar_)
  {
    return;
  }

  if (!is_initialized_)
  {
    double rho = meas_package.raw_measurements_[0];
    double phi = meas_package.raw_measurements_[1];
    double rho_dot = meas_package.raw_measurements_[2];
    
    x_ << rho * cos(phi), rho * sin(phi), rho_dot, 0, 0;

    P_ <<
      std_radr_ * std_radr_, 0, 0, 0, 0,
      0, std_radr_ * std_radr_, 0, 0, 0,
      0, 0, std_v_ * std_v_, 0, 0,
      0, 0, 0, std_fi_ * std_fi_ , 0,
      0, 0, 0, 0, std_dfi_ * std_dfi_;

    is_initialized_ = true;

    std::cout << "Init RADAR" << std::endl;
    std::cout << x_ << std::endl;
    std::cout << P_ << std::endl;

    return;
  }

  if (!is_sigma_points_predicted_)
  {
    Prediction(0);
  }

  std::cout << "Update RADAR" << std::endl;

  // Create predicted mesarement sigma points
  int n_z = 3;
  MatrixXd Zsig(n_z, n_sigma_);

  for (int i = 0; i < n_sigma_; ++i)
  {
    double& px = Xsig_pred_(0, i);
    double& py = Xsig_pred_(1, i);
    double& v = Xsig_pred_(2, i);
    double& fi = Xsig_pred_(3, i);
    
    double pxy11 = px * px + py * py;
    double pxy12 = sqrt(pxy11);
    
    Zsig(0, i) = pxy12;
    
    if (pxy12 < 0.0001)
    {
      Zsig(1, i) = 0;
      Zsig(2, i) = 0;
    }
    else
    {
      Zsig(1, i) = atan2(py, px);
      Zsig(2, i) = (px * v * cos(fi) + py * v * sin(fi)) / pxy12;
    }
  }

  // Calculate predicted measurement and measurement covariance matrix
  VectorXd z_pred = Zsig * weights_;
  z_pred(1) = normalizeFi(z_pred(1));
  MatrixXd S = MatrixXd(n_z, n_z).setZero();

  for (int i = 0; i < n_sigma_; ++i)
  {
    VectorXd z_res = Zsig.col(i) - z_pred;
    z_res(1) = normalizeFi(z_res(1));
    S += weights_(i) * z_res * z_res.transpose();
  }

  VectorXd r = VectorXd(n_z);
  r << std_radr_ * std_radr_, std_radphi_ * std_radphi_, std_radrd_ * std_radrd_;
  S += r.asDiagonal();

  // Calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z).setZero();

  for (int i = 0; i < n_sigma_; ++i)
  {
    VectorXd x_res = Xsig_pred_.col(i) - x_;
    x_res(3) = normalizeFi(x_res(3));

    VectorXd z_res = Zsig.col(i) - z_pred;
    z_res(1) = normalizeFi(z_res(1));

    Tc += weights_(i) * x_res * z_res.transpose();
  }
  
  // Calculate Kalman gain K
  MatrixXd K = Tc * S.inverse();
  
  //update state mean and covariance matrix
  VectorXd y = meas_package.raw_measurements_ - z_pred;
  y(1) = normalizeFi(y(1));
  
  x_ += K * y;
  x_(3) = normalizeFi(x_(3));
  P_ -= K * S * K.transpose();

  std::cout << K << std::endl;
  std::cout << x_ << std::endl;
  std::cout << P_ << std::endl;
}