/**
 * This file is part of CubeSLAM.
 *
 * Copyright (C) 2018, Carnegie Mellon University
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "Landmark.h"
#include <opencv2/core/eigen.hpp>
#include "Thirdparty/g2o/g2o/types/slam3d/se3quat.h"

#include <mutex>

using namespace std;
using namespace cv;

namespace ORB_SLAM2 {

Landmark::Landmark() = default;

Landmark::Landmark(Landmark& other)
{
    SetPose(other.GetPose());
    SetDimension(other.GetDimension());
}

void Landmark::SetDimension(const LandmarkDimension& dimension)
{
    unique_lock<mutex> lock(mMutexPose);
    mCuboid.scale[0] = dimension.edge13;
    mCuboid.scale[1] = dimension.edge12;
    mCuboid.scale[2] = dimension.height;
    mDimension = dimension;
}

LandmarkDimension Landmark::GetDimension()
{
    unique_lock<mutex> lock(mMutexPose);
    return mDimension;
}

void Landmark::SetPose(const Mat& Tlw_)
{
    unique_lock<mutex> lock(mMutexPose);
    Tlw_.copyTo(Tlw);
    Mat Rlw = Tlw.rowRange(0, 3).colRange(0, 3);
    Mat tlw = Tlw.rowRange(0, 3).col(3);
    Mat Rwl = Rlw.t();
    Lw = -Rwl * tlw;

    Twl = Mat::eye(4, 4, Tlw.type());
    Rwl.copyTo(Twl.rowRange(0, 3).colRange(0, 3));
    Lw.copyTo(Twl.rowRange(0, 3).col(3));

    Eigen::Matrix<double, 4, 4> homogeneous_matrix;
    cv::cv2eigen(Twl, homogeneous_matrix);
    mCuboid.pose = g2o::SE3Quat(homogeneous_matrix.block(0,0,3,3), homogeneous_matrix.col(3).head(3));
}

void Landmark::SetPose(const Mat& Rlw_, const Mat& tlw_)
{
    Mat Tlw_ = Mat::zeros(4, 4, CV_32F);
    Tlw_.at<float>(3, 3) = 1;
    Tlw_.colRange(0, 3).rowRange(0, 3) = Rlw_;
    Tlw_.col(3).rowRange(0, 3) = tlw_;
    SetPose(Tlw_);
}

void Landmark::SetPoseAndDimension(const g2o::cuboid Cuboid_)
{
    cv::eigen2cv(Cuboid_.pose.to_homogeneous_matrix(), Twl);
    Eigen::Vector3d scale = Cuboid_.scale;
    SetDimension(LandmarkDimension(scale[1], scale[2], scale[0]));

}

Point2f Landmark::GetProjectedCentroid(const Mat& Tcw)
{
    Mat centroidHomo = Tcw.rowRange(0, 3).colRange(0, 3).dot(GetCentroid()) + Tcw.rowRange(0, 3).col(3);
    return Point2f(centroidHomo.at<float>(0) / centroidHomo.at<float>(2),
                   centroidHomo.at<float>(1) / centroidHomo.at<float>(2));
}

Mat Landmark::GetPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tlw.clone();
}

Mat Landmark::GetPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose);
    return Twl.clone();
}

Mat Landmark::GetCentroid()
{
    unique_lock<mutex> lock(mMutexPose);
    return Lw.clone();
}

Mat Landmark::GetRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tlw.rowRange(0, 3).colRange(0, 3).clone();
}

Mat Landmark::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tlw.rowRange(0, 3).col(3).clone();
}

g2o::cuboid Landmark::GetCuboid()
{
    unique_lock<mutex> lock(mMutexPose);
    return mCuboid;
}

}