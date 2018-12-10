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
#include "Converter.h"

#include <mutex>

using namespace std;
using namespace cv;

namespace ORB_SLAM2 {

Landmark::Landmark() = default;

Landmark::Landmark(Landmark& other)
{
    SetPose(other.GetPose());
    SetDimension(other.GetDimension());
    mCuboid = g2o::cuboid();
}

void Landmark::SetDimension(const Dimension3D& dimension)
{
    unique_lock<mutex> lock(mMutexPose);
    mCuboid.scale[0] = dimension.edge13;
    mCuboid.scale[1] = dimension.edge12;
    mCuboid.scale[2] = dimension.edge18;
    mDimension = dimension;
}

Dimension3D Landmark::GetDimension()
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

    mCuboid.pose = ORB_SLAM2::Converter::toSE3Quat(Twl);
}

void Landmark::SetPose(const Mat& Rlw, const Mat& tlw)
{
    unique_lock<mutex> lock(mMutexPose);
    Mat Rwl = Rlw.t();
    Lw = -Rwl * tlw;

    Tlw = TFromRt(Rlw, tlw);
    Twl = TFromRt(Rwl, Lw);
}

void Landmark::SetPoseAndDimension(const g2o::cuboid Cuboid_)
{
    Twl = ORB_SLAM2::Converter::toCvMat(Cuboid_.pose);
    Eigen::Vector3d scale = Cuboid_.scale;
    SetDimension(LandmarkDimension(scale[1], scale[2], scale[0]));
}

Point2f Landmark::GetProjectedCentroid(const Mat& Tcw, const Mat& K)
{
    Mat centroidHomo = K * (Tcw.rowRange(0, 3).colRange(0, 3) * GetCentroid() + Tcw.rowRange(0, 3).col(3));
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

Cuboid2D Landmark::Project(const cv::Mat& Tcw, const cv::Mat& K) {
    Cuboid2D cuboid;
    auto centroid = Tcw.rowRange(0, 3).colRange(0, 3) * Lw + Tcw.rowRange(0, 3).col(3);
    Mat Tcl = Tcw * Twl;
    Mat Rlc = Tcl.rowRange(0, 3).colRange(0, 3).t();
    auto d1 = Rlc.col(0) * mDimension.edge13 / 2;
    auto d3 = Rlc.col(1) * mDimension.edge18 / 2;
    auto d2 = Rlc.col(2) * mDimension.edge12 / 2;

    Mat corners3D[8] {
        centroid + d1 + d2 - d3,
        centroid - d1 + d2 - d3,
        centroid + d1 - d2 - d3,
        centroid - d1 - d2 - d3,
        centroid - d1 - d2 + d3,
        centroid + d1 - d2 + d3,
        centroid - d1 + d2 + d3,
        centroid + d1 + d2 + d3,
    };
    for (int i = 0; i < 8; ++i)
        cuboid.corners[i] = PointFrom2DHomo(K * corners3D[i]);

    cuboid.valid = true;
    cuboid.Rlc = Rlc;

    return cuboid;
}

g2o::cuboid Landmark::GetCuboid()
{
    unique_lock<mutex> lock(mMutexPose);
    return mCuboid;
}

}