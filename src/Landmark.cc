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

#include <mutex>

using namespace std;

namespace ORB_SLAM2 {

void Landmark::SetDimension(const LandmarkDimension& dimension) {
    unique_lock<mutex> lock(mMutexPose);
    mDimension = dimension;
}

LandmarkDimension Landmark::GetDimension() {
    unique_lock<mutex> lock(mMutexPose);
    return mDimension;
}

void Landmark::SetPose(const cv::Mat& Tlw_)
{
    unique_lock<mutex> lock(mMutexPose);
    Tlw_.copyTo(Tlw);
    cv::Mat Rlw = Tlw.rowRange(0, 3).colRange(0, 3);
    cv::Mat tlw = Tlw.rowRange(0, 3).col(3);
    cv::Mat Rwl = Rlw.t();
    Lw = -Rwl*tlw;

    Twl = cv::Mat::eye(4, 4, Tlw.type());
    Rwl.copyTo(Twl.rowRange(0, 3).colRange(0, 3));
    Lw.copyTo(Twl.rowRange(0, 3).col(3));
}

cv::Point Landmark::GetProjectedCenter(const cv::Mat& Tcw)
{
    cv::Mat homo = Tcw.rowRange(0, 3).colRange(0, 3).dot(GetLandmarkCenter()) + Tcw.rowRange(0, 3).col(3);
    return cv::Point(
            static_cast<int>(homo.at<float>(0) / homo.at<float>(2)),
            static_cast<int>(homo.at<float>(1) / homo.at<float>(2)));
}

cv::Mat Landmark::GetPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tlw.clone();
}

cv::Mat Landmark::GetPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose);
    return Twl.clone();
}

cv::Mat Landmark::GetLandmarkCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Lw.clone();
}

cv::Mat Landmark::GetRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tlw.rowRange(0,3).colRange(0,3).clone();
}

cv::Mat Landmark::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tlw.rowRange(0,3).col(3).clone();
}

}