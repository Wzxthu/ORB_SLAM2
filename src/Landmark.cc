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
using namespace cv;

namespace ORB_SLAM2 {

void Landmark::SetDimension(const LandmarkDimension& dimension) {
    unique_lock<mutex> lock(mMutexPose);
    mDimension = dimension;
}

LandmarkDimension Landmark::GetDimension() {
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
    Lw = -Rwl*tlw;

    Twl = Mat::eye(4, 4, Tlw.type());
    Rwl.copyTo(Twl.rowRange(0, 3).colRange(0, 3));
    Lw.copyTo(Twl.rowRange(0, 3).col(3));
}

Point2f Landmark::GetProjectedCenter(const Mat& Tcw)
{
    Mat homo = Tcw.rowRange(0, 3).colRange(0, 3).dot(GetLandmarkCenter()) + Tcw.rowRange(0, 3).col(3);
    return Point2f(homo.at<float>(0) / homo.at<float>(2), homo.at<float>(1) / homo.at<float>(2));
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

Mat Landmark::GetLandmarkCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Lw.clone();
}

Mat Landmark::GetRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tlw.rowRange(0,3).colRange(0,3).clone();
}

Mat Landmark::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tlw.rowRange(0,3).col(3).clone();
}

}