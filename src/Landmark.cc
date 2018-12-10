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
#include "Converter.h"

#include <mutex>

#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace cv;

namespace ORB_SLAM2 {

int Landmark::landmarkCnt = 0;

//Landmark::Landmark(Landmark& other)
//{
//    SetPose(other.GetPose());
//    SetDimension(other.GetDimension());
//
//    bboxCenter = other.bboxCenter;
//    mQuality = other.mQuality;
//    mClassIdx = other.mClassIdx;
//    mnLandmarkId = other.mnLandmarkId;
//}

Dimension3D Landmark::GetDimension()
{
    unique_lock<mutex> lock(mMutexPose);
    return mDimension;
}

void Landmark::SetDimension(const Dimension3D& dimension)
{
    unique_lock<mutex> lock(mMutexPose);
    SetDimensionNoLock(dimension);
}

void Landmark::SetPose(const Mat& Tlw_)
{
    unique_lock<mutex> lock(mMutexPose);
    SetPoseNoLock(Tlw_);
}

void Landmark::SetPose(const Mat& Rlw, const Mat& tlw)
{
    unique_lock<mutex> lock(mMutexPose);
    SetPoseNoLock(Rlw, tlw);
}

void Landmark::SetDimensionNoLock(const Dimension3D& dimension)
{
    unique_lock<mutex> lock(mMutexPose);
    mpCuboid->setScale(dimension.edge13, dimension.edge12, dimension.edge18);
    mDimension = dimension;
}

void Landmark::SetPoseNoLock(const Mat& Tlw_)
{
    Tlw_.copyTo(Tlw);
    Mat Rlw = Tlw.rowRange(0, 3).colRange(0, 3);
    Mat tlw = Tlw.rowRange(0, 3).col(3);
    Mat Rwl = Rlw.t();
    Lw = -Rwl * tlw;

    Twl = Mat::eye(4, 4, Tlw.type());
    Rwl.copyTo(Twl.rowRange(0, 3).colRange(0, 3));
    Lw.copyTo(Twl.rowRange(0, 3).col(3));

    mpCuboid->mPose = ORB_SLAM2::Converter::toSE3Quat(Twl);
}

void Landmark::SetPoseNoLock(const Mat& Rlw, const Mat& tlw)
{
    Mat Rwl = Rlw.t();
    Lw = -Rwl * tlw;

    Tlw = TFromRt(Rlw, tlw);
    Twl = TFromRt(Rwl, Lw);
}

void Landmark::SetPoseAndDimension(const g2o::Cuboid& cuboid)
{
    Twl = ORB_SLAM2::Converter::toCvMat(cuboid.mPose);
    Eigen::Vector3d scale = cuboid.mScale;
    SetDimension(Dimension3D(scale[1], scale[2], scale[0]));
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

Cuboid2D Landmark::Project(const cv::Mat& Tcw, const cv::Mat& K)
{
    Cuboid2D cuboid;
    auto centroid = Tcw.rowRange(0, 3).colRange(0, 3) * Lw + Tcw.rowRange(0, 3).col(3);
    Mat Tcl = Tcw * Twl;
    Mat Rlc = Tcl.rowRange(0, 3).colRange(0, 3).t();
    auto d1 = Rlc.col(0) * mDimension.edge13 / 2;
    auto d3 = Rlc.col(1) * mDimension.edge18 / 2;
    auto d2 = Rlc.col(2) * mDimension.edge12 / 2;

    Mat corners3D[8]{
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

Landmark::Landmark(const Cuboid2D& proposal, float proposalQuality, const Object& object,
                   KeyFrame* pKF, const cv::Mat& invK)
        :mQuality(object.conf * proposalQuality), mClassIdx(object.classIdx), mnLandmarkId(landmarkCnt++)
{
    auto mapPoints = pKF->GetMapPointMatches();

    // Use the weighted average depth as the centroid depth.
    Mat worldAvgPos = Mat::zeros(3, 1, CV_32F);
    auto centroid = proposal.GetCentroid();
    float weightSum = 0;
    for (auto mapPoint : mapPoints) {
        if (mapPoint) {
            auto pos2D = pKF->mvKeysUn[mapPoint->GetObservations()[pKF]].pt;
            if (Inside(pos2D, object.bbox)) {
                auto worldPos = mapPoint->GetWorldPos();
                float weight = exp(-Distance(centroid, pos2D));
                worldAvgPos += worldPos * weight;
                weightSum += weight;
            }
        }
    }
    worldAvgPos /= weightSum;
    Mat camCoordAvgPos = pKF->GetRotation() * worldAvgPos + pKF->GetTranslation();
    float centroidDepth = camCoordAvgPos.at<float>(2);

    // Recover pose.
    Mat centroid3D = proposal.GetCentroid3D(centroidDepth, invK);
    Mat worldCentroid = pKF->GetRotation() * centroid3D + pKF->GetTranslation();
    Mat Rlw = proposal.Rlc * pKF->GetRotation();
    Mat tlw = -Rlw * worldCentroid;
    SetPoseNoLock(Rlw, tlw);

    // Recover the dimension of the landmark with the centroid and the proposal.
    auto dimension = proposal.GetDimension3D(centroid3D, invK);
    SetDimensionNoLock(dimension);

    bboxCenter[pKF->mnFrameId] = proposal.GetCentroid();
}

const g2o::Cuboid* Landmark::GetCuboid()
{
    unique_lock<mutex> lock(mMutexPose);
    return mpCuboid;
}

}