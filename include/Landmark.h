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

#ifndef LANDMARK_H
#define LANDMARK_H

#include "Cuboid.h"

namespace ORB_SLAM2 {

struct LandmarkDimension {
    float edge13{};   // Corresponding to vanishing point 1.
    float edge12{};   // Corresponding to vanishing point 2.
    float edge18{};   // Corresponding to vanishing point 3.

    inline LandmarkDimension() = default;
    LandmarkDimension(float edge13_, float edge12_, float edge18_)
            :edge13(edge13_), edge12(edge12_), edge18(edge18_) { }

    friend std::ostream& operator<<(std::ostream& out, const LandmarkDimension& dim);
};

inline std::ostream& operator<<(std::ostream& out, const LandmarkDimension& dim)
{
    out << '[' << dim.edge18 << 'x' << dim.edge12 << 'x' << dim.edge13 << ']';
    return out;
}

class Landmark {
public:
    Landmark();
    Landmark(Landmark& other);

    void SetDimension(const LandmarkDimension& dimension);
    LandmarkDimension GetDimension();
    void SetPose(const cv::Mat& Tlw_);
    void SetPose(const cv::Mat& Rlw, const cv::Mat& tlw);
    cv::Mat GetPose();
    cv::Mat GetPoseInverse();
    cv::Mat GetCentroid();
    cv::Mat GetRotation();
    cv::Mat GetTranslation();
    cv::Point2f GetProjectedCentroid(const cv::Mat& Tcw);
    std::unordered_map<int, cv::Point2f> bboxCenter;
    Cuboid2D Project(const cv::Mat& Tcw, const cv::Mat& K);
public:
    int classIdx;
    int landmarkID;
private:
    // SE3 Pose.
    cv::Mat Tlw;
    cv::Mat Twl;
    // Landmark centroid.
    cv::Mat Lw;
    // Landmark dimension.
    LandmarkDimension mDimension;

    std::mutex mMutexPose;
};

inline cv::Mat TFromRt(const cv::Mat& R, const cv::Mat& t)
{
    cv::Mat T = cv::Mat::eye(4, 4, R.type());
    R.copyTo(T.rowRange(0, 3).colRange(0, 3));
    t.copyTo(T.col(3).rowRange(0, 3));
    return T;
}

}

#endif //LANDMARK_H
