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

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2 {

class Landmark {
public:
    void SetPose(const cv::Mat& Tlw_);
    cv::Mat GetPose();
    cv::Mat GetPoseInverse();
    cv::Mat GetLandmarkCenter();
    cv::Mat GetRotation();
    cv::Mat GetTranslation();
    cv::Point GetProjectedCenter(const cv::Mat& Tcw);
public:
    int classIdx;
    int landmarkID;
private:
    // SE3 Pose and landmark center.
    cv::Mat Tlw;
    cv::Mat Twl;
    cv::Mat Lw;

    std::mutex mMutexPose;
};

}

#endif //LANDMARK_H
