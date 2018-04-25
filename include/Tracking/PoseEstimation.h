/**
* This file is part of CNN-SLAM.
*
* [Copyright of CNN-SLAM]
* Copyright (C) 2018
* Kai Yu <kaiy1 at andrew dot cmu dot edu> (Carnegie Mellon University)
* Zhongxu Wang <zhongxuw at andrew dot cmu dot edu> (Carnegie Mellon University)
* Manchen Wang <manchen2 at andrew dot cmu dot edu> (Carnegie Mellon University)
* For more information see <https://github.com/raulmur/CNN_SLAM>
*
* CNN-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* CNN-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with CNN-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef CNN_SLAM_POSEESTIMATOR_H
#define CNN_SLAM_POSEESTIMATOR_H
#pragma once

#include <KeyFrame.h>
#include <opencv2/opencv.hpp>
#include <util/settings.h>

namespace cnn_slam {

    float EstimateCameraPose(const cv::Mat &imColor,
                             const cv::Mat &K,
                             const cv::Mat &invK,
                             ORB_SLAM2::KeyFrame *pRefKF,
                             float cameraPixelNoise2,
                             double max_seconds,
                             cv::Mat &Tcw,
                             const cv::Mat initialTcw = cv::Mat(),
                             float *rotAngle = nullptr,
                             float *transDist = nullptr,
                             float *validRatio = nullptr);
}

#endif //CNN_SLAM_POSEESTIMATOR_H
