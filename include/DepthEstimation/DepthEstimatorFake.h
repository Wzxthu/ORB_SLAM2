//
// Created by kaiy1 on 4/30/18.
//

#ifndef ORB_SLAM2_DEPTHESTIMATORFAKE_H
#define ORB_SLAM2_DEPTHESTIMATORFAKE_H

#pragma once

#include <python2.7/Python.h>
#include <cstring>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <mutex>
#include "DepthEstimator.h"

namespace cnn_slam {
    class DepthEstimatorFake : public DepthEstimator {
    public:
        inline void Initialize() {}

        inline void EstimateDepth(const cv::Mat &im, cv::Mat &depth, float focalLength) {
            depth = mImDepth;
        }

        inline void SetDepthMap(cv::Mat imDepth) {
            mImDepth = imDepth;
        }

    private:
        cv::Mat mImDepth;
    };
}

#endif //ORB_SLAM2_DEPTHESTIMATORFAKE_H
