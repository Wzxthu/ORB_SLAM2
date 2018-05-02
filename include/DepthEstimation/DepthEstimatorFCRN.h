//
// Created by kaiy1 on 4/30/18.
//

#ifndef ORB_SLAM2_DEPTHESTIMATORFCRN_H
#define ORB_SLAM2_DEPTHESTIMATORFCRN_H

#pragma once

#include <python2.7/Python.h>
#include <cstring>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <mutex>
#include "DepthEstimator.h"

namespace cnn_slam {
    class DepthEstimatorFCRN: public DepthEstimator {
    public:
        void Initialize();
        DepthEstimatorFCRN();
        void EstimateDepth(const cv::Mat& im, cv::Mat& depth, float focalLength);
        ~DepthEstimatorFCRN();

    private:
        const char *mModelPath;
        int mHeight, mWidth;
        PyObject *mpModule, *mpFunc, *mpInstance;
        float mTrainingFocalLength;
        float mDepthRatio;
        bool mInitialized;
        std::mutex mForwardMutex;
    };
}

#endif //ORB_SLAM2_DEPTHESTIMATORFCRN_H
