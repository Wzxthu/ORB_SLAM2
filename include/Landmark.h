#ifndef LANDMARK_H
#define LANDMARK_H

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2 {

class Landmark {
public:
    void SetPose(const cv::Mat& Tcw);
    cv::Mat GetPose();
    cv::Mat GetPoseInverse();
    cv::Mat GetLandmarkCenter();
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
