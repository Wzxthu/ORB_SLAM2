#include <mutex>
#include "Landmark.h"

using namespace std;

namespace ORB_SLAM2 {

void Landmark::SetPose(const cv::Mat& Tlw_)
{
    unique_lock<mutex> lock(mMutexPose);
    Tlw_.copyTo(Tlw);
    cv::Mat Rcw = Tlw.rowRange(0, 3).colRange(0, 3);
    cv::Mat tcw = Tlw.rowRange(0, 3).col(3);
    cv::Mat Rwc = Rcw.t();
    Lw = -Rwc*tcw;

    Twl = cv::Mat::eye(4, 4, Tlw.type());
    Rwc.copyTo(Twl.rowRange(0, 3).colRange(0, 3));
    Lw.copyTo(Twl.rowRange(0, 3).col(3));
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

}