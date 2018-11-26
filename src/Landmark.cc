#include <mutex>
#include "Landmark.h"

using namespace std;

namespace ORB_SLAM2 {

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