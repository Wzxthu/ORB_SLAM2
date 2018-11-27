
#include <include/LineSegmentDetector.h>

#include "LineSegmentDetector.h"

using namespace std;
using namespace cv;

namespace ORB_SLAM2 {

LineSegmentDetector::LineSegmentDetector()
        :mpDetector(cv::createLineSegmentDetector()) { }

std::vector<LineSegment> LineSegmentDetector::Detect(const cv::Mat& imGray)
{
    Mat matSegs;
    mpDetector->detect(imGray, matSegs);
    std::vector<LineSegment> vecSegs;
    vecSegs.reserve(static_cast<unsigned long>(matSegs.rows));
    for (int i = 0; i<matSegs.rows; ++i)
        vecSegs.emplace_back(
                Point(static_cast<int>(matSegs.at<float>(i, 0)), static_cast<int>(matSegs.at<float>(i, 1))),
                Point(static_cast<int>(matSegs.at<float>(i, 2)), static_cast<int>(matSegs.at<float>(i, 3))));
    return vecSegs;
}

}