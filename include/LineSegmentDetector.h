#ifndef LINESEGMENTDETECTOR_H
#define LINESEGMENTDETECTOR_H

#include <vector>

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2 {

typedef std::pair<cv::Point, cv::Point> LineSegment;

class LineSegmentDetector {
public:
    LineSegmentDetector();

    std::vector<LineSegment> Detect(const cv::Mat& imGray);
private:
    cv::Ptr<cv::LineSegmentDetector> mpDetector;
};

}

#endif //LINESEGMENTDETECTOR_H
