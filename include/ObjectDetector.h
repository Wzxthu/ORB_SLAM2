#ifndef OBJECTDETECTOR_H
#define OBJECTDETECTOR_H

#include <vector>

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2 {

struct Object {
    cv::Rect bbox;
    int classes;
    float objectness;
    int sort_class;
};

/// Based on YOLOv3 from DarkNet.
class ObjectDetector {
public:
    ObjectDetector(
            const char *cfgfile,
            const char *weightfile,
            float nms=.45,
            float thresh=.5,
            float hierThresh=.5);

    void Detect(const cv::Mat &img, std::vector<Object> &objects);

private:
    void *mpNet;
    float mNms;
    float mThresh;
    float mHierThresh;
};

}

#endif //OBJECTDETECTOR_H
