#ifndef OBJECTDETECTOR_H
#define OBJECTDETECTOR_H

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2 {

struct Object {
    cv::Rect bbox;
    float conf;
    int classIdx;

    Object(const cv::Rect& bbox, float conf, int classIdx): bbox(bbox), conf(conf), classIdx(classIdx) {}
};

/// Based on YOLOv3 from DarkNet.
class ObjectDetector {
public:
    ObjectDetector(
            const char *cfgFile,
            const char *weightFile,
            float nmsThresh=.45,
            float thresh=.5);

    void Detect(const cv::Mat &im, std::vector<Object> &objects);

private:
    // Remove the bounding boxes with low confidence using non-maxima suppression
    void Postprocess(const cv::Mat& im, const std::vector<cv::Mat>& outs, std::vector<Object>& objects);

private:
    cv::dnn::Net mNet;
    cv::Mat mBlob;
    float mNmsThresh;
    float mConfThresh;

    const int mInputWidth = 416;        // Width of network's input image
    const int mInputHeight = 416;       // Height of network's input image

    std::vector<cv::String> mOutputNames;
};

}

#endif //OBJECTDETECTOR_H
