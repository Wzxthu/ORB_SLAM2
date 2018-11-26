
#include <include/ObjectDetector.h>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace cv::dnn;
using namespace std;

namespace ORB_SLAM2 {

ObjectDetector::ObjectDetector(
        const char* cfgfile,
        const char* weightfile,
        float nmsThresh,
        float confThresh,
        float hierThresh)
        :
        mNet(readNetFromDarknet(cfgfile, weightfile)),
        mNmsThresh(nmsThresh), mConfThresh(confThresh), mHierThresh(hierThresh)
{
    mNet.setPreferableBackend(DNN_BACKEND_OPENCV);
    mNet.setPreferableTarget(DNN_TARGET_OPENCL);

    //Get the indices of the output layers, i.e. the layers with unconnected outputs
    vector<int> outLayers = mNet.getUnconnectedOutLayers();

    //get the names of all the layers in the network
    vector<String> layersNames = mNet.getLayerNames();

    // Get the names of the output layers in names
    mOutputNames.resize(outLayers.size());
    for (size_t i = 0; i<outLayers.size(); ++i)
        mOutputNames[i] = layersNames[outLayers[i]-1];
}

void ObjectDetector::Detect(const cv::Mat& im, std::vector<Object>& objects)
{
    // Create a 4D blob from a frame.
    blobFromImage(im, blob, 1/255.0, cvSize(mInputWidth, mInputHeight), Scalar(0, 0, 0), true, false);

    //Sets the input to the network
    mNet.setInput(blob);

    // Runs the forward pass to get output of the output layers
    vector<Mat> outs;
    mNet.forward(outs, mOutputNames);

    // Remove the bounding boxes with low confidence
    Postprocess(im, outs, objects);
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void ObjectDetector::Postprocess(const Mat& im, const vector<Mat>& outs, std::vector<Object>& objects)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (const auto& out : outs) {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        auto* data = (float*) out.data;
        for (int j = 0; j<out.rows; ++j, data += out.cols) {
            Mat scores = out.row(j).colRange(5, out.cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence>mConfThresh) {
                int centerX = (int) (data[0]*im.cols);
                int centerY = (int) (data[1]*im.rows);
                int width = (int) (data[2]*im.cols);
                int height = (int) (data[3]*im.rows);
                int left = centerX-width/2;
                int top = centerY-height/2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float) confidence);
                boxes.emplace_back(left, top, width, height);
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, mConfThresh, mNmsThresh, indices);

    objects.clear();
    objects.reserve(indices.size());
    for (auto idx: indices)
        objects.emplace_back(boxes[idx], classIds[idx], confidences[idx]);
}

}