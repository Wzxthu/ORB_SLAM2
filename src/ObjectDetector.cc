/**
 * This file is part of CubeSLAM.
 *
 * Copyright (C) 2018, Carnegie Mellon University
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <include/ObjectDetector.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

#include <OpenCL/opencl.h>

using namespace cv;
using namespace cv::dnn;
using namespace std;

namespace ORB_SLAM2 {

std::vector<std::string> ObjectDetector::mClasses;

ObjectDetector::ObjectDetector(
        const char* cfgFile,
        const char* weightFile,
        float nmsThresh,
        float confThresh,
        float inputArea)
        :
        mCfgFile(cfgFile), mWeightFile(weightFile),
        mNet(readNetFromDarknet(cfgFile, weightFile)),
        mNmsThresh(nmsThresh), mConfThresh(confThresh), mInputArea(inputArea),
        mInputWidth(0), mInputHeight(0)
{
    ocl::Context context = ocl::Context::getDefault(true);

    vector<ocl::PlatformInfo> platforms;
    ocl::getPlatfomsInfo(platforms);
    for (auto& platform : platforms) {
        //Platform Name
        cout << "Platform Name: " << platform.name().c_str() << "\n" << endl;

        //Access Device within Platform
        ocl::Device currentDevice;
        for (int j = 0; j < platform.deviceNumber(); j++) {
            //Access Device
            platform.getDevice(currentDevice, j);
            int deviceType = currentDevice.type();
            cout << "Device name:  " << currentDevice.name() << endl;
            if (deviceType == 2)
                cout << context.ndevices() << " CPU devices are detected." << endl;
            if (deviceType == 4)
                cout << context.ndevices() << " GPU devices are detected." << endl;
            cout << "===============================================" << endl << endl;
        }
    }

    ocl::Device device;
    string deviceName;
    for (int i = 0; i < context.ndevices(); i++) {
        device = context.device(i);
        deviceName = device.name();
        cout << "Using device: " << deviceName << endl;
    }

    mNet.setPreferableBackend(DNN_BACKEND_OPENCV);
    mNet.setPreferableTarget(DNN_TARGET_OPENCL);

    //Get the indices of the output layers, i.e. the layers with unconnected outputs
    vector<int> outLayers = mNet.getUnconnectedOutLayers();

    //get the names of all the layers in the network
    vector<String> layersNames = mNet.getLayerNames();

    // Get the names of the output layers in names
    mOutputNames.resize(outLayers.size());
    cout << "YOLOv3 outputs from layers:" << endl;
    for (size_t i = 0; i < outLayers.size(); ++i) {
        mOutputNames[i] = layersNames[outLayers[i] - 1];
        cout << "\t" << mOutputNames[i] << endl;
    }
}

void ObjectDetector::Draw(Mat& frame, const Object& obj)
{
    DrawPred(frame, obj.bbox, obj.classIdx, obj.conf);
}

void ObjectDetector::DrawPred(Mat& frame, cv::Rect bbox, int classId, float conf)
{
    DrawPred(frame, bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height, classId, conf);
}

// Draw the predicted bounding box
void ObjectDetector::DrawPred(Mat& frame, int left, int top, int right, int bottom, int classId, float conf)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 0), 3);
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 255, 255), 1);

    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (mClasses.empty()) {
        LoadClassNames();
    }
    if (!mClasses.empty()) {
        CV_Assert(classId < (int) mClasses.size());
        label = mClasses[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 4);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
}

std::vector<std::string> ObjectDetector::LoadClassNames()
{
    // Load names of classes
    string classesFile = "Thirdparty/darknet/data/coco.names";
    ifstream ifs(classesFile.c_str());
    string line;
    mClasses.clear();
    while (getline(ifs, line))
        mClasses.push_back(line);
    return mClasses;
}

void ObjectDetector::Detect(const cv::Mat& im, vector<Object>& objects)
{
    // Compute the input size to fit the preset input area. Both width and height should be times of 32.
    float resizeRatio = sqrtf(mInputArea / (im.cols * im.rows));
    int inputWidth = static_cast<int>(ceil(im.cols * resizeRatio / 32)) << 5;
    int inputHeight = static_cast<int>(ceil(im.rows * resizeRatio / 32)) << 5;

    if (mInputWidth) {
        if (mInputWidth != inputWidth || mInputHeight != inputHeight) {
            cerr << "WARNING: Image input to the network changes."
                    " Due to the current implementation of OpenCV, the network needs reinitializing." << endl;
            mNet = readNetFromDarknet(mCfgFile, mWeightFile);
            mInputWidth = inputWidth;
            mInputHeight = inputHeight;
        }
    } else {
        mInputWidth = inputWidth;
        mInputHeight = inputHeight;
    }

    // Create a 4D blob from the frame.
    Mat blob = blobFromImage(im, 1 / 255.0, cvSize(inputWidth, inputHeight), Scalar(0, 0, 0), true, false);

    //Sets the input to the network
    mNet.setInput(blob);
    cout << blob.size << endl;

    // Runs the forward pass to get output of the output layers
    vector<Mat> outs;
    mNet.forward(outs, mOutputNames);

    // Remove the bounding boxes with low confidence
    Postprocess(im, outs, objects, inputWidth, inputHeight);
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void ObjectDetector::Postprocess(const Mat& im, const vector<Mat>& outs, vector<Object>& objects,
                                 int inputWidth, int inputHeight)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (const auto& out : outs) {
        float widthRatio = float(im.cols * inputHeight) / inputWidth;
        float heightRatio = float(im.rows * inputWidth) / inputHeight;

        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        auto* data = (float*) out.data;
        for (int j = 0; j < out.rows; ++j, data += out.cols) {
            Mat scores = out.row(j).colRange(5, out.cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, nullptr, &confidence, nullptr, &classIdPoint);
            if (confidence > mConfThresh) {
                int centerX = (int) (data[0] * im.cols);
                int centerY = (int) (data[1] * im.rows);
                int width = (int) (data[2] * widthRatio);
                int height = (int) (data[3] * heightRatio);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

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
        objects.emplace_back(boxes[idx], confidences[idx], classIds[idx]);
}

}