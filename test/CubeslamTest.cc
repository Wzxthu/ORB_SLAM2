#include "CubeSLAM.h"
#include "ObjectDetector.h"

#include <fstream>

using namespace std;
using namespace cv;
using namespace ORB_SLAM2;

int main()
{
    string testImgPaths[] {
            "data/cubeslam_test_example_0.jpg",
            "data/cubeslam_test_example_1.jpg",
            "data/cubeslam_test_example_2.png",
            "data/cubeslam_test_example_3.png",
    };

    string testInfoPaths[] {
            "data/cubeslam_test_example_0_info.txt",
            "data/cubeslam_test_example_1_info.txt",
            "data/cubeslam_test_example_2_info.txt",
            "data/cubeslam_test_example_3_info.txt",
    };

    const float alignErrWeight = 0.7, shapeErrWeight = 2.5, shapeErrThresh = 2.f;

    auto objectDetector = new ObjectDetector("Thirdparty/darknet/cfg/yolov3.cfg", "model/yolov3.weights",
                                             .45, .6, 224 * 224);
    auto lineSegDetector = new ORB_SLAM2::LineSegmentDetector();

    using namespace chrono;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    for (int i = 0; i < sizeof(testImgPaths) / sizeof(string); ++i) {
        auto img = imread(testImgPaths[i]);
        Mat canvas = img.clone();
        Mat imGray;
        cvtColor(img, imGray, cv::COLOR_RGB2GRAY);

        vector<Object> objects2D;
        objectDetector->Detect(img, objects2D);

        auto lineSegs = lineSegDetector->Detect(imGray);

        for (auto& seg: lineSegs) {
            line(canvas, seg.first, seg.second, Scalar(0, 0, 255), 1);
        }

        // TODO: Load camera roll, pitch, yaw and intrinsics for different test images.
//        float c_roll, c_pitch, c_yaw;
        Mat K(3, 3, CV_32F);
        ifstream fin(testInfoPaths[i]);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                fin >> K.at<float>(r, c);
//        fin >> c_roll >> c_pitch >> c_yaw;

        for (auto& object : objects2D) {
            auto& bbox = object.bbox;
            // draw bbox
            ObjectDetector::DrawPred(object, canvas);

            // Ignore the bounding box that goes outside the frame.
            if (bbox.x < 0 || bbox.y < 0
                || bbox.x + bbox.width >= img.cols
                || bbox.y + bbox.height >= img.rows) {
                continue;
            }

            // Choose the line segments lying in the bounding box for scoring.
            vector<LineSegment*> segsInBbox;
            segsInBbox.reserve(lineSegs.size());
            for (auto& lineSeg : lineSegs) {
                if (lineSeg.first.inside(bbox) && lineSeg.second.inside(bbox)) {
                    segsInBbox.emplace_back(&lineSeg);
                }
            }

            // Find landmarks with respect to the detected objects.
            Mat bestRlw, bestInvRlw;
            CuboidProposal bestProposal = FindBestProposal(bbox, segsInBbox, K,
                                                           shapeErrThresh, shapeErrWeight, alignErrWeight,
                                                           0, 0, 0,
                                                           0, img, false);

            if (bestProposal.Rlc.empty())
                continue;
            {
                Vec3f theta = EulerAnglesFromRotation(bestProposal.Rlc);
                cout << object.conf << endl;
                cout << "Roll=" << theta[0] * 180 / M_PI << " Pitch=" << theta[1] * 180 / M_PI << " Yaw=" << theta[2] * 180 / M_PI << endl;

                // Draw cuboid proposal
                DrawProposal(canvas, bestProposal);
            }
        }

        imshow("test_" + to_string(i), canvas);
        waitKey(0);
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> timeSpan = duration_cast<duration<double>>(t2 - t1);
    cout << "Finding landmarks took " << timeSpan.count() << " seconds." << endl;

    return 0;
}