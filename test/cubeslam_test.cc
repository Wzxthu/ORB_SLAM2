#include "CubeSLAM.h"
#include "ObjectDetector.h"

using namespace std;
using namespace cv;
using namespace ORB_SLAM2;

int main()
{
    string testImgPaths[] {
            "data/cubeslam_test_example_0.jpg",
            "data/cubeslam_test_example_1.jpg",
    };

    const float alignErrWeight = 0.7, shapeErrWeight = 2.5, shapeErrThresh = 2.f;

    auto objectDetector = new ObjectDetector("Thirdparty/darknet/cfg/yolov3.cfg", "model/yolov3.weights",
                                             .45, .6, 224 * 224);
    auto lineSegDetector = new ORB_SLAM2::LineSegmentDetector();

    for (int i = 0; i < sizeof(testImgPaths) / sizeof(string); ++i) {
        auto img = imread(testImgPaths[i]);
        Mat canvas = img.clone();
        Mat imGray;
        cvtColor(img, imGray, cv::COLOR_RGB2GRAY);

        vector<Object> objects2D;
        objectDetector->Detect(img, objects2D);

        auto lineSegs = lineSegDetector->Detect(imGray);

        for (auto& seg: lineSegs) {
            line(canvas, seg.first, seg.second, Scalar(0, 0, 255), 2);
        }

        // TODO: Load camera roll, pitch, yaw and intrinsics for different test images.
        const float c_roll = -1.9151f, c_pitch = -0.0011f, c_yaw = 0;
        Mat K = (Mat_<float>(3, 3, CV_32F)
                << 529.5000, 0, 365.0000,
                0, 529.5000, 265.0000,
                0, 0, 1.0000);

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
            float bestErr;
            CuboidProposal bestProposal = FindBestProposal(bbox, c_yaw, c_roll, c_pitch,
                                                           shapeErrThresh, shapeErrWeight, alignErrWeight,
                                                           segsInBbox, K,
                                                           bestErr,
                                                           0, img);

            if (bestErr == -1)
                continue;
            {
                // Draw cuboid proposal
                line(canvas, bestProposal[0], bestProposal[1], Scalar(0, 255, 0), 1, CV_AA);
                line(canvas, bestProposal[1], bestProposal[3], Scalar(0, 255, 0), 1, CV_AA);
                line(canvas, bestProposal[3], bestProposal[2], Scalar(0, 255, 0), 1, CV_AA);
                line(canvas, bestProposal[2], bestProposal[0], Scalar(0, 255, 0), 1, CV_AA);
                line(canvas, bestProposal[0], bestProposal[7], Scalar(0, 255, 0), 1, CV_AA);
                line(canvas, bestProposal[1], bestProposal[6], Scalar(0, 255, 0), 1, CV_AA);
                line(canvas, bestProposal[2], bestProposal[5], Scalar(0, 255, 0), 1, CV_AA);
                line(canvas, bestProposal[3], bestProposal[4], Scalar(0, 255, 0), 1, CV_AA);
                line(canvas, bestProposal[7], bestProposal[6], Scalar(0, 255, 0), 1, CV_AA);
                line(canvas, bestProposal[6], bestProposal[4], Scalar(0, 255, 0), 1, CV_AA);
                line(canvas, bestProposal[4], bestProposal[5], Scalar(0, 255, 0), 1, CV_AA);
                line(canvas, bestProposal[5], bestProposal[7], Scalar(0, 255, 0), 1, CV_AA);
            }
        }

        imshow("Outputs/test_" + to_string(i) + ".jpg", canvas);
        waitKey(0);
    }

    return 0;
}