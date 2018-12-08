#include "CubeSLAM.h"
#include "ObjectDetector.h"

#include <opencv2/opencv.hpp>

#include <fstream>

using namespace std;
using namespace cv;
using namespace ORB_SLAM2;

void RunEulerAngleTransformationTest();

void RunCuboidProposalGenerationTest(const Mat& img, const Rect& bbox, const Mat& K);

int main()
{
    RunEulerAngleTransformationTest();

    string testImgPaths[]{
            "data/cubeslam_test_example_0.jpg",
            "data/cubeslam_test_example_1.jpg",
            "data/cubeslam_test_example_2.png",
            "data/cubeslam_test_example_3.png",
    };

    string testInfoPaths[]{
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
        Mat K(3, 3, CV_32F);
        ifstream fin(testInfoPaths[i]);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                fin >> K.at<float>(r, c);

        for (int objId = 0; objId < objects2D.size(); ++objId) {
            const auto object = objects2D[objId];
            auto& bbox = object.bbox;
            // draw bbox
            cout << bbox << endl;
            ObjectDetector::DrawPred(canvas, object);

//            RunCuboidProposalGenerationTest(img, bbox, K);

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
                                                           -M_PI, 0, objId,
                                                           0, img, false);

            if (!bestProposal.valid)
                continue;
            {
                Vec3f theta = EulerAnglesFromRotation(bestProposal.Rlc);
                cout << "Roll=" << theta[0] * 180 / M_PI << " Pitch=" << theta[1] * 180 / M_PI << " Yaw="
                     << theta[2] * 180 / M_PI << endl;

                // Draw cuboid proposal
                DrawCuboidProposal(canvas, bestProposal, bbox, K);
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

void RunCuboidProposalGenerationTest(const Mat& img, const Rect& bbox, const Mat& K)
{
    cout << "Bounding box: " << bbox << endl;
//    float roll = -M_PI - 6 * M_PI / 180, pitch = -16 * M_PI / 180, yaw = -M_PI+2 * M_PI / 180;
    float roll = -M_PI, pitch = 0, yaw = -M_PI;
    cout << "Roll=" << roll * 180 / M_PI << " Pitch=" << pitch * 180 / M_PI << " Yaw=" << yaw * 180 / M_PI << endl;
    // Recover rotation of the landmark.
    Mat Rlc = EulerAnglesToRotationMatrix(Vec3f(roll, pitch, yaw));
    Mat invRlc = Rlc.t();
    cout << Rlc << endl;
    // Compute the vanishing points from the pose.
    Mat vp1Homo = K * Rlc.col(0);
    Mat vp3Homo = K * Rlc.col(1);
    Mat vp2Homo = K * Rlc.col(2);
    Point2f vp1 = Point2FromHomo(vp1Homo);
    Point2f vp2 = Point2FromHomo(vp2Homo);
    Point2f vp3 = Point2FromHomo(vp3Homo);

    cout << vp1 << ' ' << vp2 << ' '<< vp3 << endl;

    auto proposal = GenerateCuboidProposal(bbox, bbox.x + bbox.width / 2, vp1, vp2, vp3);
    Mat canvas = img.clone();
    rectangle(canvas, bbox, Scalar(0, 0, 0), 2);
    if (proposal.valid)
        DrawCuboidProposal(canvas, proposal, bbox, K);
    line(canvas, vp1, proposal.corners[0], Scalar(255, 0, 0));
    line(canvas, vp2, proposal.corners[0], Scalar(255, 0, 0));
    line(canvas, vp1, Point(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2), Scalar(0, 0, 255), 2);
    line(canvas, vp2, Point(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2), Scalar(0, 255, 0), 2);
    line(canvas, vp3, Point(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2), Scalar(255, 0, 0), 2);
    imshow("Cuboid Proposal Generation Test", canvas);
    waitKey(0);
}

void RunEulerAngleTransformationTest()
{
    {
        Vec3f theta(0, 0, 0);
        Mat R = EulerAnglesToRotationMatrix(theta);
        CV_Assert(norm(R, Mat::eye(3, 3, CV_32F)) < 0.001);
    }

    {
        Vec3f theta(M_PI, 0, 0);
        Mat R = EulerAnglesToRotationMatrix(theta);
        Mat target = (Mat_<float>(3, 3, CV_32F) << 1, 0, 0, 0, -1, 0, 0, 0, -1);
        CV_Assert(norm(R, target) < 0.001);
    }

    {
        Vec3f theta(0, M_PI, 0);
        Mat R = EulerAnglesToRotationMatrix(theta);
        Mat target = (Mat_<float>(3, 3, CV_32F) << -1, 0, 0, 0, 1, 0, 0, 0, -1);
        CV_Assert(norm(R, target) < 0.001);
    }

    {
        Vec3f theta(0, 0, M_PI);
        Mat R = EulerAnglesToRotationMatrix(theta);
        Mat target = (Mat_<float>(3, 3, CV_32F) << -1, 0, 0, 0, -1, 0, 0, 0, 1);
        CV_Assert(norm(R, target) < 0.01);
    }

    {
        Vec3f theta(0, 0, M_PI_2);
        Mat R = EulerAnglesToRotationMatrix(theta);
        Mat target = (Mat_<float>(3, 3, CV_32F) << 0, -1, 0, 1, 0, 0, 0, 0, 1);
        CV_Assert(norm(R, target) < 0.01);
    }

    {
        Vec3f theta(0, 0, -M_PI_2);
        Mat R = EulerAnglesToRotationMatrix(theta);
        Mat target = (Mat_<float>(3, 3, CV_32F) << 0, 1, 0, -1, 0, 0, 0, 0, 1);
        CV_Assert(norm(R, target) < 0.01);
    }
}