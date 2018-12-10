#include "CubeSLAM.h"
#include "ObjectDetector.h"

#include <opencv2/opencv.hpp>

#include <fstream>
#include <sys/stat.h>

using namespace std;
using namespace cv;
using namespace ORB_SLAM2;

void RunEulerAngleTransformationTest();

void RunCuboidProposalGenerationTest(const Mat& img, const Rect& bbox, const Mat& K);

int main()
{
    mkdir("Outputs", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    RunEulerAngleTransformationTest();

    string testImgPaths[]{
            "data/cubeslam_test_example_0.jpg",
            "data/cubeslam_test_example_1.jpg",
            "data/cubeslam_test_example_2.png",
            "data/cubeslam_test_example_3.png",
            "data/cubeslam_test_example_4.png",
            "data/cubeslam_test_example_5.png",
    };

    string testInfoPaths[]{
            "data/cubeslam_test_example_0_info.txt",
            "data/cubeslam_test_example_1_info.txt",
            "data/cubeslam_test_example_2_info.txt",
            "data/cubeslam_test_example_3_info.txt",
            "data/cubeslam_test_example_4_info.txt",
            "data/cubeslam_test_example_5_info.txt",
    };

    const float alignErrWeight = 6, shapeErrWeight = 1, shapeErrThresh = 4.f;

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

        // Load intrinsics for different test images.
        Mat K(3, 3, CV_32F);
        ifstream fin(testInfoPaths[i]);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                fin >> K.at<float>(r, c);
        Mat invK = K.inv();
        // Load prior of roll and pitch range.
        float rollRange, pitchRange;
        fin >> rollRange >> pitchRange;
        rollRange *= M_PI / 180;
        pitchRange *= M_PI / 180;

        for (int objId = 0; objId < objects2D.size(); ++objId) {
            const auto object = objects2D[objId];
            auto& bbox = object.bbox;
            // draw bbox
            cout << bbox << endl;
            ObjectDetector::Draw(canvas, object);

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
            Cuboid2D proposal = FindBestProposal(bbox, segsInBbox, K,
                                                 shapeErrThresh, shapeErrWeight, alignErrWeight,
                                                 -M_PI_F, 0,
                                                 rollRange, pitchRange,
                                                 0, objId, img, false, false);

            if (!proposal.valid)
                continue;
            {
                Vec3f theta = EulerAnglesFromRotation(proposal.Rlc);
                cout << "Roll=" << theta[0] * 180 / M_PI_F
                     << " Yaw=" << theta[1] * 180 / M_PI_F
                     << " Pitch=" << theta[2] * 180 / M_PI_F << endl;

                // Draw cuboid proposal
                proposal.Draw(canvas, K, Scalar(128, 128, 128));
            }

            Landmark landmark;

            Mat camCoordCentroid = proposal.GetCentroid3D(100, invK);

            landmark.SetPose(proposal.Rlc, -proposal.Rlc * camCoordCentroid);

            // Recover the dimension of the landmark with the centroid and the proposal.
            auto dimension = proposal.GetDimension3D(camCoordCentroid, invK);
            landmark.SetDimension(dimension);

            cout << dimension << endl;

            auto projCuboid = landmark.Project(Mat::eye(4, 4, CV_32F), K);
            projCuboid.Draw(canvas, K);

            cout << proposal << endl;
            cout << projCuboid << endl << endl;
        }

        imshow("test_" + to_string(i), canvas);
        waitKey(0);

        imwrite("Outputs/test_" + to_string(i) + ".jpg", canvas);

        fin.close();
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
    float roll = -M_PI_F, pitch = 0, yaw = -M_PI_F;
    cout << "Roll=" << roll * 180 / M_PI_F << " Pitch=" << pitch * 180 / M_PI_F << " Yaw=" << yaw * 180 / M_PI_F << endl;
    // Recover rotation of the landmark.
    Mat Rlc = EulerAnglesToRotationMatrix(Vec3f(roll, yaw, pitch));
    Mat invRlc = Rlc.t();
    cout << Rlc << endl;
    // Compute the vanishing points from the pose.
    Mat vp1Homo = K * Rlc.col(0);
    Mat vp3Homo = K * Rlc.col(1);
    Mat vp2Homo = K * Rlc.col(2);
    Point2f vp1 = PointFrom2DHomo(vp1Homo);
    Point2f vp2 = PointFrom2DHomo(vp2Homo);
    Point2f vp3 = PointFrom2DHomo(vp3Homo);

    cout << vp1 << ' ' << vp2 << ' ' << vp3 << endl;

    auto proposal = GenerateCuboidProposal(bbox, bbox.x + bbox.width / 2, vp1, vp2, vp3);
    Mat canvas = img.clone();
    rectangle(canvas, bbox, Scalar(0, 0, 0), 2);
    if (proposal.valid) {
        proposal.Rlc = Rlc;
        proposal.Draw(canvas, K);
    }

    Point2f centroid = proposal.valid ? proposal.GetCentroid()
                                      : Point2f(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);

    line(canvas, vp1, centroid, Scalar(0, 0, 255), 2);
    line(canvas, vp2, centroid, Scalar(0, 255, 0), 2);
    line(canvas, vp3, centroid, Scalar(255, 0, 0), 2);
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
        Vec3f theta(M_PI_F, 0, 0);
        Mat R = EulerAnglesToRotationMatrix(theta);
        Mat target = (Mat_<float>(3, 3, CV_32F) << 1, 0, 0, 0, -1, 0, 0, 0, -1);
        CV_Assert(norm(R, target) < 0.001);
    }

    {
        Vec3f theta(0, M_PI_F, 0);
        Mat R = EulerAnglesToRotationMatrix(theta);
        Mat target = (Mat_<float>(3, 3, CV_32F) << -1, 0, 0, 0, 1, 0, 0, 0, -1);
        CV_Assert(norm(R, target) < 0.001);
    }

    {
        Vec3f theta(0, 0, M_PI_F);
        Mat R = EulerAnglesToRotationMatrix(theta);
        Mat target = (Mat_<float>(3, 3, CV_32F) << -1, 0, 0, 0, -1, 0, 0, 0, 1);
        CV_Assert(norm(R, target) < 0.01);
    }

    {
        Vec3f theta(0, 0, M_PI_2_F);
        Mat R = EulerAnglesToRotationMatrix(theta);
        Mat target = (Mat_<float>(3, 3, CV_32F) << 0, -1, 0, 1, 0, 0, 0, 0, 1);
        CV_Assert(norm(R, target) < 0.01);
    }

    {
        Vec3f theta(0, 0, -M_PI_2_F);
        Mat R = EulerAnglesToRotationMatrix(theta);
        Mat target = (Mat_<float>(3, 3, CV_32F) << 0, 1, 0, -1, 0, 0, 0, 0, 1);
        CV_Assert(norm(R, target) < 0.01);
    }
}