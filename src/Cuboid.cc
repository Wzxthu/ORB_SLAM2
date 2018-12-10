#include "Cuboid.h"

using namespace cv;
using namespace std;

namespace ORB_SLAM2 {

cv::Point2f Cuboid2D::GetCentroid() const
{
    if (!IsParallel(corners[2], corners[6], corners[1], corners[5]))
        return LineIntersection(corners[2], corners[6], corners[1], corners[5]);
    else
        return LineIntersection(corners[0], corners[4], corners[3], corners[7]);
}

void Cuboid2D::Draw(Mat& canvas, const Mat& K, const Scalar& edgeColor) const
{
    Mat vp1Homo = K * Rlc.col(0);
    Mat vp3Homo = K * Rlc.col(1);
    Mat vp2Homo = K * Rlc.col(2);
    Point2f vp1 = PointFrom2DHomo(vp1Homo);
    Point2f vp2 = PointFrom2DHomo(vp2Homo);
    Point2f vp3 = PointFrom2DHomo(vp3Homo);

    line(canvas, corners[0], corners[1], edgeColor, 1 + (isCornerVisible[0] && isCornerVisible[1]), CV_AA);
    line(canvas, corners[1], corners[3], edgeColor, 1 + (isCornerVisible[1] && isCornerVisible[3]), CV_AA);
    line(canvas, corners[3], corners[2], edgeColor, 1 + (isCornerVisible[3] && isCornerVisible[2]), CV_AA);
    line(canvas, corners[2], corners[0], edgeColor, 1 + (isCornerVisible[2] && isCornerVisible[0]), CV_AA);
    line(canvas, corners[0], corners[7], edgeColor, 1 + (isCornerVisible[0] && isCornerVisible[7]), CV_AA);
    line(canvas, corners[1], corners[6], edgeColor, 1 + (isCornerVisible[1] && isCornerVisible[6]), CV_AA);
    line(canvas, corners[2], corners[5], edgeColor, 1 + (isCornerVisible[2] && isCornerVisible[5]), CV_AA);
    line(canvas, corners[3], corners[4], edgeColor, 1 + (isCornerVisible[3] && isCornerVisible[4]), CV_AA);
    line(canvas, corners[7], corners[6], edgeColor, 1 + (isCornerVisible[7] && isCornerVisible[6]), CV_AA);
    line(canvas, corners[6], corners[4], edgeColor, 1 + (isCornerVisible[6] && isCornerVisible[4]), CV_AA);
    line(canvas, corners[4], corners[5], edgeColor, 1 + (isCornerVisible[4] && isCornerVisible[5]), CV_AA);
    line(canvas, corners[5], corners[7], edgeColor, 1 + (isCornerVisible[5] && isCornerVisible[7]), CV_AA);

    auto centroid = GetCentroid();
    line(canvas, vp1, centroid, Scalar(0, 0, 255), 4);
    line(canvas, vp2, centroid, Scalar(0, 255, 0), 4);
    line(canvas, vp3, centroid, Scalar(255, 0, 0), 4);

    circle(canvas, vp1, 4, Scalar(0, 0, 255), 2);
    circle(canvas, vp2, 4, Scalar(0, 255, 0), 2);
    circle(canvas, vp3, 4, Scalar(255, 0, 0), 2);

//    line(canvas, vp1, proposal.corners[0], Scalar(0, 0, 255));
//    line(canvas, vp1, proposal.corners[2], Scalar(0, 0, 255));
//    line(canvas, vp1, proposal.corners[5], Scalar(0, 0, 255));
//    line(canvas, vp1, proposal.corners[7], Scalar(0, 0, 255));
//    line(canvas, vp2, proposal.corners[0], Scalar(0, 255, 0));
//    line(canvas, vp2, proposal.corners[1], Scalar(0, 255, 0));
//    line(canvas, vp2, proposal.corners[6], Scalar(0, 255, 0));
//    line(canvas, vp2, proposal.corners[7], Scalar(0, 255, 0));
//    line(canvas, vp3, proposal.corners[4], Scalar(255, 0, 0));
//    line(canvas, vp3, proposal.corners[5], Scalar(255, 0, 0));
//    line(canvas, vp3, proposal.corners[6], Scalar(255, 0, 0));
//    line(canvas, vp3, proposal.corners[7], Scalar(255, 0, 0));

    if (isCornerVisible[0] && isCornerVisible[1])
        line(canvas, corners[0], corners[1], edgeColor, 1 + (isCornerVisible[0] && isCornerVisible[1]), CV_AA);
    if (isCornerVisible[1] && isCornerVisible[3])
        line(canvas, corners[1], corners[3], edgeColor, 1 + (isCornerVisible[1] && isCornerVisible[3]), CV_AA);
    if (isCornerVisible[3] && isCornerVisible[2])
        line(canvas, corners[3], corners[2], edgeColor, 1 + (isCornerVisible[3] && isCornerVisible[2]), CV_AA);
    if (isCornerVisible[2] && isCornerVisible[0])
        line(canvas, corners[2], corners[0], edgeColor, 1 + (isCornerVisible[2] && isCornerVisible[0]), CV_AA);
    if (isCornerVisible[0] && isCornerVisible[7])
        line(canvas, corners[0], corners[7], edgeColor, 1 + (isCornerVisible[0] && isCornerVisible[7]), CV_AA);
    if (isCornerVisible[1] && isCornerVisible[6])
        line(canvas, corners[1], corners[6], edgeColor, 1 + (isCornerVisible[1] && isCornerVisible[6]), CV_AA);
    if (isCornerVisible[2] && isCornerVisible[5])
        line(canvas, corners[2], corners[5], edgeColor, 1 + (isCornerVisible[2] && isCornerVisible[5]), CV_AA);
    if (isCornerVisible[3] && isCornerVisible[4])
        line(canvas, corners[3], corners[4], edgeColor, 1 + (isCornerVisible[3] && isCornerVisible[4]), CV_AA);
    if (isCornerVisible[7] && isCornerVisible[6])
        line(canvas, corners[7], corners[6], edgeColor, 1 + (isCornerVisible[7] && isCornerVisible[6]), CV_AA);
    if (isCornerVisible[6] && isCornerVisible[4])
        line(canvas, corners[6], corners[4], edgeColor, 1 + (isCornerVisible[6] && isCornerVisible[4]), CV_AA);
    if (isCornerVisible[4] && isCornerVisible[5])
        line(canvas, corners[4], corners[5], edgeColor, 1 + (isCornerVisible[4] && isCornerVisible[5]), CV_AA);
    if (isCornerVisible[5] && isCornerVisible[7])
        line(canvas, corners[5], corners[7], edgeColor, 1 + (isCornerVisible[5] && isCornerVisible[7]), CV_AA);

    for (int i = 7; i >= 0; --i) {
        putText(canvas, to_string(i + 1), corners[i], FONT_HERSHEY_SIMPLEX, 0.5f, Scalar(0, 0, 0), 8);
        putText(canvas, to_string(i + 1), corners[i], FONT_HERSHEY_SIMPLEX, 0.5f, Scalar(255, 255, 255), 2);
    }
}

}