#ifndef CUBESLAM_H
#define CUBESLAM_H

#include "LineSegmentDetector.h"

#include <array>
#include <opencv2/opencv.hpp>

namespace ORB_SLAM2 {

// Represent the cuboid proposal with the coordinates in frame of the 8 corners.
struct CuboidProposal {
    cv::Mat Rlc;
    cv::Point2f corners[8];
    bool isCornerVisible[8]{true, true, true, true};

    friend std::ostream& operator<<(std::ostream& out, const CuboidProposal& proposal);

    inline CuboidProposal() = default;

    inline CuboidProposal(const CuboidProposal& other)
    {
        Rlc = other.Rlc.clone();
        memcpy(corners, other.corners, sizeof(corners));
        memcpy(isCornerVisible, other.isCornerVisible, sizeof(isCornerVisible));
    }
};

inline std::ostream& operator<<(std::ostream& out, const CuboidProposal& proposal)
{
    out << '[';
    for (int i = 0; i < 7; ++i)
        out << proposal.corners[i] << ',';
    out << proposal.corners[7] << ']';
    return out;
}

inline cv::Point2f Point2FromHomo(const cv::Mat& homo)
{
    const float X = homo.at<float>(0, 0);
    const float Y = homo.at<float>(1, 0);
    const float Z = homo.at<float>(2, 0);
    const float absZ = fabs(Z);
//    std::cout << X << ' ' << Y << ' ' << Z << ' ' << X / Z << ' ' << Y / Z << "?????" << std::endl;
    if (absZ >= 1)
        return cv::Point2f(X / Z, Y / Z);
    const float maxAbsXY = std::max(fabs(homo.at<float>(0)), fabs(homo.at<float>(1)));
    if (maxAbsXY < FLT_MAX * absZ)
        return cv::Point2f(X / Z, Y / Z);
//    std::cout << maxAbsXY << ' ' << absZ << ' ' << FLT_MAX << "!!!!!" << std::endl;
    if (fabs(X) > fabs(Y)) {
        const float x = X > 0 ? FLT_MAX : -FLT_MIN;
        const float y = x * (Y / X);
        return cv::Point2f(x, y);
    }
    else {
        const float y = Y > 0 ? FLT_MAX : -FLT_MIN;
        const float x = y * (X / Y);
        return cv::Point2f(x, y);
    }
}

template<class T>
inline float DistanceSquare(const cv::Point_<T>& pt1, const cv::Point_<T>& pt2)
{
    return powf(pt1.x - pt2.x, 2) + powf(pt1.y - pt2.y, 2);
}

template<class T1, class T2>
inline bool inside(const cv::Point_<T1>& pt, const cv::Rect_<T2>& bbox)
{
    return pt.x >= bbox.x && pt.x <= bbox.x + bbox.width && pt.y >= bbox.y && pt.y <= bbox.y + bbox.height;
}

template<class T>
inline float Distance(const cv::Point_<T>& pt1, const cv::Point_<T>& pt2)
{
    return sqrtf(DistanceSquare(pt1, pt2));
}

inline float Distance(const cv::Point2f& pt, const LineSegment& edge)
{
    cv::Vec2f v1(edge.first.x - pt.x, edge.first.y - pt.y);
    cv::Vec2f v2(edge.second.x - pt.x, edge.second.y - pt.y);
    cv::Vec2f v3(edge.second.x - edge.first.x, edge.second.y - edge.first.y);
    auto l1sq = DistanceSquare(edge.first, pt);
    auto l2sq = DistanceSquare(edge.second, pt);
    auto l3sq = DistanceSquare(edge.first, edge.second);
    if (l1sq + l3sq < l2sq)
        return sqrtf(l1sq);
    else if (l2sq + l3sq < l1sq)
        return sqrtf(l2sq);
    else {
        // The pedal falls on the edge.
        float l1 = sqrtf(l1sq);
        float l2 = sqrtf(l2sq);
        float l3 = sqrtf(l3sq);
        float l1l2 = l1 * l2;
        float cosine = (l1sq + l2sq - l3sq) / (2 * l1l2);
        float sine = sqrtf(1 - cosine * cosine);
        float h = l1l2 * sine / l3;
        return h;
    }
}

inline float ChamferDist(const LineSegment& hypothesis,
                         const std::vector<LineSegment*>& actualEdges,
                         int numSamples = 10)
{
    float dx = (hypothesis.second.x - hypothesis.first.x) / (numSamples - 1);
    float dy = (hypothesis.second.y - hypothesis.first.y) / (numSamples - 1);
    float x = hypothesis.first.x;
    float y = hypothesis.first.y;
    float chamferDist = 0;
    for (int i = 0; i < numSamples; ++i) {
        cv::Point2f pt(x, y);
        float smallest = -1;
        for (const auto& edge : actualEdges) {
            float dist = Distance(pt, *edge);
            if (smallest == -1 || dist < smallest) {
                smallest = dist;
            }
        }
        chamferDist += smallest;

        x += dx;
        y += dy;
    }
    return chamferDist;
}

inline std::vector<std::vector<float>> PrecomputeChamferDistMap(const cv::Rect& bbox,
                                                                const std::vector<LineSegment*>& edges)
{
    std::vector<std::vector<float>> distMap(static_cast<unsigned long>(bbox.height + 1),
                                            std::vector<float>(static_cast<unsigned long>(bbox.width + 1)));
    int longerSide = std::max(bbox.width, bbox.height);
    for (int y = 0; y <= bbox.height; ++y) {
        for (int x = 0; x <= bbox.width; ++x) {
            cv::Point2f pt(x + bbox.x, y + bbox.y);
            float& minDist = distMap[y][x];
            minDist = longerSide;
            for (auto edge : edges) {
                float dist = Distance(pt, *edge);
                if (dist < minDist)
                    minDist = dist;
            }
        }
    }
    return distMap;
}

inline float ChamferDist(const LineSegment& hypothesis,
                         const cv::Rect& bbox,
                         const std::vector<std::vector<float>>& distMap,
                         int numSamples = 10)
{
    float dx = (hypothesis.second.x - hypothesis.first.x) / (numSamples - 1);
    float dy = (hypothesis.second.y - hypothesis.first.y) / (numSamples - 1);
    float x = hypothesis.first.x - bbox.x;
    float y = hypothesis.first.y - bbox.y;
    float chamferDist = 0;
    for (int i = 0; i < numSamples; ++i) {
        chamferDist += distMap[int(y)][int(x)];
        x += dx;
        y += dy;
    }
    return chamferDist;
}

template<class T1, class T2>
inline cv::Point_<T1> LineIntersectionX(const cv::Point_<T1>& A, const cv::Point_<T1>& B, T2 x)
{
    return cv::Point_<T1>(x, A.y + (B.y - A.y) * (x - A.x) / (B.x - A.x));
}

template<class T1, class T2>
inline cv::Point_<T1> LineIntersectionY(const cv::Point_<T1>& A, const cv::Point_<T1>& B, T2 y)
{
    return cv::Point_<T1>(A.x + (B.x - A.x) * (y - A.y) / (B.y - A.y), y);
}

template<class T>
inline cv::Point_<T>
LineIntersection(const cv::Point_<T>& A, const cv::Point_<T>& B, const cv::Point_<T>& C, const cv::Point_<T>& D)
{
    // Line AB represented as a1x + b1y = c1
    auto a1 = B.y - A.y;
    auto b1 = A.x - B.x;
    auto c1 = a1 * (A.x) + b1 * (A.y);

    // Line CD represented as a2x + b2y = c2
    auto a2 = D.y - C.y;
    auto b2 = C.x - D.x;
    auto c2 = a2 * (C.x) + b2 * (C.y);

    auto determinant = a1 * b2 - a2 * b1;

    if (determinant == 0) {
        // The lines are parallel. This is simplified
        // by returning a pair of FLT_MAX
        return cv::Point_<T>(FLT_MAX, FLT_MAX);
    }
    else {
        auto x = (b2 * c1 - b1 * c2) / determinant;
        auto y = (a1 * c2 - a2 * c1) / determinant;
        return cv::Point_<T>(x, y);
    }
}

// Checks if a matrix is a valid rotation matrix.
inline bool IsRotationMatrix(const cv::Mat& R)
{
    cv::Mat Rt;
    cv::transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3, 3, shouldBeIdentity.type());

    return cv::norm(I, shouldBeIdentity) < 1e-6;
}

inline cv::Vec3f EulerAnglesFromRotation(const cv::Mat& R)
{
    assert(IsRotationMatrix(R));

    float sy = sqrtf(R.at<float>(0, 0) * R.at<float>(0, 0) + R.at<float>(1, 0) * R.at<float>(1, 0));

    bool singular = sy < 1e-6; // If

    float roll, pitch, yaw;
    if (!singular) {
        roll = atan2(R.at<float>(2, 1), R.at<float>(2, 2));
        pitch = atan2(-R.at<float>(2, 0), sy);
        yaw = atan2(R.at<float>(1, 0), R.at<float>(0, 0));
    }
    else {
        roll = atan2(-R.at<float>(1, 2), R.at<float>(1, 1));
        pitch = atan2(-R.at<float>(2, 0), sy);
        yaw = 0;
    }
    return cv::Vec3f(roll, pitch, yaw);
}

inline cv::Mat EulerAnglesToRotationMatrix(const cv::Vec3f& theta)
{
    // Calculate rotation about x axis
    cv::Mat R_x = (cv::Mat_<float>(3, 3)
            << 1, 0, 0,
            0, cos(theta[0]), -sin(theta[0]),
            0, sin(theta[0]), cos(theta[0])
    );

    // Calculate rotation about y axis
    cv::Mat R_y = (cv::Mat_<float>(3, 3)
            << cos(theta[1]), 0, sin(theta[1]),
            0, 1, 0,
            -sin(theta[1]), 0, cos(theta[1])
    );

    // Calculate rotation about z axis
    cv::Mat R_z = (cv::Mat_<float>(3, 3)
            << cos(theta[2]), -sin(theta[2]), 0,
            sin(theta[2]), cos(theta[2]), 0,
            0, 0, 1);

    // Combined rotation matrix
    cv::Mat R = R_z * R_y * R_x;

    return R;

}

inline float AlignmentError(const cv::Point2f& pt, const LineSegment& edge)
{
    cv::Vec2f v1(pt.x - edge.first.x, pt.y - edge.first.y);
    cv::Vec2f v2(edge.second.x - edge.first.x, edge.second.y - edge.first.y);
    float cosine = v1.dot(v2) / (norm(v1) * norm(v2));
    float angle = acos(cosine);
    if (angle > M_PI_2)
        angle = M_PI - angle;
    return angle;
}

inline void DrawProposal(cv::Mat& canvas, const CuboidProposal& proposal)
{
    cv::line(canvas, proposal.corners[0], proposal.corners[1], cv::Scalar(0, 255, 0),
             1 + (proposal.isCornerVisible[0] && proposal.isCornerVisible[1]), CV_AA);
    cv::line(canvas, proposal.corners[1], proposal.corners[3], cv::Scalar(0, 255, 0),
             1 + (proposal.isCornerVisible[1] && proposal.isCornerVisible[3]), CV_AA);
    cv::line(canvas, proposal.corners[3], proposal.corners[2], cv::Scalar(0, 255, 0),
             1 + (proposal.isCornerVisible[3] && proposal.isCornerVisible[2]), CV_AA);
    cv::line(canvas, proposal.corners[2], proposal.corners[0], cv::Scalar(0, 255, 0),
             1 + (proposal.isCornerVisible[2] && proposal.isCornerVisible[0]), CV_AA);
    cv::line(canvas, proposal.corners[0], proposal.corners[7], cv::Scalar(0, 255, 0),
             1 + (proposal.isCornerVisible[0] && proposal.isCornerVisible[7]), CV_AA);
    cv::line(canvas, proposal.corners[1], proposal.corners[6], cv::Scalar(0, 255, 0),
             1 + (proposal.isCornerVisible[1] && proposal.isCornerVisible[6]), CV_AA);
    cv::line(canvas, proposal.corners[2], proposal.corners[5], cv::Scalar(0, 255, 0),
             1 + (proposal.isCornerVisible[2] && proposal.isCornerVisible[5]), CV_AA);
    cv::line(canvas, proposal.corners[3], proposal.corners[4], cv::Scalar(0, 255, 0),
             1 + (proposal.isCornerVisible[3] && proposal.isCornerVisible[4]), CV_AA);
    cv::line(canvas, proposal.corners[7], proposal.corners[6], cv::Scalar(0, 255, 0),
             1 + (proposal.isCornerVisible[7] && proposal.isCornerVisible[6]), CV_AA);
    cv::line(canvas, proposal.corners[6], proposal.corners[4], cv::Scalar(0, 255, 0),
             1 + (proposal.isCornerVisible[6] && proposal.isCornerVisible[4]), CV_AA);
    cv::line(canvas, proposal.corners[4], proposal.corners[5], cv::Scalar(0, 255, 0),
             1 + (proposal.isCornerVisible[4] && proposal.isCornerVisible[5]), CV_AA);
    cv::line(canvas, proposal.corners[5], proposal.corners[7], cv::Scalar(0, 255, 0),
             1 + (proposal.isCornerVisible[5] && proposal.isCornerVisible[7]), CV_AA);

    for (int i = 0; i < 8; ++i) {
        cv::putText(canvas, std::to_string(i), proposal.corners[i],
                    cv::FONT_HERSHEY_SIMPLEX, 0.5f, cv::Scalar(0, 0, 0), 4);
        cv::putText(canvas, std::to_string(i), proposal.corners[i],
                    cv::FONT_HERSHEY_SIMPLEX, 0.5f, cv::Scalar(255, 255, 255), 2);
    }
}

CuboidProposal FindBestProposal(cv::Rect bbox, std::vector<LineSegment*> lineSegs, cv::Mat K,
                                float shapeErrThresh, float shapeErrWeight, float alignErrWeight,
                                float initRoll, float initPitch, float initYaw,
                                float frameId = 0, cv::Mat image = cv::Mat(), bool display = false, bool save = false);

}

#endif //CUBESLAM_H
