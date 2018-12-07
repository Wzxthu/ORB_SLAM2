#ifndef CUBESLAM_H
#define CUBESLAM_H

#include "LineSegmentDetector.h"

#include <array>
#include <opencv2/opencv.hpp>

namespace ORB_SLAM2 {

// Represent the cuboid proposal with the coordinates in frame of the 8 corners.
typedef std::array<cv::Point2f, 8> CuboidProposal;

template<class T>
inline float DistanceSquare(const cv::Point_<T>& pt1, const cv::Point_<T>& pt2)
{
    return powf(pt1.x - pt2.x, 2) + powf(pt1.y - pt2.y, 2);
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

inline void RollPitchYawFromRotation(const cv::Mat& rot, float& roll, float& pitch, float& yaw)
{
    roll = atan2(rot.at<float>(2, 1), rot.at<float>(2, 2));
    pitch = atan2(-rot.at<float>(2, 0), sqrt(powf(rot.at<float>(2, 1), 2) + powf(rot.at<float>(2, 2), 2)));
    yaw = atan2(rot.at<float>(1, 0), rot.at<float>(0, 0));
}

inline cv::Mat RotationFromRollPitchYaw(float roll, float pitch, float yaw)
{
    cv::Mat rot(3, 3, CV_32F);
    rot.at<float>(0, 0) = cos(yaw) * cos(pitch);
    rot.at<float>(0, 1) = cos(yaw) * sin(pitch) * sin(roll) - sin(yaw) * cos(roll);
    rot.at<float>(0, 2) = cos(yaw) * sin(pitch) * cos(roll) + sin(yaw) * sin(roll);
    rot.at<float>(1, 0) = sin(yaw) * cos(pitch);
    rot.at<float>(1, 1) = sin(yaw) * sin(pitch) * sin(roll) + cos(yaw) * cos(roll);
    rot.at<float>(1, 2) = sin(yaw) * sin(pitch) * cos(roll) - cos(yaw) * sin(roll);
    rot.at<float>(2, 0) = -sin(pitch);
    rot.at<float>(2, 1) = cos(pitch) * sin(roll);
    rot.at<float>(2, 2) = cos(pitch) * cos(roll);
    return rot;
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

CuboidProposal FindBestProposal(cv::Rect bbox, float c_yaw, float c_roll, float c_pitch,
                                float shapeErrThresh, float shapeErrWeight, float alignErrWeight,
                                std::vector<LineSegment*> lineSegs,
                                cv::Mat K,
                                float& bestErr,
                                float frameId = 0,
                                cv::Mat image = cv::Mat());

}

#endif //CUBESLAM_H
