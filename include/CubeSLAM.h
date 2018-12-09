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

#pragma once

#ifndef CUBESLAM_H
#define CUBESLAM_H

#include "LineSegmentDetector.h"
#include "Landmark.h"

#include <array>
#include <opencv2/opencv.hpp>

namespace ORB_SLAM2 {

inline cv::Mat PointToHomo(const cv::Point2f& pt)
{
    return (cv::Mat_<float>(3, 1, CV_32F) << pt.x, pt.y, 1);
}

inline cv::Mat PointToHomo(const cv::Point3f& pt)
{
    return (cv::Mat_<float>(4, 1, CV_32F) << pt.x, pt.y, pt.z, 1);
}

template<class T>
inline float DistanceSquare(const cv::Point_<T>& pt1, const cv::Point_<T>& pt2)
{
    return powf(pt1.x - pt2.x, 2) + powf(pt1.y - pt2.y, 2);
}

template<class T1, class T2>
inline bool Inside(const cv::Point_<T1>& pt, const cv::Rect_<T2>& bbox)
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

/**
 * Compute distance from a 3D point to a ray along the given direction.
 * @param pt3D 3x1 vector
 * @param direction 3x1 vector
 * @param ray 3x1 vector
 * @return distance
 */
inline float DistanceToRay(const cv::Mat& pt3D, const cv::Mat& direction, const cv::Mat& ray)
{
    const float Px = pt3D.at<float>(0, 0);
    const float Py = pt3D.at<float>(1, 0);
    const float Pz = pt3D.at<float>(2, 0);
    const float dx = direction.at<float>(0, 0);
    const float dy = direction.at<float>(1, 0);
    const float dz = direction.at<float>(2, 0);
    const float rx = ray.at<float>(0, 0);
    const float ry = ray.at<float>(1, 0);
    const float rz = ray.at<float>(2, 0);
    std::cout << "Dimension: " << (rz * Px - rx * Pz) / (rx * dz - rz * dx) << ' '
              << (rz * Py - ry * Pz) / (ry * dz - rz * dy) << std::endl;
    return ((rz * Px - rx * Pz) / (rx * dz - rz * dx) + (rz * Py - ry * Pz) / (ry * dz - rz * dy)) * .5f /
           sqrtf(dx * dx + dy * dy + dz * dz);
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
    float cosine = v1.dot(v2) / (normf(v1) * normf(v2));
    float angle = acos(cosine);
    if (angle > M_PI_2_F)
        angle = M_PI_F - angle;
    return angle;
}

Cuboid2D GenerateCuboidProposal(const cv::Rect& bbox, int topX,
                                const cv::Point2f& vp1, const cv::Point2f& vp2, const cv::Point2f& vp3);

Cuboid2D FindBestProposal(const cv::Rect& bbox, const std::vector<LineSegment*>& lineSegs, const cv::Mat& K,
                          float shapeErrThresh, float shapeErrWeight, float alignErrWeight,
                          float refRoll, float refPitch,
                          float rollRange = 45 * M_PI_F / 180,
                          float pitchRange = 45 * M_PI_F / 180,
                          unsigned long frameId = 0, int objId = 0, const cv::Mat& image = cv::Mat(),
                          bool display = false, bool save = false);

LandmarkDimension DimensionFromProposal(const Cuboid2D& proposal, const cv::Mat& camCoordCentroid);

}

#endif //CUBESLAM_H
