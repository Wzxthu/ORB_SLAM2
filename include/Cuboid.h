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

#ifndef CUBOID_H
#define CUBOID_H

#include <opencv2/opencv.hpp>

#define M_PI_F        3.14159265358979323846264338327950288f   /* pi             */
#define M_PI_2_F      1.57079632679489661923132169163975144f   /* pi/2           */
#define M_PI_4_F      0.785398163397448309615660845819875721f  /* pi/4           */

namespace ORB_SLAM2 {

struct Dimension3D {
    float edge13{};   // Corresponding to vanishing point 1.
    float edge12{};   // Corresponding to vanishing point 2.
    float edge18{};   // Corresponding to vanishing point 3.

    inline Dimension3D() = default;
    Dimension3D(float edge13_, float edge12_, float edge18_)
            :edge13(edge13_), edge12(edge12_), edge18(edge18_) { }

    friend std::ostream& operator<<(std::ostream& out, const Dimension3D& dim);
};

inline std::ostream& operator<<(std::ostream& out, const Dimension3D& dim)
{
    out << '[' << dim.edge18 << 'x' << dim.edge12 << 'x' << dim.edge13 << ']';
    return out;
}

// Represent the cuboid proposal with the coordinates in frame of the 8 corners.
struct Cuboid2D {
    cv::Mat Rlc;
    cv::Point2f corners[8];
    bool isCornerVisible[8]{};
    bool valid = false;

    friend std::ostream& operator<<(std::ostream& out, const Cuboid2D& cuboid);

    inline Cuboid2D() = default;

    inline Cuboid2D(const Cuboid2D& other)
    {
        valid = other.valid;
        Rlc = other.Rlc.clone();
        memcpy(corners, other.corners, sizeof(corners));
        memcpy(isCornerVisible, other.isCornerVisible, sizeof(isCornerVisible));
    }

    cv::Point2f GetCentroid() const;

    void Draw(cv::Mat& canvas, const cv::Mat& K, const cv::Scalar& edgeColor = cv::Scalar(255, 255, 255)) const;

    cv::Mat GetCentroid3D(float depth, const cv::Mat& invK) const;

    Dimension3D GetDimension3D(const cv::Mat& centroid3D, const cv::Mat& invK) const;
    inline Dimension3D GetDimension3D(float centroidDepth, const cv::Mat& invK) const {
        return GetDimension3D(GetCentroid3D(centroidDepth, invK), invK);
    }
};

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
    return ((rz * Px - rx * Pz) / (rx * dz - rz * dx) + (rz * Py - ry * Pz) / (ry * dz - rz * dy)) * .5f /
           sqrtf(dx * dx + dy * dy + dz * dz);
}

/**
 * Compute the intersection point of line AB and line CD (not segment!).
 * @tparam T type of point.
 * @return the intersection point.
 */
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

inline cv::Point2f PointFrom2DHomo(const cv::Mat& homo)
{
    const float RANGE = 1e6;
    const float X = homo.at<float>(0, 0);
    const float Y = homo.at<float>(1, 0);
    const float Z = homo.at<float>(2, 0);
    const float absZ = fabs(Z);
    if (absZ >= 1)
        return cv::Point2f(X / Z, Y / Z);
    const float maxAbsXY = std::max(fabs(homo.at<float>(0)), fabs(homo.at<float>(1)));
    if (maxAbsXY < RANGE * absZ)
        return cv::Point2f(X / Z, Y / Z);
    if (fabs(X) > fabs(Y)) {
        const float x = X > 0 ? RANGE : -RANGE;
        const float y = x * (Y / X);
        return cv::Point2f(x, y);
    }
    else {
        const float y = Y > 0 ? RANGE : -RANGE;
        const float x = y * (X / Y);
        return cv::Point2f(x, y);
    }
}

inline std::ostream& operator<<(std::ostream& out, const Cuboid2D& cuboid)
{
    out << '[';
    for (int i = 0; i < 7; ++i)
        out << cuboid.corners[i] << ',';
    out << cuboid.corners[7] << ']';
    return out;
}

template<class T>
inline bool IsParallel(const cv::Point_<T>& A, const cv::Point_<T>& B, const cv::Point_<T>& C, const cv::Point_<T>& D)
{
    return (A.x - B.x) * (C.y - D.y) - (A.y - B.y) * (C.x - D.x) < 1e-6;
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

inline cv::Mat PointToHomo(const cv::Point2f& pt)
{
    return (cv::Mat_<float>(3, 1, CV_32F) << pt.x, pt.y, 1);
}

inline cv::Mat PointToHomo(const cv::Point3f& pt)
{
    return (cv::Mat_<float>(4, 1, CV_32F) << pt.x, pt.y, pt.z, 1);
}

}

#endif //CUBOID_H
