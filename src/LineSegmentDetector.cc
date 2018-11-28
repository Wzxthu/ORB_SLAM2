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

#include <include/LineSegmentDetector.h>

#include "LineSegmentDetector.h"

using namespace std;
using namespace cv;

namespace ORB_SLAM2 {

LineSegmentDetector::LineSegmentDetector()
        :mpDetector(cv::createLineSegmentDetector()) { }

std::vector<LineSegment> LineSegmentDetector::Detect(const cv::Mat& imGray)
{
    Mat matSegs;
    mpDetector->detect(imGray, matSegs);
    std::vector<LineSegment> vecSegs;
    vecSegs.reserve(static_cast<unsigned long>(matSegs.rows));
    for (int i = 0; i<matSegs.rows; ++i)
        vecSegs.emplace_back(
                Point(static_cast<int>(matSegs.at<float>(i, 0)), static_cast<int>(matSegs.at<float>(i, 1))),
                Point(static_cast<int>(matSegs.at<float>(i, 2)), static_cast<int>(matSegs.at<float>(i, 3))));
    return vecSegs;
}

}