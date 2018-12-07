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

#ifndef LINESEGMENTDETECTOR_H
#define LINESEGMENTDETECTOR_H

#include <vector>

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2 {

typedef std::pair<cv::Point2f, cv::Point2f> LineSegment;

class LineSegmentDetector {
public:
    LineSegmentDetector();

    std::vector<LineSegment> Detect(const cv::Mat& imGray);
private:
    cv::Ptr<cv::LineSegmentDetector> mpDetector;
};

}

#endif //LINESEGMENTDETECTOR_H
