/**
* This file is part of CNN-SLAM.
*
* [Copyright of CNN-SLAM]
* Copyright (C) 2018
* Kai Yu <kaiy1 at andrew dot cmu dot edu> (Carnegie Mellon University)
* Zhongxu Wang <zhongxuw at andrew dot cmu dot edu> (Carnegie Mellon University)
* Manchen Wang <manchen2 at andrew dot cmu dot edu> (Carnegie Mellon University)
* For more information see <https://github.com/raulmur/CNN_SLAM>
*
* CNN-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* CNN-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with CNN-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef CNN_SLAM_SETTINGS_H
#define CNN_SLAM_SETTINGS_H
#pragma once

#include <cstring>
#include <string>
#include <cmath>

namespace cnn_slam {
    const unsigned int TRACKING_NUM_PT = static_cast<const unsigned int>(sqrt(640 * 480));
    const float TRACKING_HUBER_DELTA =1.0;
    const float TRACKING_SOLVER_TIMECOST_RATIO = 0.5;
    const float INIT_TOLERANCE = 25;
}

#endif //CNN_SLAM_SETTINGS_H
