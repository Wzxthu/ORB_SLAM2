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
    const float MIN_USE_GRAD = 5.f;
    const float TRACKING_HUBER_DELTA =1.0;
    const float TRACKING_SOLVER_TIMECOST_RATIO = 0.5;

    #define DIVISION_EPS 1e-10f
    #define UNZERO(val) (val < 0 ? (val > -1e-10 ? -1e-10 : val) : (val < 1e-10 ? 1e-10 : val))

    #if defined(ENABLE_SSE)
    #define USESSE true
    #else
    #define USESSE false
    #endif


    #if defined(NDEBUG)
    #define enablePrintDebugInfo false
    #else
    #define enablePrintDebugInfo true
    #endif


    /** ============== Depth Variance Handeling ======================= */
    #define SUCC_VAR_INC_FAC (1.01f) // before an ekf-update, the variance is increased by this factor.


    #ifdef ANDROID
        // tracking pyramid levels.
        #define MAPPING_THREADS 2
        #define RELOCALIZE_THREADS 4
    #else
        // tracking pyramid levels.
    #define MAPPING_THREADS 4
    #define RELOCALIZE_THREADS 6
    #endif

    // ============== stereo & gradient calculation ======================
    #define MIN_DEPTH 0.05f // this is the minimal depth tested for stereo.

    // particularely important for initial pixel.
    #define MAX_EPL_LENGTH_CROP 30.0f // maximum length of epl to search.
    #define MIN_EPL_LENGTH_CROP (3.0f) // minimum length of epl to search.

    // this is the distance of the sample points used for the stereo descriptor.
    #define GRADIENT_SAMPLE_DIST 1.0f

    // pixel a point needs to be away from border... if too small: segfaults!
    #define SAMPLE_POINT_TO_BORDER 7

    // pixels with too big an error are definitely thrown out.
    #define MAX_ERROR_STEREO (1300.0f) // maximal photometric error for stereo to be successful (sum over 5 squared intensity differences)
    #define MIN_DISTANCE_ERROR_STEREO (1.5f) // minimal multiplicative difference to second-best match to not be considered ambiguous.

    // defines how large the stereo-search region is. it is [mean] +/- [std.dev]*STEREO_EPL_VAR_FAC
    #define STEREO_EPL_VAR_FAC 2.0f


    // ============== initial stereo pixel selection ======================
    #define MIN_EPL_GRAD_SQUARED (2.0f*2.0f)
    #define MIN_EPL_LENGTH_SQUARED (1.0f*1.0f)
    #define MIN_EPL_ANGLE_SQUARED (0.3f*0.3f)

    // abs. grad at that location needs to be larger than this.
    #define MIN_ABS_GRAD_CREATE (minUseGrad)
    #define MIN_ABS_GRAD_DECREASE (minUseGrad)

    // ============== RE-LOCALIZATION, KF-REACTIVATION etc. ======================
    // defines the level on which we do the quick tracking-check for relocalization.



    #define MAX_DIFF_CONSTANT (40.0f*40.0f)
    #define MAX_DIFF_GRAD_MULT (0.5f*0.5f)

    #define MIN_GOODPERGOODBAD_PIXEL (0.5f)
    #define MIN_GOODPERALL_PIXEL (0.04f)
    #define MIN_GOODPERALL_PIXEL_ABSMIN (0.01f)

    #define INITIALIZATION_PHASE_COUNT 5

    #define MIN_NUM_MAPPED 5

    // settings variables
    // controlled via keystrokes
        extern bool autoRun;
        extern bool autoRunWithinFrame;
        extern int debugDisplay;
        extern bool displayDepthMap;
        extern bool onSceenInfoDisplay;
        extern bool dumpMap;
        extern bool doFullReConstraintTrack;


    // dyn config
        extern bool printPropagationStatistics;
        extern bool printFillHolesStatistics;
        extern bool printObserveStatistics;
        extern bool printObservePurgeStatistics;
        extern bool printRegularizeStatistics;
        extern bool printLineStereoStatistics;
        extern bool printLineStereoFails;

        extern bool printTrackingIterationInfo;
        extern bool printThreadingInfo;

        extern bool printKeyframeSelectionInfo;
        extern bool printConstraintSearchInfo;
        extern bool printOptimizationInfo;
        extern bool printRelocalizationInfo;

        extern bool printFrameBuildDebugInfo;
        extern bool printMemoryDebugInfo;

        extern bool printMappingTiming;
        extern bool printOverallTiming;
        extern bool plotTrackingIterationInfo;
        extern bool plotSim3TrackingIterationInfo;
        extern bool plotStereoImages;
        extern bool plotTracking;


        extern bool allowNegativeIdepths;
        extern bool useMotionModel;
        extern bool useSubpixelStereo;
        extern bool multiThreading;
        extern bool useAffineLightningEstimation;

        extern float freeDebugParam1;
        extern float freeDebugParam2;
        extern float freeDebugParam3;
        extern float freeDebugParam4;
        extern float freeDebugParam5;


        extern float KFDistWeight;
        extern float KFUsageWeight;
        extern int maxLoopClosureCandidates;
        extern int propagateKeyFrameDepthCount;
        extern float loopclosureStrictness;
        extern float relocalizationTH;


        extern float minUseGrad;
        extern float cameraPixelNoise2;
        extern float depthSmoothingFactor;

        extern bool useFabMap;
        extern bool doSlam;
        extern bool doKFReActivation;
        extern bool doMapping;

        extern bool saveKeyframes;
        extern bool saveAllTracked;
        extern bool saveLoopClosureImages;
        extern bool saveAllTrackingStages;
        extern bool saveAllTrackingStagesInternal;

        extern bool continuousPCOutput;


    /// Relative path of calibration file, map saving directory etc. for live_odometry
        extern std::string packagePath;

        extern bool fullResetRequested;
        extern bool manualTrackingLossIndicated;
}

#endif //CNN_SLAM_SETTINGS_H
