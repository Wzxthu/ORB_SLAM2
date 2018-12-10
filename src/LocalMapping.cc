/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include "CubeSLAM.h"
#include <sys/stat.h>

#include <mutex>
#include <chrono>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <utility>
#include <vector>
#include <opencv2/core/ocl.hpp>

using namespace std;
using namespace cv;

namespace ORB_SLAM2 {

LocalMapping::LocalMapping(Map* pMap, FrameDrawer* pFrameDrawer, bool bMonocular,
                           float alignErrWeight, float shapeErrWeight, float shapeErrThresh)
        :
        mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
        mpFrameDrawer(pFrameDrawer), mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false),
        mbAcceptKeyFrames(true), mAlignErrWeight(alignErrWeight), mShapeErrWeight(shapeErrWeight),
        mShapeErrThresh(shapeErrThresh)
{
    if (mkdir("Outputs", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != -1) {
        std::cout << "Success!" << endl;
    }
}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking* pTracker)
{
    mpTracker = pTracker;
}

void LocalMapping::Run()
{
    ocl::setUseOpenCL(true);

    mpObjectDetector = new ObjectDetector("Thirdparty/darknet/cfg/yolov3.cfg", "model/yolov3.weights",
                                          .45, .6, 224 * 224);
    mpLineSegDetector = new LineSegmentDetector();

    mbFinished = false;

    while (true) {
        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue
        if (CheckNewKeyFrames()) {
            // BoW conversion and insertion in Map
            ProcessNewKeyFrame();

            // Check recent MapPoints
            MapPointCulling();

            // Triangulate new MapPoints
            CreateNewMapPoints();

            if (!CheckNewKeyFrames()) {
                // Find more matches in neighbor keyframes and fuse point duplications
                SearchInNeighbors();

                // From CubeSLAM, detect landmarks and put them into bundle adjustment.
                FindLandmarks();
            }

            // We won't need its images anymore.
            mpCurrentKeyFrame->mImColor.release();
            mpCurrentKeyFrame->mImGray.release();

            mbAbortBA = false;

            if (!CheckNewKeyFrames() && !stopRequested()) {
                // Local BA
                // TODO: Add landmarks into BA.
                if (mpMap->KeyFramesInMap() > 2)
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &mbAbortBA, mpMap);

                // Check redundant local Keyframes
                KeyFrameCulling();
            }

            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        }
        else if (Stop()) {
            // Safe area to stop
            while (isStopped() && !CheckFinish()) {
                this_thread::sleep_for(chrono::milliseconds(3));
            }
            if (CheckFinish())
                break;
        }

        ResetIfRequested();

        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(true);

        if (CheckFinish())
            break;

        this_thread::sleep_for(chrono::milliseconds(3));
    }

    SetFinish();

    delete mpObjectDetector;
    delete mpLineSegDetector;
}

void LocalMapping::InsertKeyFrame(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA = true;
}

bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return (!mlNewKeyFrames.empty());
}

void LocalMapping::ProcessNewKeyFrame()
{
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for (size_t i = 0; i < vpMapPointMatches.size(); i++) {
        MapPoint* pMP = vpMapPointMatches[i];
        if (pMP) {
            if (!pMP->isBad()) {
                if (!pMP->IsInKeyFrame(mpCurrentKeyFrame)) {
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    pMP->UpdateNormalAndDepth();
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }

    // Update links in the Covisibility Graph
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    auto lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;
    if (mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    while (lit != mlpRecentAddedMapPoints.end()) {
        MapPoint* pMP = *lit;
        if (pMP->isBad()) {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if (pMP->GetFoundRatio() < 0.25f) {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if (((int) nCurrentKFid - (int) pMP->mnFirstKFid) >= 2 && pMP->Observations() <= cnThObs) {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if (((int) nCurrentKFid - (int) pMP->mnFirstKFid) >= 3)
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++;
    }
}

void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;
    if (mbMonocular)
        nn = 20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    ORBmatcher matcher(0.6, false);

    Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    Mat Rwc1 = Rcw1.t();
    Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    Mat Tcw1(3, 4, CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0, 3));
    tcw1.copyTo(Tcw1.col(3));
    Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const float& fx1 = mpCurrentKeyFrame->fx;
    const float& fy1 = mpCurrentKeyFrame->fy;
    const float& cx1 = mpCurrentKeyFrame->cx;
    const float& cy1 = mpCurrentKeyFrame->cy;
    const float& invfx1 = mpCurrentKeyFrame->invfx;
    const float& invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f * mpCurrentKeyFrame->mfScaleFactor;

    int nnew = 0;

    // Search matches with epipolar restriction and triangulate
    for (size_t i = 0; i < vpNeighKFs.size(); i++) {
        if (i > 0 && CheckNewKeyFrames())
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        Mat Ow2 = pKF2->GetCameraCenter();
        Mat vBaseline = Ow2 - Ow1;
        const float baseline = norm(vBaseline);

        if (!mbMonocular) {
            if (baseline < pKF2->mb)
                continue;
        }
        else {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline / medianDepthKF2;

            if (ratioBaselineDepth < 0.01)
                continue;
        }

        // Compute Fundamental Matrix
        Mat F12 = ComputeF12(mpCurrentKeyFrame, pKF2);

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t, size_t> > vMatchedIndices;
        matcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, F12, vMatchedIndices, false);

        Mat Rcw2 = pKF2->GetRotation();
        Mat Rwc2 = Rcw2.t();
        Mat tcw2 = pKF2->GetTranslation();
        Mat Tcw2(3, 4, CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0, 3));
        tcw2.copyTo(Tcw2.col(3));

        const float& fx2 = pKF2->fx;
        const float& fy2 = pKF2->fy;
        const float& cx2 = pKF2->cx;
        const float& cy2 = pKF2->cy;
        const float& invfx2 = pKF2->invfx;
        const float& invfy2 = pKF2->invfy;

        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        for (int ikp = 0; ikp < nmatches; ikp++) {
            const auto& idx1 = vMatchedIndices[ikp].first;
            const auto& idx2 = vMatchedIndices[ikp].second;

            const KeyPoint& kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            const float kp1_ur = mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = kp1_ur >= 0;

            const KeyPoint& kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur >= 0;

            // Check parallax between rays
            Mat xn1 = (Mat_<float>(3, 1) << (kp1.pt.x - cx1) * invfx1, (kp1.pt.y - cy1) * invfy1, 1.0);
            Mat xn2 = (Mat_<float>(3, 1) << (kp2.pt.x - cx2) * invfx2, (kp2.pt.y - cy2) * invfy2, 1.0);

            Mat ray1 = Rwc1 * xn1;
            Mat ray2 = Rwc2 * xn2;
            const float cosParallaxRays = ray1.dot(ray2) / (norm(ray1) * norm(ray2));

            float cosParallaxStereo = cosParallaxRays + 1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if (bStereo1)
                cosParallaxStereo1 = cos(2 * atan2(mpCurrentKeyFrame->mb / 2, mpCurrentKeyFrame->mvDepth[idx1]));
            else if (bStereo2)
                cosParallaxStereo2 = cos(2 * atan2(pKF2->mb / 2, pKF2->mvDepth[idx2]));

            cosParallaxStereo = min(cosParallaxStereo1, cosParallaxStereo2);

            Mat x3D;
            if (cosParallaxRays < cosParallaxStereo && cosParallaxRays > 0
                && (bStereo1 || bStereo2 || cosParallaxRays < 0.9998)) {
                // Linear Triangulation Method
                Mat A(4, 4, CV_32F);
                A.row(0) = xn1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
                A.row(1) = xn1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
                A.row(2) = xn2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
                A.row(3) = xn2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

                Mat w, u, vt;
                SVD::compute(A, w, u, vt, SVD::MODIFY_A | SVD::FULL_UV);

                x3D = vt.row(3).t();

                if (x3D.at<float>(3) == 0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);

            }
            else if (bStereo1 && cosParallaxStereo1 < cosParallaxStereo2) {
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);
            }
            else if (bStereo2 && cosParallaxStereo2 < cosParallaxStereo1) {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else
                continue; //No stereo and very low parallax

            Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);
            if (z1 <= 0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
            if (z2 <= 0)
                continue;

            //Check reprojection error in first keyframe
            const float& sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
            const float invz1 = 1.0 / z1;

            if (!bStereo1) {
                float u1 = fx1 * x1 * invz1 + cx1;
                float v1 = fy1 * y1 * invz1 + cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                if ((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1)
                    continue;
            }
            else {
                float u1 = fx1 * x1 * invz1 + cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf * invz1;
                float v1 = fy1 * y1 * invz1 + cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if ((errX1 * errX1 + errY1 * errY1 + errX1_r * errX1_r) > 7.8 * sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1);
            const float invz2 = 1.0 / z2;
            if (!bStereo2) {
                float u2 = fx2 * x2 * invz2 + cx2;
                float v2 = fy2 * y2 * invz2 + cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if ((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2)
                    continue;
            }
            else {
                float u2 = fx2 * x2 * invz2 + cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf * invz2;
                float v2 = fy2 * y2 * invz2 + cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if ((errX2 * errX2 + errY2 * errY2 + errX2_r * errX2_r) > 7.8 * sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            Mat normal1 = x3D - Ow1;
            float dist1 = norm(normal1);

            Mat normal2 = x3D - Ow2;
            float dist2 = norm(normal2);

            if (dist1 == 0 || dist2 == 0)
                continue;

            const float ratioDist = dist2 / dist1;
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave] / pKF2->mvScaleFactors[kp2.octave];

            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
            if (ratioDist * ratioFactor < ratioOctave || ratioDist > ratioOctave * ratioFactor)
                continue;

            // Triangulation is succesfull
            auto* pMP = new MapPoint(x3D, mpCurrentKeyFrame, mpMap);

            pMP->AddObservation(mpCurrentKeyFrame, idx1);
            pMP->AddObservation(pKF2, idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP, idx1);
            pKF2->AddMapPoint(pMP, idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);

            nnew++;
        }
    }
}

void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    int nn = 10;
    if (mbMonocular)
        nn = 20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFrame*> vpTargetKFs;
    for (auto pKFi : vpNeighKFs) {
        if (pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        // Extend to some second neighbors
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for (auto pKFi2 : vpSecondNeighKFs) {
            if (pKFi2->isBad() || pKFi2->mnFuseTargetForKF == mpCurrentKeyFrame->mnId
                || pKFi2->mnId == mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }


    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for (auto pKFi : vpTargetKFs) {
        matcher.Fuse(pKFi, vpMapPointMatches);
    }

    // Search matches by projection from target KFs in current KF
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size() * vpMapPointMatches.size());

    for (auto pKFi : vpTargetKFs) {
        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for (auto pMP : vpMapPointsKFi) {
            if (!pMP)
                continue;
            if (pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame, vpFuseCandidates);


    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for (auto pMP : vpMapPointMatches) {
        if (pMP) {
            if (!pMP->isBad()) {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();

    // Project landmarks in previous keyframes to the current keyframe.
    auto kfPose = mpCurrentKeyFrame->GetPose();
    auto Rcw_z = kfPose.row(2).colRange(0, 3).t();
    auto tcw_z = kfPose.at<float>(3, 2);
    for (auto pKFi : vpTargetKFs) {
        for (const auto& pLandmark : pKFi->GetLandmarks()) {
            // See if this landmark is visible in the current keyframe.
            auto Lc_z = Rcw_z.dot(pLandmark->GetCentroid()) + tcw_z;
            if (Lc_z > 0) {
                mpCurrentKeyFrame->AddLandmark(pLandmark);
            }
        }
    }
}

Mat LocalMapping::ComputeF12(KeyFrame*& pKF1, KeyFrame*& pKF2)
{
    Mat R1w = pKF1->GetRotation();
    Mat t1w = pKF1->GetTranslation();
    Mat R2w = pKF2->GetRotation();
    Mat t2w = pKF2->GetTranslation();

    Mat R12 = R1w * R2w.t();
    Mat t12 = -R1w * R2w.t() * t2w + t1w;

    Mat t12x = SkewSymmetricMatrix(t12);

    const Mat& K1 = pKF1->mK;
    const Mat& K2 = pKF2->mK;

    return K1.t().inv() * t12x * R12 * K2.inv();
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if (mbStopRequested && !mbNotStop) {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if (mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for (auto& mlNewKeyFrame : mlNewKeyFrames)
        delete mlNewKeyFrame;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames = flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if (flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    for (auto pKF : vpLocalKeyFrames) {
        if (pKF->mnId == 0)
            continue;
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs = nObs;
        int nRedundantObservations = 0;
        int nMPs = 0;
        for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++) {
            MapPoint* pMP = vpMapPoints[i];
            if (pMP) {
                if (!pMP->isBad()) {
                    if (!mbMonocular) {
                        if (pKF->mvDepth[i] > pKF->mThDepth || pKF->mvDepth[i] < 0)
                            continue;
                    }

                    nMPs++;
                    if (pMP->Observations() > thObs) {
                        const int& scaleLevel = pKF->mvKeysUn[i].octave;
                        const map<KeyFrame*, size_t> observations = pMP->GetObservations();
                        int nObs = 0;
                        for (auto observation : observations) {
                            KeyFrame* pKFi = observation.first;
                            if (pKFi == pKF)
                                continue;
                            const int& scaleLeveli = pKFi->mvKeysUn[observation.second].octave;

                            if (scaleLeveli <= scaleLevel + 1) {
                                nObs++;
                                if (nObs >= thObs)
                                    break;
                            }
                        }
                        if (nObs >= thObs) {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }

        if (nRedundantObservations > 0.9 * nMPs)
            pKF->SetBadFlag();
    }
}

Mat LocalMapping::SkewSymmetricMatrix(const Mat& v)
{
    return (Mat_<float>(3, 3) << 0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2), 0, -v.at<float>(0),
            -v.at<float>(1), v.at<float>(0), 0);
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while (true) {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if (!mbResetRequested)
                break;
        }
        this_thread::sleep_for(chrono::milliseconds(3));
    }
}

void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if (mbResetRequested) {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested = false;
    }
}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void LocalMapping::FindLandmarks()
{
    if (mpCurrentKeyFrame->mImColor.empty())
        return;

    Mat canvas = mpCurrentKeyFrame->mImColor.clone();

    using namespace chrono;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    vector<Object> objects2D;
    mpObjectDetector->Detect(mpCurrentKeyFrame->mImColor, objects2D);

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> timeSpan = duration_cast<duration<double>>(t2 - t1);
    cout << "YOLOv3 took " << timeSpan.count() << " seconds." << endl;

    auto lineSegs = mpLineSegDetector->Detect(mpCurrentKeyFrame->mImGray);

    // Get Intrinsic and Extrinsic Matrix
    const auto M = mpCurrentKeyFrame->GetPose(); // 4 x 4 projection matrix
    const auto R = mpCurrentKeyFrame->GetRotation();
    const auto t = mpCurrentKeyFrame->GetTranslation();
    const auto& K = mpCurrentKeyFrame->mK; // 3 x 3 intrinsic matrix
    const auto invR = R.t();
    const auto invK = K.inv();

    // Compute the projection of the landmark centers for removing redundant objects.
    vector<Point2f> projCenters;
    auto landmarks = mpCurrentKeyFrame->GetLandmarks();
    projCenters.reserve(landmarks.size());
    for (const auto& pLandmark : landmarks) {
        projCenters.emplace_back(pLandmark->GetProjectedCentroid(mpCurrentKeyFrame->GetPose()));
    }

    t1 = high_resolution_clock::now();
    for (int objId = 0; objId < objects2D.size(); ++objId) {
        const auto object = objects2D[objId];
        auto& bbox = object.bbox;

        // Ignore objects that are too small.
        if (bbox.width <= 96 || bbox.height <= 96)
            continue;

        // Draw the bounding bbox.
        ObjectDetector::Draw(canvas, object);

        // Ignore the bounding box that goes outside the frame.
        if (bbox.x < 0 || bbox.y < 0
            || bbox.x + bbox.width >= mpCurrentKeyFrame->mImColor.cols
            || bbox.y + bbox.height >= mpCurrentKeyFrame->mImColor.rows) {
            continue;
        }

        // Remove objects already seen in previous keyframes.
        bool seen = false;
        for (int cIdx = 0; cIdx < projCenters.size(); ++cIdx) {
            // Check if the previous center is near the center of the current bounding box.
            const auto& center = projCenters[cIdx];
            if (max(center.x - bbox.x, center.y - bbox.y) <= min(bbox.width, bbox.height) >> 2) {
                seen = true;
                landmarks[cIdx]->bboxCenter[mpCurrentKeyFrame->mnFrameId] =
                        cv::Point2f(bbox.x + bbox.width * .5f, bbox.y + bbox.height * .5f);
                break;
            }
        }
        if (seen)
            continue;

        // Choose the line segments lying in the bounding box for scoring.
        vector<LineSegment*> segsInBbox;
        segsInBbox.reserve(lineSegs.size());
        for (auto& lineSeg : lineSegs) {
            if (lineSeg.first.inside(bbox) && lineSeg.second.inside(bbox)) {
                segsInBbox.emplace_back(&lineSeg);
            }
        }

        Landmark landmark;
        landmark.classIdx = object.classIdx;
        landmark.bboxCenter[mpCurrentKeyFrame->mnFrameId] = Point2f(bbox.x + bbox.width * .5f,
                                                                    bbox.y + bbox.height * .5f);

        // Find landmarks with respect to the detected objects.
        auto proposal = FindBestProposal(bbox, segsInBbox, K,
                                         mShapeErrThresh, mShapeErrWeight, mAlignErrWeight,
                                         -M_PI_F, 0,
                                         10 * M_PI_F / 180, 10 * M_PI_F / 180,
                                         mpCurrentKeyFrame->mnFrameId, objId,
                                         mpCurrentKeyFrame->mImColor, false, false);
        if (!proposal.valid)
            continue;

        proposal.Draw(canvas, K);

        // Reason the pose and dimension of the landmark from the best proposal.
        // Approximate the depth of the centroid to be the average of the map points that fall in the bounding box.
        auto mapPoints = mpCurrentKeyFrame->GetMapPointMatches();
        vector<MapPoint*> includedMapPoints;
        includedMapPoints.reserve(mapPoints.size());
        for (auto& mapPoint : mpCurrentKeyFrame->GetMapPoints()) {
            const auto& pt = mpCurrentKeyFrame->mvKeysUn[mapPoint->GetObservations()[mpCurrentKeyFrame]].pt;
            if (pt.inside(bbox)) {
                includedMapPoints.emplace_back(mapPoint);
            }
        }
        Mat worldAvgPos = (Mat_<float>(4, 1, CV_32F) << 0, 0, 0, 1);
        for (auto mapPoint : includedMapPoints) {
            worldAvgPos.rowRange(0, 3) += mapPoint->GetWorldPos();
        }
        worldAvgPos.rowRange(0, 3) /= includedMapPoints.size();
        Mat camCoordAvgPos = mpCurrentKeyFrame->GetPose() * worldAvgPos;
        float avgDepth = camCoordAvgPos.at<float>(2) / camCoordAvgPos.at<float>(3);

        // Recover pose.
        Mat centroid3D = proposal.GetCentroid3D(avgDepth, invK);
        Mat worldCentroid = mpCurrentKeyFrame->GetRotation() * centroid3D + mpCurrentKeyFrame->GetTranslation();
        Mat Rlw = proposal.Rlc * mpCurrentKeyFrame->GetRotation();
        Mat tlw = -Rlw * worldCentroid;
        landmark.SetPose(Rlw, tlw);

        // Recover the dimension of the landmark with the centroid and the proposal.
        auto dimension = proposal.ComputeDimension3D(centroid3D);
        landmark.SetDimension(dimension);

        // Store the pose corresponding to best proposal into the keyframe.
        mpCurrentKeyFrame->AddLandmark(make_shared<Landmark>(landmark));
    }
    t2 = high_resolution_clock::now();
    timeSpan = duration_cast<duration<double>>(t2 - t1);
    cout << "Finding landmarks took " << timeSpan.count() << " seconds." << endl;

    imwrite("Outputs/" + std::to_string(mpCurrentKeyFrame->mnFrameId) + "_best.jpg", canvas);

    // Visualize intermediate results used for finding landmarks.
    mpFrameDrawer->UpdateKeyframe(mpCurrentKeyFrame, objects2D);
}

} //namespace ORB_SLAM
