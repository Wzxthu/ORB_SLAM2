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

#include "KeyFrame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include<mutex>
#include <DepthEstimation/DepthEstimator.h>
#include <util/settings.h>
#include <thread>

namespace ORB_SLAM2
{

long unsigned int KeyFrame::nNextId=0;

inline cv::Mat SelectHighGradientPoints(const cv::Mat& imColor, int numPt) {
    using namespace cv;
    using namespace cnn_slam;

    // Find high gradient points.
    // For efficiency, we first down-sample the image by 4,
    // then find top TRACKING_NUM_PT/16 points with greatest gradient.
    // The finally selected points are points from the 4x4 pathces around the corresponding points.
    int numPatch = numPt >> 4;
    Mat smallIm;
    resize(imColor, smallIm, Size(imColor.cols >> 2, imColor.rows >> 2));
    GaussianBlur(smallIm, smallIm, Size(3,3), 0, 0, BORDER_DEFAULT );
    cvtColor(smallIm, smallIm, CV_BGR2GRAY);
    Mat grad;
    Laplacian(smallIm, grad, CV_16S, 3);
    convertScaleAbs(grad, grad);    // Convert to CV_8U image.

    // Sort the gradients.
    typedef pair<uchar, Point2i> GradInfo;
    vector<GradInfo> gradCoord;
    gradCoord.reserve(static_cast<unsigned long>(grad.rows * grad.cols));
    for (int i = 0; i < grad.rows; ++i) {
        auto gradRow = grad.ptr<uchar>(i);
        for (int j = 0; j < grad.cols; ++j) {
            gradCoord.emplace_back(gradRow[j],
                                   Point2i(j, i));
        }
    }
    sort(gradCoord.begin(), gradCoord.end(), [](const GradInfo &p1, const GradInfo &p2) {
        return p1.first > p2.first; // Sort in descending order.
    });

//    {
//        Mat canvas = grad.clone();
//        resize(canvas, canvas, imColor.size());
//        cvtColor(canvas, canvas, CV_GRAY2BGR);
//        cout << numPatch << endl;
//        for (int i = 0; i < numPatch; ++i) {
//            cout << int(gradCoord[i].first) << ' ' << int(grad.at<uchar>(gradCoord[i].second.y, gradCoord[i].second.x)) << endl;
//            cv::circle(canvas, Point(gradCoord[i].second.x << 2, gradCoord[i].second.y << 2), 2, Scalar(0, 255, 0));
//        }
//        imshow("Selected patches", canvas);
//        waitKey(0);
//    }

    // Recover original coordinates and extract data.
    Mat highGradPtHomo2dCoord = Mat::ones(numPt, 3, CV_32F);
    highGradPtHomo2dCoord.col(2) = Mat::ones(numPt, 1, CV_32F);
#pragma omp parallel for
    for (int i = 0; i < numPatch; ++i) {
        int x = gradCoord[i].second.x << 2;
        int y = gradCoord[i].second.y << 2;
        int row = i * 16;
        for (int dx = 0; dx < 4; ++dx) {
            for (int dy = 0; dy < 4; ++dy) {
                highGradPtHomo2dCoord.at<float>(row, 0) = x + dx;
                highGradPtHomo2dCoord.at<float>(row, 1) = y + dy;
                ++row;
            }
        }
    }

    return highGradPtHomo2dCoord;
}

KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB,
                   cv::Mat imColor,
                   cnn_slam::DepthEstimator *pDepthEstimator,
                   KeyFrame *pPrevKF,
                   float focalLength):
    mnFrameId(F.mnId),  mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
    mfGridElementWidthInv(F.mfGridElementWidthInv), mfGridElementHeightInv(F.mfGridElementHeightInv),
    mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),
    mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
    fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
    mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), N(F.N), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn),
    mvuRight(F.mvuRight), mvDepth(F.mvDepth), mDescriptors(F.mDescriptors.clone()),
    mBowVec(F.mBowVec), mFeatVec(F.mFeatVec), mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),
    mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
    mvInvLevelSigma2(F.mvInvLevelSigma2), mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX),
    mnMaxY(F.mnMaxY), mK(F.mK), mInvK(F.mK.inv()), mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB),
    mpORBvocabulary(F.mpORBvocabulary), mbFirstConnection(true), mpParent(NULL), mbNotErase(false),
    mbToBeErased(false), mbBad(false), mHalfBaseline(F.mb/2), mpMap(pMap), mbDepthReady(false), mbWorking(true)
{
    mnId=nNextId++;

    mGrid.resize(mnGridCols);
    for(int i=0; i<mnGridCols;i++)
    {
        mGrid[i].resize(mnGridRows);
        for(int j=0; j<mnGridRows; j++)
            mGrid[i][j] = F.mGrid[i][j];
    }

    SetPose(F.mTcw);

    if (!imColor.empty() and pDepthEstimator) {
        mHighGradPtHomo2dCoord = SelectHighGradientPoints(imColor, cnn_slam::TRACKING_NUM_PT);

        thread depth_estimation_thread([this, imColor, pDepthEstimator, pPrevKF, focalLength]() {
            EstimateDepth(imColor, pDepthEstimator, pPrevKF, focalLength);
        });
        depth_estimation_thread.detach();
    }
}

KeyFrame::~KeyFrame() {
    while (mbWorking)
        usleep(1000);
}

void KeyFrame::EstimateDepth(cv::Mat imColor, cnn_slam::DepthEstimator *pDepthEstimator, KeyFrame *pPrevKF, float focalLength) {
    mbWorking = true;
    mbDepthReady = false;

    using namespace cv;

    // Estimate depth.
    pDepthEstimator->EstimateDepth(imColor, mDepthMap, focalLength);
    imwrite("image.jpg", imColor);
    Mat depthDisplay;
    double minDepth, maxDepth;
    cv::minMaxLoc(mDepthMap, &minDepth, &maxDepth);
    depthDisplay = mDepthMap / maxDepth * 255;
    depthDisplay.convertTo(depthDisplay, CV_8U);
    imwrite("depth.jpg", depthDisplay);

    // Estimate uncertainty map.
    if (pPrevKF && !pPrevKF->GetPose().empty()) {
        // Calculate projected 2D location in the current frame
        // of the high gradient points in the reference keyframe.
        Mat depthVec = mDepthMap.reshape(0, mDepthMap.rows * mDepthMap.cols);

        Mat vertices(depthVec.rows, 3, CV_32F);
#pragma omp parallel for
        for (int i = 0; i < mDepthMap.rows; ++i) {
            int row_cnt = i * mDepthMap.cols;
            for (int j = 0; j < mDepthMap.cols; ++j)
                vertices.row(row_cnt++) = (Mat_<float>(1, 3) << j, i, 1) * mDepthMap.at<float>(i, j);
        }

        Mat Trel = pPrevKF->GetPose() * GetPoseInverse();
        vertices = (vertices * mInvK.t() * Trel.rowRange(0, 3).colRange(0, 3).t()
                    + repeat(Trel.col(3).rowRange(0, 3).t(), vertices.rows, 1)) * pPrevKF->mK.t();
        Mat proj2d = vertices.colRange(0, 2) / repeat(vertices.col(2), 1, 2);
        assert(proj2d.cols == 2);
        proj2d.convertTo(proj2d, CV_32S);

        Mat valid = proj2d.col(0) >= 0;
        bitwise_and(valid, proj2d.col(0) < pPrevKF->mDepthMap.cols, valid, valid);
        bitwise_and(valid, proj2d.col(1) >= 0, valid, valid);
        bitwise_and(valid, proj2d.col(1) < pPrevKF->mDepthMap.rows, valid, valid);

        // Estimate uncertainty on each valid point.
        mUncertaintyMap = Mat(mDepthMap.rows * mDepthMap.cols, 1, CV_32F);
#pragma omp parallel for
        for (int i = 0; i < valid.rows; ++i)
            if (valid.at<uchar>(i)) {
                auto proj_depth = pPrevKF->mDepthMap.at<float>(proj2d.at<int>(i, 1),
                                                               proj2d.at<int>(i, 0));
                auto proj_uncertainty = pPrevKF->mUncertaintyMap.at<float>(proj2d.at<int>(i, 1),
                                                                           proj2d.at<int>(i, 0));
                mUncertaintyMap.at<float>(i) = powf(depthVec.at<float>(i) - proj_depth, 2);
                depthVec.at<float>(i) = (depthVec.at<float>(i) * proj_uncertainty
                                         + proj_depth * mUncertaintyMap.at<float>(i)) /
                                        (proj_uncertainty + mUncertaintyMap.at<float>(i));
                mUncertaintyMap.at<float>(i) = (mUncertaintyMap.at<float>(i) * proj_uncertainty)
                                               / (mUncertaintyMap.at<float>(i) + proj_uncertainty);
            }
        mDepthMap = depthVec.reshape(0, mDepthMap.rows);

        // Set mean uncertainty to invalid points.
        mMeanUncertainty = static_cast<float>(mean(mUncertaintyMap, valid)[0]);
#pragma omp parallel for
        for (int i = 0; i < valid.rows; ++i)
            if (!valid.at<uchar>(i))
                mUncertaintyMap.at<float>(i) = mMeanUncertainty;

        // Reshape back to same as the depth map.
        mUncertaintyMap = mUncertaintyMap.reshape(0, mDepthMap.rows);
    } else {
        // No previous keyframe given.
        cv::pow(mDepthMap, 2, mUncertaintyMap);
        mMeanUncertainty = static_cast<float>(mean(mUncertaintyMap)[0]);
    }

    mHighGradPtDepth = Mat(mHighGradPtHomo2dCoord.rows, 1, CV_32F);
    mHighGradPtPixels = Mat(mHighGradPtHomo2dCoord.rows, 3, CV_8U);
    mHighGradPtSqrtUncertainty = Mat(mHighGradPtHomo2dCoord.rows, 1, CV_32F);
    mHighGradPtUncertainty = Mat(mHighGradPtHomo2dCoord.rows, 1, CV_32F);
#pragma omp parallel for
    for (int i = 0; i < mHighGradPtHomo2dCoord.rows; ++i) {
        auto x = static_cast<int>(mHighGradPtHomo2dCoord.at<float>(i, 0));
        auto y = static_cast<int>(mHighGradPtHomo2dCoord.at<float>(i, 1));
        mHighGradPtDepth.at<float>(i) = mDepthMap.at<float>(y, x);
        mHighGradPtUncertainty.at<float>(i) = mUncertaintyMap.at<float>(y, x);
        mHighGradPtPixels.row(i) = imColor.col(x).row(y).reshape(1, 3);
    }
    sqrt(mHighGradPtUncertainty, mHighGradPtSqrtUncertainty);

    Mat canvas = imColor.clone();
    for (int i = 0; i < mHighGradPtHomo2dCoord.rows; ++i)
        cv::circle(canvas, Point(static_cast<int>(mHighGradPtHomo2dCoord.at<float>(i, 0)),
                                 static_cast<int>(mHighGradPtHomo2dCoord.at<float>(i, 1))), 2, Scalar(255, 0, 0));
    imwrite("selected.jpg", canvas);

    mbDepthReady = true;
    mbWorking = false;
}

void KeyFrame::ComputeBoW()
{
    if(mBowVec.empty() || mFeatVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void KeyFrame::SetPose(const cv::Mat &Tcw_)
{
    unique_lock<mutex> lock(mMutexPose);
    if (Tcw_.type() == CV_32F)
        Tcw_.copyTo(Tcw);
    else
        Tcw_.convertTo(Tcw, CV_32F);
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    cv::Mat Rwc = Rcw.t();
    Ow = -Rwc*tcw;

    Twc = cv::Mat::eye(4,4,Tcw.type());
    Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
    Ow.copyTo(Twc.rowRange(0,3).col(3));
    cv::Mat center = (cv::Mat_<float>(4,1) << mHalfBaseline, 0 , 0, 1);
    Cw = Twc*center;
}

cv::Mat KeyFrame::GetPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.clone();
}

cv::Mat KeyFrame::GetPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose);
    return Twc.clone();
}

cv::Mat KeyFrame::GetCameraCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Ow.clone();
}

cv::Mat KeyFrame::GetStereoCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Cw.clone();
}


cv::Mat KeyFrame::GetRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).colRange(0,3).clone();
}

cv::Mat KeyFrame::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).col(3).clone();
}

void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(!mConnectedKeyFrameWeights.count(pKF))
            mConnectedKeyFrameWeights[pKF]=weight;
        else if(mConnectedKeyFrameWeights[pKF]!=weight)
            mConnectedKeyFrameWeights[pKF]=weight;
        else
            return;
    }

    UpdateBestCovisibles();
}

void KeyFrame::UpdateBestCovisibles()
{
    unique_lock<mutex> lock(mMutexConnections);
    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
       vPairs.push_back(make_pair(mit->second,mit->first));

    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for(size_t i=0, iend=vPairs.size(); i<iend;i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());    
}

set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    set<KeyFrame*> s;
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin();mit!=mConnectedKeyFrameWeights.end();mit++)
        s.insert(mit->first);
    return s;
}

vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}

vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    unique_lock<mutex> lock(mMutexConnections);
    if((int)mvpOrderedConnectedKeyFrames.size()<N)
        return mvpOrderedConnectedKeyFrames;
    else
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),mvpOrderedConnectedKeyFrames.begin()+N);

}

vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int &w)
{
    unique_lock<mutex> lock(mMutexConnections);

    if(mvpOrderedConnectedKeyFrames.empty())
        return vector<KeyFrame*>();

    vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(),mvOrderedWeights.end(),w,KeyFrame::weightComp);
    if(it==mvOrderedWeights.end())
        return vector<KeyFrame*>();
    else
    {
        int n = it-mvOrderedWeights.begin();
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+n);
    }
}

int KeyFrame::GetWeight(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexConnections);
    if(mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}

void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=pMP;
}

void KeyFrame::EraseMapPointMatch(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}

void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
{
    int idx = pMP->GetIndexInKeyFrame(this);
    if(idx>=0)
        mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}


void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP)
{
    mvpMapPoints[idx]=pMP;
}

set<MapPoint*> KeyFrame::GetMapPoints()
{
    unique_lock<mutex> lock(mMutexFeatures);
    set<MapPoint*> s;
    for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
    {
        if(!mvpMapPoints[i])
            continue;
        MapPoint* pMP = mvpMapPoints[i];
        if(!pMP->isBad())
            s.insert(pMP);
    }
    return s;
}

int KeyFrame::TrackedMapPoints(const int &minObs)
{
    unique_lock<mutex> lock(mMutexFeatures);

    int nPoints=0;
    const bool bCheckObs = minObs>0;
    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = mvpMapPoints[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(bCheckObs)
                {
                    if(mvpMapPoints[i]->Observations()>=minObs)
                        nPoints++;
                }
                else
                    nPoints++;
            }
        }
    }

    return nPoints;
}

vector<MapPoint*> KeyFrame::GetMapPointMatches()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}

MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}

void KeyFrame::UpdateConnections()
{
    map<KeyFrame*,int> KFcounter;

    vector<MapPoint*> vpMP;

    {
        unique_lock<mutex> lockMPs(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    //For all map points in keyframe check in which other keyframes are they seen
    //Increase counter for those keyframes
    for(vector<MapPoint*>::iterator vit=vpMP.begin(), vend=vpMP.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        if(!pMP)
            continue;

        if(pMP->isBad())
            continue;

        map<KeyFrame*,size_t> observations = pMP->GetObservations();

        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            if(mit->first->mnId==mnId)
                continue;
            KFcounter[mit->first]++;
        }
    }

    // This should not happen
    if(KFcounter.empty())
        return;

    //If the counter is greater than threshold add connection
    //In case no keyframe counter is over threshold add the one with maximum counter
    int nmax=0;
    KeyFrame* pKFmax=NULL;
    int th = 15;

    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(KFcounter.size());
    for(map<KeyFrame*,int>::iterator mit=KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++)
    {
        if(mit->second>nmax)
        {
            nmax=mit->second;
            pKFmax=mit->first;
        }
        if(mit->second>=th)
        {
            vPairs.push_back(make_pair(mit->second,mit->first));
            (mit->first)->AddConnection(this,mit->second);
        }
    }

    if(vPairs.empty())
    {
        vPairs.push_back(make_pair(nmax,pKFmax));
        pKFmax->AddConnection(this,nmax);
    }

    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for(size_t i=0; i<vPairs.size();i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    {
        unique_lock<mutex> lockCon(mMutexConnections);

        // mspConnectedKeyFrames = spConnectedKeyFrames;
        mConnectedKeyFrameWeights = KFcounter;
        mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

        if(mbFirstConnection && mnId!=0)
        {
            mpParent = mvpOrderedConnectedKeyFrames.front();
            mpParent->AddChild(this);
            mbFirstConnection = false;
        }

    }
}

void KeyFrame::AddChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}

void KeyFrame::EraseChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}

void KeyFrame::ChangeParent(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mpParent = pKF;
    pKF->AddChild(this);
}

set<KeyFrame*> KeyFrame::GetChilds()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens;
}

KeyFrame* KeyFrame::GetParent()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}

bool KeyFrame::hasChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}

void KeyFrame::AddLoopEdge(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspLoopEdges.insert(pKF);
}

set<KeyFrame*> KeyFrame::GetLoopEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspLoopEdges;
}

void KeyFrame::SetNotErase()
{
    unique_lock<mutex> lock(mMutexConnections);
    mbNotErase = true;
}

void KeyFrame::SetErase()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mspLoopEdges.empty())
        {
            mbNotErase = false;
        }
    }

    if(mbToBeErased)
    {
        SetBadFlag();
    }
}

void KeyFrame::SetBadFlag()
{   
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mnId==0)
            return;
        else if(mbNotErase)
        {
            mbToBeErased = true;
            return;
        }
    }

    for(map<KeyFrame*,int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
        mit->first->EraseConnection(this);

    for(size_t i=0; i<mvpMapPoints.size(); i++)
        if(mvpMapPoints[i])
            mvpMapPoints[i]->EraseObservation(this);
    {
        unique_lock<mutex> lock(mMutexConnections);
        unique_lock<mutex> lock1(mMutexFeatures);

        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        // Update Spanning Tree
        set<KeyFrame*> sParentCandidates;
        sParentCandidates.insert(mpParent);

        // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
        // Include that children as new parent candidate for the rest
        while(!mspChildrens.empty())
        {
            bool bContinue = false;

            int max = -1;
            KeyFrame* pC;
            KeyFrame* pP;

            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(), send=mspChildrens.end(); sit!=send; sit++)
            {
                KeyFrame* pKF = *sit;
                if(pKF->isBad())
                    continue;

                // Check if a parent candidate is connected to the keyframe
                vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                for(size_t i=0, iend=vpConnected.size(); i<iend; i++)
                {
                    for(set<KeyFrame*>::iterator spcit=sParentCandidates.begin(), spcend=sParentCandidates.end(); spcit!=spcend; spcit++)
                    {
                        if(vpConnected[i]->mnId == (*spcit)->mnId)
                        {
                            int w = pKF->GetWeight(vpConnected[i]);
                            if(w>max)
                            {
                                pC = pKF;
                                pP = vpConnected[i];
                                max = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }

            if(bContinue)
            {
                pC->ChangeParent(pP);
                sParentCandidates.insert(pC);
                mspChildrens.erase(pC);
            }
            else
                break;
        }

        // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
        if(!mspChildrens.empty())
            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(); sit!=mspChildrens.end(); sit++)
            {
                (*sit)->ChangeParent(mpParent);
            }

        mpParent->EraseChild(this);
        mTcp = Tcw*mpParent->GetPoseInverse();
        mbBad = true;
    }


    mpMap->EraseKeyFrame(this);
    mpKeyFrameDB->erase(this);
}

bool KeyFrame::isBad()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mbBad;
}

void KeyFrame::EraseConnection(KeyFrame* pKF)
{
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mConnectedKeyFrameWeights.count(pKF))
        {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate=true;
        }
    }

    if(bUpdate)
        UpdateBestCovisibles();
}

vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=mnGridCols)
        return vIndices;

    const int nMaxCellX = min((int)mnGridCols-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=mnGridRows)
        return vIndices;

    const int nMaxCellY = min((int)mnGridRows-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool KeyFrame::IsInImage(const float &x, const float &y) const
{
    return (x>=mnMinX && x<mnMaxX && y>=mnMinY && y<mnMaxY);
}

cv::Mat KeyFrame::UnprojectStereo(int i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeys[i].pt.x;
        const float v = mvKeys[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);

        unique_lock<mutex> lock(mMutexPose);
        return Twc.rowRange(0,3).colRange(0,3)*x3Dc+Twc.rowRange(0,3).col(3);
    }
    else
        return cv::Mat();
}

float KeyFrame::ComputeSceneMedianDepth(const int q)
{
    vector<MapPoint*> vpMapPoints;
    cv::Mat Tcw_;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPose);
        vpMapPoints = mvpMapPoints;
        Tcw_ = Tcw.clone();
    }

    vector<float> vDepths;
    vDepths.reserve(N);
    cv::Mat Rcw2 = Tcw_.row(2).colRange(0,3);
    Rcw2 = Rcw2.t();
    float zcw = Tcw_.at<float>(2,3);
    for(int i=0; i<N; i++)
    {
        if(mvpMapPoints[i])
        {
            MapPoint* pMP = mvpMapPoints[i];
            cv::Mat x3Dw = pMP->GetWorldPos();
            float z = Rcw2.dot(x3Dw)+zcw;
            vDepths.push_back(z);
        }
    }

    sort(vDepths.begin(),vDepths.end());

    return vDepths[(vDepths.size()-1)/q];
}

} //namespace ORB_SLAM
