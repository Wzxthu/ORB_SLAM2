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


#include "Tracking.h"

#include<opencv2/core/eigen.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"
#include <Tracking/PoseEstimation.h>

#include"Optimizer.h"
#include"PnPsolver.h"
#include"util/settings.h"
#include"util/global_Funcs.h"

#include<iostream>

#include<mutex>
#include <unistd.h>
#include <DepthEstimation/DepthEstimatorFCRN.h>
#include <DepthEstimation/DepthEstimatorFake.h>


using namespace std;

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    if(sensor==System::MONOCULAR)
        mpDepthEstimator = new cnn_slam::DepthEstimatorFCRN;
    else
        mpDepthEstimator = new cnn_slam::DepthEstimatorFake;
    mpDepthEstimator->Initialize();

    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);
    mInvK = mK.inv();

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;
    mFPS = fps;

    float cameraPixelNoise = fSettings["Camera.PixelNoise"];
    if (cameraPixelNoise <= 0)
        cameraPixelNoise = 4;
    mCameraPixelNoise2 = static_cast<float>(pow(cameraPixelNoise, 2));

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}


cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    // Change the channel order to RGB to fit the depth prediction network.
    mImColor = imRGB;
    if (mImColor.channels() == 3) {
        if (!mbRGB)
            cvtColor(mImColor, mImColor, CV_BGR2RGB);
    } else if (mImColor.channels() == 4) {
        if (mbRGB)
            cvtColor(mImColor, mImColor, CV_RGBA2RGB);
        else
            cvtColor(mImColor, mImColor, CV_BGRA2RGB);
    } else
        cvtColor(mImColor, mImColor, CV_GRAY2RGB);

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    static_cast<cnn_slam::DepthEstimatorFake *>(mpDepthEstimator)->SetDepthMap(imDepth);

    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im.clone();

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    // Change the channel order to RGB to fit the depth prediction network.
    mImColor = im;
    if (mImColor.channels() == 3) {
        if (!mbRGB)
            cvtColor(mImColor, mImColor, CV_BGR2RGB);
    } else if (mImColor.channels() == 4) {
        if (mbRGB)
            cvtColor(mImColor, mImColor, CV_RGBA2RGB);
        else
            cvtColor(mImColor, mImColor, CV_BGRA2RGB);
    } else
        cvtColor(mImColor, mImColor, CV_GRAY2RGB);

    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

void Tracking::Track()
{
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();
        else
            MonocularInitialization();

        mpFrameDrawer->Update(this);

        if(mState!=OK)
            return;
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.

            if(mState==OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    bOK = TrackWithMotionModel();
                    if(!bOK)
                        bOK = TrackReferenceKeyFrame();
                }
            }
            else
            {
                bOK = Relocalization();
            }
        }
        else
        {
            // Localization Mode: Local Mapping is deactivated

            if(mState==LOST)
            {
                bOK = Relocalization();
            }
            else
            {
                if(!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map

                    if(!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if(!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    bOKReloc = Relocalization();

                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO)
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(!mbOnlyTracking)
        {
            if(bOK)
                bOK = TrackLocalMap();
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        if(bOK)
            mState = OK;
        else
            mState=LOST;

        // Update drawer
        mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        if(bOK)
        {
            // Update motion model
            if(!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame.mTcw*LastTwc;
            }
            else
                mVelocity = cv::Mat();

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            if(NeedNewKeyFrame())
                CreateNewKeyFrame();

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

}


void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB,
                                        mImColor,mpDepthEstimator,nullptr,mCurrentFrame.focalLength);

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;
    }
}

void Tracking::MonocularInitialization()
{

    if(!mpInitializer)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>100)
        {
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)
                delete mpInitializer;

            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            return;
        }
    }
    else
    {
        // Try to initialize
        if((int)mCurrentFrame.mvKeys.size()<=100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        if(nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

            CreateInitialMapMonocular();
        }
    }
}

void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB,
                                    mImColor,mpDepthEstimator,nullptr,mInitialFrame.focalLength);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB,
                                    mImColor,mpDepthEstimator,pKFini,mCurrentFrame.focalLength);


    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    Optimizer::GlobalBundleAdjustemnt(mpMap,20);

    // Set median depth to 1
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}


bool Tracking::makeAndCheckEPL(const int x, const int y, KeyFrame* const ref, float* pepx, float* pepy)
{

    // ======= make epl ========
    // calculate the plane spanned by the two camera centers and the point (x,y,1)
    // intersect it with the keyframe's image plane (at depth=1)
    cv::Mat thisToOther = mCurrentFrame.mTcw * ref->GetPoseInverse();
    float epx = - ref->fx * thisToOther.at<float>(2,0) + thisToOther.at<float>(2,2)*(x - ref->cx);
    float epy = - ref->fy * thisToOther.at<float>(2,1) + thisToOther.at<float>(2,2)*(y - ref->cy);

    if(std::isnan(epx+epy))
        return false;


    // ======== check epl length =========
    float eplLengthSquared = epx*epx+epy*epy;
    if(eplLengthSquared < MIN_EPL_LENGTH_SQUARED)
    {
        return false;
    }


    // ===== check epl-grad magnitude ======
    float gx = mImGray.at<float>(y,x) - mImGray.at<float>(y,x-1);
    float gy = mImGray.at<float>(y,x) - mImGray.at<float>(y-1,x);
    float eplGradSquared = gx * epx + gy * epy;
    eplGradSquared = eplGradSquared*eplGradSquared / eplLengthSquared;	// square and norm with epl-length

    if(eplGradSquared < MIN_EPL_GRAD_SQUARED)
    {
        return false;
    }


    // ===== check epl-grad angle ======
    if(eplGradSquared / (gx*gx+gy*gy) < MIN_EPL_ANGLE_SQUARED)
    {
        return false;
    }


    // ===== DONE - return "normalized" epl =====
    float fac = GRADIENT_SAMPLE_DIST / sqrt(eplLengthSquared);
    *pepx = epx * fac;
    *pepy = epy * fac;

    return true;
}


bool Tracking::observeDepthUpdate(const int &x, const int &y, KeyFrame* const ref)
{
    float epx, epy;
    bool isGood = makeAndCheckEPL(x, y, ref, &epx, &epy);
    if(!isGood) return false;

    // which exact point to track, and where from.
    float depth = 1.0f / ref->mDepthMap.at<float>(y,x);
    float sv = sqrt(depth);
    float min_idepth = depth - sv*STEREO_EPL_VAR_FAC;
    float max_idepth = depth + sv*STEREO_EPL_VAR_FAC;
    if(min_idepth < 0) min_idepth = 0;
    if(max_idepth > 1/MIN_DEPTH) max_idepth = 1/MIN_DEPTH;


    float result_idepth, result_var, result_eplLength;

    float error = doLineStereo(
            x, y, epx, epy,
            min_idepth, depth ,max_idepth,
            ref, result_idepth, result_var, result_eplLength);

    float var = 1.0f / ref->mUncertaintyMap.at<float>(y,x);
    if(error == -1 or error == -2 or error == -3 or error == -4) {
        return false;
    }
    else
    {

        // do textbook ekf update:
        // increase var by a little (prediction-uncertainty)
        float id_var = var*SUCC_VAR_INC_FAC;

        // update var with observation
        float w = result_var / (result_var + id_var);
        float new_idepth = (1-w)*result_idepth + w*depth;
        // porpogation
        cv::Mat thisToOther_t = mCurrentFrame.mTcw * ref->GetPoseInverse();
        float tz = thisToOther_t.at<float>(2, 3);
        float depth_t = 1.0f / UNZERO(1.0f / new_idepth - tz);


        // variance can only decrease from observation; never increase.
        id_var = id_var * w;
//        if(id_var < var) {
            float uncentainty_t = pow(1.0f / ref->mDepthMap.at<float>(y, x) * new_idepth, 4) * id_var + result_var;
            //fusion
            float ww = 1.0f / ref->mUncertaintyMap.at<float>(y, x) * uncentainty_t;
            float depth_k = (1.0f / ref->mDepthMap.at<float>(y, x) + ww * depth_t) / (1 + ww);
            float uncertainty_k = 1.0f / ref->mUncertaintyMap.at<float>(y, x) * (1 + ww);
            ref->mDepthMap.at<float>(y, x) = 1.0f / depth_k;
            ref->mUncertaintyMap.at<float>(y, x) = 1.0f / uncertainty_k;
//        }

        return true;
    }
}

float Tracking::doLineStereo(
            const float u, const float v, const float epxn, const float epyn,
            const float min_idepth, const float prior_idepth, float max_idepth,
            KeyFrame* const ref,
            float &result_idepth, float &result_var, float &result_eplLength)
{
    // find pixel in image (do stereo along epipolar line).
    // mat: NEW image
    // KinvP: point in OLD image (Kinv * (u_old, v_old, 1)), projected
    // trafo: x_old = trafo * x_new; (from new to old image)
    // realVal: descriptor in OLD image.
    // returns: result_idepth : point depth in new camera's coordinate system
    // returns: result_u/v : point's coordinates in new camera's coordinate system
    // returns: idepth_var: (approximated) measurement variance of inverse depth of result_point_NEW
    // returns error if sucessful; -1 if out of bounds, -2 if not found.
    {
        float fxi = ref->mInvK.at<float>(0, 0);
        float fyi = ref->mInvK.at<float>(1, 1);
        float cxi = ref->mInvK.at<float>(0, 2);
        float cyi = ref->mInvK.at<float>(1, 2);

        cv::Mat referenceFrameImage;
        cv::cvtColor(ref->mImColor, referenceFrameImage, CV_RGB2GRAY);
        cv::Mat FrameImageData = mImGray;
        cv::Size s = referenceFrameImage.size();

        float height = s.height;
        float width = s.width;

        cv::Mat thisToOther_t = mCurrentFrame.mTcw * ref->GetPoseInverse();
        cv::Mat T = ref->GetPose() * mCurrentFrame.mTcw.inv();
        cv::Mat KT = cv::Mat::zeros(cv::Size(4, 4), CV_64FC1);
        KT.colRange(0, 4).rowRange(0, 3) = ref->mK * T.colRange(0, 4).rowRange(0, 3);
        KT.at<float>(3, 3) = 1;
        
        cv::Mat R = T.colRange(0, 3).rowRange(0, 3);
        cv::Mat t = T.col(3).rowRange(0, 3);
        cv::Mat KR = KT.colRange(0, 3).rowRange(0, 3);
        cv::Mat Kt = KT.col(3).rowRange(0, 3);

        Eigen::Matrix3f otherToThis_R;
        Eigen::Vector3f otherToThis_t;
        Eigen::Matrix3f K_otherToThis_R;
        Eigen::Vector3f K_otherToThis_t;

        cv::cv2eigen(R, otherToThis_R);
        cv::cv2eigen(t, otherToThis_t);
        cv::cv2eigen(KR, K_otherToThis_R);
        cv::cv2eigen(Kt, K_otherToThis_t);

        Eigen::Vector3f otherToThis_R_row0 = otherToThis_R.col(0);
        Eigen::Vector3f otherToThis_R_row1 = otherToThis_R.col(1);
        Eigen::Vector3f otherToThis_R_row2 = otherToThis_R.col(2);


        // calculate epipolar line start and end point in old image
        Eigen::Vector3f KinvP = Eigen::Vector3f(fxi*u+cxi,fyi*v+cyi,1);
        Eigen::Vector3f pInf = K_otherToThis_R * KinvP;
        Eigen::Vector3f pReal = pInf / prior_idepth + K_otherToThis_t;

        float rescaleFactor = pReal[2] * prior_idepth;

        float firstX = u - 2*epxn*rescaleFactor;
        float firstY = v - 2*epyn*rescaleFactor;
        float lastX = u + 2*epxn*rescaleFactor;
        float lastY = v + 2*epyn*rescaleFactor;
        // width - 2 and height - 2 comes from the one-sided gradient calculation at the bottom
        if (firstX <= 0 || firstX >= width - 2
            || firstY <= 0 || firstY >= height - 2
            || lastX <= 0 || lastX >= width - 2
            || lastY <= 0 || lastY >= height - 2) {
            return -1;
        }

        if(!(rescaleFactor > 0.7f && rescaleFactor < 1.4f))
        {
            return -1;
        }

        // calculate values to search for
        float realVal_p1 = cnn_slam::getInterpolatedElement(FrameImageData,u + epxn*rescaleFactor, v + epyn*rescaleFactor);
        float realVal_m1 = cnn_slam::getInterpolatedElement(FrameImageData,u - epxn*rescaleFactor, v - epyn*rescaleFactor);
        float realVal = cnn_slam::getInterpolatedElement(FrameImageData,u, v);
        float realVal_m2 = cnn_slam::getInterpolatedElement(FrameImageData,u - 2*epxn*rescaleFactor, v - 2*epyn*rescaleFactor);
        float realVal_p2 = cnn_slam::getInterpolatedElement(FrameImageData,u + 2*epxn*rescaleFactor, v + 2*epyn*rescaleFactor);

        //	if(K_otherToThis_t[2] * max_idepth + pInf[2] < 0.01)


        Eigen::Vector3f pClose = pInf + K_otherToThis_t*max_idepth;
        // if the assumed close-point lies behind the
        // image, have to change that.
        if(pClose[2] < 0.001)
        {
            max_idepth = (0.001-pInf[2]) / K_otherToThis_t[2];
            pClose = pInf + K_otherToThis_t*max_idepth;
        }
        pClose = pClose / pClose[2]; // pos in new image of point (xy), assuming max_idepth

        Eigen::Vector3f pFar = pInf + K_otherToThis_t*min_idepth;
        // if the assumed far-point lies behind the image or closter than the near-point,
        // we moved past the Point it and should stop.
        if(pFar[2] < 0.001 || max_idepth < min_idepth)
        {
            return -1;
        }
        pFar = pFar / pFar[2]; // pos in new image of point (xy), assuming min_idepth


        // check for nan due to eg division by zero.
        if(std::isnan((float)(pFar[0]+pClose[0])))
            return -4;

        // calculate increments in which we will step through the epipolar line.
        // they are sampleDist (or half sample dist) long
        float incx = pClose[0] - pFar[0];
        float incy = pClose[1] - pFar[1];
        float eplLength = sqrt(incx*incx+incy*incy);
        if(!eplLength > 0 || std::isinf(eplLength)) return -4;

        if(eplLength > MAX_EPL_LENGTH_CROP)
        {
            pClose[0] = pFar[0] + incx*MAX_EPL_LENGTH_CROP/eplLength;
            pClose[1] = pFar[1] + incy*MAX_EPL_LENGTH_CROP/eplLength;
        }

        incx *= GRADIENT_SAMPLE_DIST/eplLength;
        incy *= GRADIENT_SAMPLE_DIST/eplLength;


        // extend one sample_dist to left & right.
        pFar[0] -= incx;
        pFar[1] -= incy;
        pClose[0] += incx;
        pClose[1] += incy;


        // make epl long enough (pad a little bit).
        if(eplLength < MIN_EPL_LENGTH_CROP)
        {
            float pad = (MIN_EPL_LENGTH_CROP - (eplLength)) / 2;
            pFar[0] -= incx*pad;
            pFar[1] -= incy*pad;

            pClose[0] += incx*pad;
            pClose[1] += incy*pad;
        }

        // if inf point is outside of image: skip pixel.
        if(
                pFar[0] <= SAMPLE_POINT_TO_BORDER ||
                pFar[0] >= width-SAMPLE_POINT_TO_BORDER ||
                pFar[1] <= SAMPLE_POINT_TO_BORDER ||
                pFar[1] >= height-SAMPLE_POINT_TO_BORDER)
        {
            return -1;
        }



        // if near point is outside: move inside, and test length again.
        if(
                pClose[0] <= SAMPLE_POINT_TO_BORDER ||
                pClose[0] >= width-SAMPLE_POINT_TO_BORDER ||
                pClose[1] <= SAMPLE_POINT_TO_BORDER ||
                pClose[1] >= height-SAMPLE_POINT_TO_BORDER)
        {
            if(pClose[0] <= SAMPLE_POINT_TO_BORDER)
            {
                float toAdd = (SAMPLE_POINT_TO_BORDER - pClose[0]) / incx;
                pClose[0] += toAdd * incx;
                pClose[1] += toAdd * incy;
            }
            else if(pClose[0] >= width-SAMPLE_POINT_TO_BORDER)
            {
                float toAdd = (width-SAMPLE_POINT_TO_BORDER - pClose[0]) / incx;
                pClose[0] += toAdd * incx;
                pClose[1] += toAdd * incy;
            }

            if(pClose[1] <= SAMPLE_POINT_TO_BORDER)
            {
                float toAdd = (SAMPLE_POINT_TO_BORDER - pClose[1]) / incy;
                pClose[0] += toAdd * incx;
                pClose[1] += toAdd * incy;
            }
            else if(pClose[1] >= height-SAMPLE_POINT_TO_BORDER)
            {
                float toAdd = (height-SAMPLE_POINT_TO_BORDER - pClose[1]) / incy;
                pClose[0] += toAdd * incx;
                pClose[1] += toAdd * incy;
            }

            // get new epl length
            float fincx = pClose[0] - pFar[0];
            float fincy = pClose[1] - pFar[1];
            float newEplLength = sqrt(fincx*fincx+fincy*fincy);

            // test again
            if(
                    pClose[0] <= SAMPLE_POINT_TO_BORDER ||
                    pClose[0] >= width-SAMPLE_POINT_TO_BORDER ||
                    pClose[1] <= SAMPLE_POINT_TO_BORDER ||
                    pClose[1] >= height-SAMPLE_POINT_TO_BORDER ||
                    newEplLength < 8
                    )
            {
                return -1;
            }


        }


        // from here on:
        // - pInf: search start-point
        // - p0: search end-point
        // - incx, incy: search steps in pixel
        // - eplLength, min_idepth, max_idepth: determines search-resolution, i.e. the result's variance.


        float cpx = pFar[0];
        float cpy =  pFar[1];

        float val_cp_m2 = cnn_slam::getInterpolatedElement(referenceFrameImage,cpx-2*incx, cpy-2*incy);
        float val_cp_m1 = cnn_slam::getInterpolatedElement(referenceFrameImage,cpx-incx, cpy-incy);
        float val_cp = cnn_slam::getInterpolatedElement(referenceFrameImage,cpx, cpy);
        float val_cp_p1 = cnn_slam::getInterpolatedElement(referenceFrameImage,cpx+incx, cpy+incy);
        float val_cp_p2;



        /*
         * Subsequent exact minimum is found the following way:
         * - assuming lin. interpolation, the gradient of Error at p1 (towards p2) is given by
         *   dE1 = -2sum(e1*e1 - e1*e2)
         *   where e1 and e2 are summed over, and are the residuals (not squared).
         *
         * - the gradient at p2 (coming from p1) is given by
         * 	 dE2 = +2sum(e2*e2 - e1*e2)
         *
         * - linear interpolation => gradient changes linearely; zero-crossing is hence given by
         *   p1 + d*(p2-p1) with d = -dE1 / (-dE1 + dE2).
         *
         *
         *
         * => I for later exact min calculation, I need sum(e_i*e_i),sum(e_{i-1}*e_{i-1}),sum(e_{i+1}*e_{i+1})
         *    and sum(e_i * e_{i-1}) and sum(e_i * e_{i+1}),
         *    where i is the respective winning index.
         */


        // walk in equally sized steps, starting at depth=infinity.
        int loopCounter = 0;
        float best_match_x = -1;
        float best_match_y = -1;
        float best_match_err = 1e50;
        float second_best_match_err = 1e50;

        // best pre and post errors.
        float best_match_errPre=NAN, best_match_errPost=NAN, best_match_DiffErrPre=NAN, best_match_DiffErrPost=NAN;
        bool bestWasLastLoop = false;

        float eeLast = -1; // final error of last comp.

        // alternating intermediate vars
        float e1A=NAN, e1B=NAN, e2A=NAN, e2B=NAN, e3A=NAN, e3B=NAN, e4A=NAN, e4B=NAN, e5A=NAN, e5B=NAN;

        int loopCBest=-1, loopCSecond =-1;
        while(((incx < 0) == (cpx > pClose[0]) && (incy < 0) == (cpy > pClose[1])) || loopCounter == 0)
        {
            // interpolate one new point
            val_cp_p2 = cnn_slam::getInterpolatedElement(referenceFrameImage,cpx+2*incx, cpy+2*incy);


            // hacky but fast way to get error and differential error: switch buffer variables for last loop.
            float ee = 0;
            if(loopCounter%2==0)
            {
                // calc error and accumulate sums.
                e1A = val_cp_p2 - realVal_p2;ee += e1A*e1A;
                e2A = val_cp_p1 - realVal_p1;ee += e2A*e2A;
                e3A = val_cp - realVal;      ee += e3A*e3A;
                e4A = val_cp_m1 - realVal_m1;ee += e4A*e4A;
                e5A = val_cp_m2 - realVal_m2;ee += e5A*e5A;
            }
            else
            {
                // calc error and accumulate sums.
                e1B = val_cp_p2 - realVal_p2;ee += e1B*e1B;
                e2B = val_cp_p1 - realVal_p1;ee += e2B*e2B;
                e3B = val_cp - realVal;      ee += e3B*e3B;
                e4B = val_cp_m1 - realVal_m1;ee += e4B*e4B;
                e5B = val_cp_m2 - realVal_m2;ee += e5B*e5B;
            }


            // do I have a new winner??
            // if so: set.
            if(ee < best_match_err)
            {
                // put to second-best
                second_best_match_err=best_match_err;
                loopCSecond = loopCBest;

                // set best.
                best_match_err = ee;
                loopCBest = loopCounter;

                best_match_errPre = eeLast;
                best_match_DiffErrPre = e1A*e1B + e2A*e2B + e3A*e3B + e4A*e4B + e5A*e5B;
                best_match_errPost = -1;
                best_match_DiffErrPost = -1;

                best_match_x = cpx;
                best_match_y = cpy;
                bestWasLastLoop = true;
            }
                // otherwise: the last might be the current winner, in which case i have to save these values.
            else
            {
                if(bestWasLastLoop)
                {
                    best_match_errPost = ee;
                    best_match_DiffErrPost = e1A*e1B + e2A*e2B + e3A*e3B + e4A*e4B + e5A*e5B;
                    bestWasLastLoop = false;
                }

                // collect second-best:
                // just take the best of all that are NOT equal to current best.
                if(ee < second_best_match_err)
                {
                    second_best_match_err=ee;
                    loopCSecond = loopCounter;
                }
            }


            // shift everything one further.
            eeLast = ee;
            val_cp_m2 = val_cp_m1; val_cp_m1 = val_cp; val_cp = val_cp_p1; val_cp_p1 = val_cp_p2;


            cpx += incx;
            cpy += incy;

            loopCounter++;
        }

        // if error too big, will return -3, otherwise -2.
        if(best_match_err > 4*(float)MAX_ERROR_STEREO)
        {
            return -3;
        }


        // check if clear enough winner
        if(abs(loopCBest - loopCSecond) > 1 && MIN_DISTANCE_ERROR_STEREO * best_match_err > second_best_match_err)
        {
            return -2;
        }

        bool didSubpixel = false;


        // sampleDist is the distance in pixel at which the realVal's were sampled
        float sampleDist = GRADIENT_SAMPLE_DIST*rescaleFactor;

        float gradAlongLine = 0;
        float tmp = realVal_p2 - realVal_p1;  gradAlongLine+=tmp*tmp;
        tmp = realVal_p1 - realVal;  gradAlongLine+=tmp*tmp;
        tmp = realVal - realVal_m1;  gradAlongLine+=tmp*tmp;
        tmp = realVal_m1 - realVal_m2;  gradAlongLine+=tmp*tmp;

        gradAlongLine /= sampleDist*sampleDist;

        // check if interpolated error is OK. use evil hack to allow more error if there is a lot of gradient.
        if(best_match_err > (float)MAX_ERROR_STEREO + sqrtf( gradAlongLine)*20)
        {
            return -3;
        }


        // ================= calc depth (in KF) ====================
        // * KinvP = Kinv * (x,y,1); where x,y are pixel coordinates of point we search for, in the KF.
        // * best_match_x = x-coordinate of found correspondence in the reference frame.

        float idnew_best_match;	// depth in the new image
        float alpha; // d(idnew_best_match) / d(disparity in pixel) == conputed inverse depth derived by the pixel-disparity.
        if(incx*incx>incy*incy)
        {
            float oldX = fxi*best_match_x+cxi;
            float nominator = (oldX*otherToThis_t[2] - otherToThis_t[0]);
            float dot0 = KinvP.dot(otherToThis_R_row0);
            float dot2 = KinvP.dot(otherToThis_R_row2);

            idnew_best_match = (dot0 - oldX*dot2) / nominator;
            alpha = incx*fxi*(dot0*otherToThis_t[2] - dot2*otherToThis_t[0]) / (nominator*nominator);

        }
        else
        {
            float oldY = fyi*best_match_y+cyi;

            float nominator = (oldY*otherToThis_t[2] - otherToThis_t[1]);
            float dot1 = KinvP.dot(otherToThis_R_row1);
            float dot2 = KinvP.dot(otherToThis_R_row2);

            idnew_best_match = (dot1 - oldY*dot2) / nominator;
            alpha = incy*fyi*(dot1*otherToThis_t[2] - dot2*otherToThis_t[1]) / (nominator*nominator);

        }


        if(idnew_best_match < 0)
        {
            return -2;
        }


        // ================= calc var (in NEW image) ====================

        // calculate error from photometric noise
        float photoDispError = 4 * mCameraPixelNoise2 / (gradAlongLine + DIVISION_EPS);

        float trackingErrorFac = 0.25 * (1 + ref->initialTrackedResidual);

        // calculate error from geometric noise (wrong camera pose / calibration)
        Eigen::Vector2f gradsInterp;
        cv::Mat Dx, Dy;
        cv::Sobel(mImGray, Dx, CV_64F, 1, 0, 3);
        cv::Sobel(mImGray, Dy, CV_64F, 0, 1, 3);
        gradsInterp[0] = cnn_slam::getInterpolatedElement(Dx, u, v);
        gradsInterp[1] = cnn_slam::getInterpolatedElement(Dy, u, v);
        float geoDispError = (gradsInterp[0]*epxn + gradsInterp[1]*epyn) + DIVISION_EPS;
        geoDispError = trackingErrorFac*trackingErrorFac*(gradsInterp[0]*gradsInterp[0] + gradsInterp[1]*gradsInterp[1]) / (geoDispError*geoDispError);


        //geoDispError *= (0.5 + 0.5 *result_idepth) * (0.5 + 0.5 *result_idepth);

        // final error consists of a small constant part (discretization error),
        // geometric and photometric error.
        result_var = alpha*alpha*((didSubpixel ? 0.05f : 0.5f)*sampleDist*sampleDist +  geoDispError + photoDispError);	// square to make variance

        result_idepth = idnew_best_match;

        result_eplLength = eplLength;

        return best_match_err;
    }

}


void Tracking::DepthRefinement(KeyFrame* mpReferenceKF)
{
    float fxi = mpReferenceKF->mInvK.at<float>(0, 0);
    float fyi = mpReferenceKF->mInvK.at<float>(1, 1);
    float cxi = mpReferenceKF->mInvK.at<float>(0, 2);
    float cyi = mpReferenceKF->mInvK.at<float>(1, 2);
    cv::Mat T = mCurrentFrame.mTcw * mpReferenceKF->GetPoseInverse();
    Eigen::Vector3f trafoInv_t;
    Eigen::Matrix3f trafoInv_R;
    cv::Mat R = T.colRange(0, 3).rowRange(0, 3);
    cv::Mat t = T.col(3).rowRange(0, 3);
    cv::cv2eigen(R, trafoInv_R);
    cv::cv2eigen(t, trafoInv_t);
    cv::Size s = mImGray.size();
    float height = s.height;
    float width = s.width;

    for(int y = mCurrentFrame.mnMinY; y < mCurrentFrame.mnMaxY; y++)
        for(int x = mCurrentFrame.mnMinX; x < mCurrentFrame.mnMaxX; x++) {

            Eigen::Vector3f pn = (trafoInv_R * Eigen::Vector3f(x*fxi + cxi,y*fyi + cyi,1.0f)) * mpReferenceKF->mDepthMap.at<float>(y, x) + trafoInv_t;

            float new_idepth = 1.0f / pn[2];

            float u_new = pn[0]*new_idepth*mCurrentFrame.fx + mCurrentFrame.cx;
            float v_new = pn[1]*new_idepth*mCurrentFrame.fy + mCurrentFrame.cy;

            // check if still within image, if not: DROP.
            if(!(u_new > 2.1f && v_new > 2.1f && u_new < width-3.1f && v_new < height-3.1f))
            {
                continue;
            }


            if (observeDepthUpdate(x, y, mpReferenceKF)) {
                printf("Update Depth & Uncertainty successfully!\n");
            }
            else {
//                printf("Update Depth & Uncertainty unsuccessfully!\n");
            }
        }
}


bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    if(nmatches<15)
        return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    Optimizer::PoseOptimization(&mCurrentFrame);
    cv::Mat Tcw;
    cv::Mat initTcw = mLastFrame.mTcw;
//    cv::Mat initTcw = mCurrentFrame.mTcw;
    cnn_slam::EstimateCameraPose(mImColor, mK, mInvK, mpReferenceKF, mCameraPixelNoise2,
                                 cnn_slam::TRACKING_SOLVER_TIMECOST_RATIO / mFPS, Tcw, initTcw);
    cout << "Adjustment: ";
    cnn_slam::PrintRotTrans(Tcw * mCurrentFrame.mTcw.inv());
    mCurrentFrame.SetPose(Tcw);

    // Frame-wise depth refinement
//    DepthRefinement(mpReferenceKF);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    return nmatchesMap>=10;
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr*pRef->GetPose());

    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}

bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();

    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
    }

    if(nmatches<20)
        return false;

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);
    cv::Mat Tcw;
    cv::Mat initTcw = mLastFrame.mTcw;
//    cv::Mat initTcw = mCurrentFrame.mTcw;
    cnn_slam::EstimateCameraPose(mImColor, mK, mInvK, mpReferenceKF, mCameraPixelNoise2,
                                 cnn_slam::TRACKING_SOLVER_TIMECOST_RATIO / mFPS, Tcw, initTcw);
    cout << "Adjustment: ";
    cnn_slam::PrintRotTrans(Tcw * mCurrentFrame.mTcw.inv());
    mCurrentFrame.SetPose(Tcw);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    return nmatchesMap>=10;
}

bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    UpdateLocalMap();

    SearchLocalPoints();

    // Optimize Pose
    Optimizer::PoseOptimization(&mCurrentFrame);
    cv::Mat Tcw;
    cv::Mat initTcw = mLastFrame.mTcw;
//    cv::Mat initTcw = mCurrentFrame.mTcw;
    cnn_slam::EstimateCameraPose(mImColor, mK, mInvK, mpReferenceKF, mCameraPixelNoise2,
                                 cnn_slam::TRACKING_SOLVER_TIMECOST_RATIO / mFPS, Tcw, initTcw);
    cout << "Adjustment: ";
    cnn_slam::PrintRotTrans(Tcw * mCurrentFrame.mTcw.inv());
    mCurrentFrame.SetPose(Tcw);

    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    if(mnMatchesInliers<30)
        return false;
    else
        return true;
}


bool Tracking::NeedNewKeyFrame()
{
    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    //Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB,mImColor,mpDepthEstimator,mCurrentFrame.mpReferenceKF,mCurrentFrame.focalLength);

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mSensor!=System::MONOCULAR)
    {
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}


void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

void Tracking::Reset()
{

    cout << "System Reseting" << endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}



} //namespace ORB_SLAM
