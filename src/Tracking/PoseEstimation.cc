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

#include <Tracking/PoseEstimation.h>
#include <KeyFrame.h>

#include <ceres/autodiff_cost_function.h>
#include <ceres/numeric_diff_cost_function.h>
#include <ceres/sized_cost_function.h>
#include <ceres/solver.h>
#include <ceres/loss_function.h>
#include <ceres/problem.h>
#include <ceres/rotation.h>

#include <thread>
#include <util/settings.h>
#include <Frame.h>

using namespace cv;
using namespace std;

namespace cnn_slam {

    // Use ceres for Gaussian-Newton optimization.
    using namespace ceres;

    struct CostFunctor {
        Mat imColor;
        Mat imGradX;
        Mat imGradY;
        ORB_SLAM2::KeyFrame *pRefKF;
        Mat Kt;      // Transposed calibration matrix.
        Mat invKt;   // Transposed inverse calibration matrix.
        float cameraPixelNoise2;

        CostFunctor(Mat imColor,
                    Mat imGradX,
                    Mat imGradY,
                    ORB_SLAM2::KeyFrame *pReferenceKF,
                    Mat K,
                    Mat invK,
                    float cameraPixelNoise2)
                : imColor(imColor),
                  imGradX(imGradX),
                  imGradY(imGradY),
                  pRefKF(pReferenceKF),
                  Kt(K.t()),
                  invKt(invK.t()),
                  cameraPixelNoise2(cameraPixelNoise2) {}

        template<class T>
        bool operator()(const T *const r, const T *const t, T *residual) const {
            int type;
            if (is_same<T, double>::value) {
                type = CV_64F;
            } else if (is_same<T, float>::value) {
                type = CV_32F;
            } else {
                // Unsupported.
                return false;
            }

            // Recover rotation and translation from the state vector.
            Mat Rt(3, 3, type);    // Transposed rotation matrix.
            Rodrigues(Mat(1, 3, type, (void *) r), Rt);
            Mat(Rt.t()).convertTo(Rt, CV_32F);
            Mat tt(1, 3, type, (void *) t);  // Transposed translation matrix.
            tt.convertTo(tt, CV_32F);
            tt = repeat(tt, pRefKF->mHighGradPtDepth.rows, 1);

            // Calculate projected 2D location in the current frame
            // of the high gradient points in the reference keyframe.
            Mat proj3d = (repeat(pRefKF->mHighGradPtDepth, 1, 3)
                                  .mul(pRefKF->mHighGradPtHomo2dCoord)
                          * invKt * Rt + tt) * Kt;
            Mat proj2d = proj3d.colRange(0, 2) / repeat(proj3d.col(2), 1, 2);
            Mat proj2d_round;
            proj2d.convertTo(proj2d_round, CV_32S);
            Mat proj2d_offset = proj2d - proj2d_round;

            // Calculate derivative of projected 2D coordinate regarding depth.
            Mat coord_deriv = pRefKF->mHighGradPtHomo2dCoord * invKt * Rt * Kt / repeat(proj3d.col(2), 1, 3);

            // Find out points whose 2D projection is in frame.
            Mat valid = proj2d_round.col(0) >= 0;
            bitwise_and(valid, proj2d_round.col(0) < imColor.cols, valid, valid);
            bitwise_and(valid, proj2d_round.col(1) >= 0, valid, valid);
            bitwise_and(valid, proj2d_round.col(1) < imColor.rows, valid, valid);

            Mat regRes(valid.rows, 3, CV_32F);
            // Extract pixel values from the current frame, then calculate photometric residual and variance, and
            // finally regularize the photometric residual.
#pragma omp parallel for
            for (int i = 0; i < valid.rows; ++i) {
                if (valid.at<uchar>(i)) {
                    // Extract pixel at rounded position in the image.
                    auto pixel = imColor.at<Vec3b>(proj2d_round.at<int>(i, 1), proj2d_round.at<int>(i, 0));
                    // Approximate accurate pixel value by gradient and rounding offset.
                    auto gradX = imGradX.at<Vec3b>(proj2d_round.at<int>(i, 1), proj2d_round.at<int>(i, 0));
                    auto gradY = imGradY.at<Vec3b>(proj2d_round.at<int>(i, 1), proj2d_round.at<int>(i, 0));
                    float pc0 = pixel.val[0] +
                                gradX[0] * proj2d_offset.at<float>(i, 0) +
                                gradY[0] * proj2d_offset.at<float>(i, 1);
                    float pc1 = pixel.val[1] +
                                gradX[1] * proj2d_offset.at<float>(i, 0) +
                                gradY[1] * proj2d_offset.at<float>(i, 1);
                    float pc2 = pixel.val[2] +
                                gradX[2] * proj2d_offset.at<float>(i, 0) +
                                gradY[2] * proj2d_offset.at<float>(i, 1);

                    // Calculate photometric residual.
                    float res_c0 = pc0 - pRefKF->mHighGradPtPixels.at<uchar>(i, 0);
                    float res_c1 = pc1 - pRefKF->mHighGradPtPixels.at<uchar>(i, 1);
                    float res_c2 = pc2 - pRefKF->mHighGradPtPixels.at<uchar>(i, 2);

                    // Calculate derivative of photometric residual with respect to depth.
                    float der_c0 = gradX[0] * coord_deriv.at<float>(i, 0);
                    float der_c1 = gradX[1] * coord_deriv.at<float>(i, 1);
                    float der_c2 = gradX[2] * coord_deriv.at<float>(i, 2);

                    // Calculate variance of photometric residual.
                    float var_c0 = 2 * cameraPixelNoise2 +
                                   der_c0 * der_c0 * pRefKF->mHighGradPtSqrtUncertainty.at<float>(i, 0);
                    float var_c1 = 2 * cameraPixelNoise2 +
                                   der_c1 * der_c1 * pRefKF->mHighGradPtSqrtUncertainty.at<float>(i, 1);
                    float var_c2 = 2 * cameraPixelNoise2 +
                                   der_c2 * der_c2 * pRefKF->mHighGradPtSqrtUncertainty.at<float>(i, 2);

                    // Calculate regularized photometric residual.
                    regRes.at<float>(i, 0) = fabs(res_c0 / std::sqrt(var_c0));
                    regRes.at<float>(i, 1) = fabs(res_c1 / std::sqrt(var_c1));
                    regRes.at<float>(i, 2) = fabs(res_c2 / std::sqrt(var_c2));
                }
            }

            // Set the invalid points to mean residual.
            float meanRes = static_cast<float>(mean(regRes, valid)[0]);
#pragma omp parallel for
            for (int i = 0; i < valid.rows; ++i)
                if (!valid.at<uchar>(i))
                    regRes.at<float>(i) = meanRes;

            // Fill the residual.
            regRes.convertTo(regRes, type);
            assert(regRes.isContinuous());
            memcpy(residual, regRes.data, sizeof(T) * regRes.rows);
            return true;
        }
    };

    float EstimateCameraPose(const Mat &imColor,
                             const Mat &K,
                             const Mat &invK,
                             ORB_SLAM2::KeyFrame *pRefKF,
                             float cameraPixelNoise2,
                             double max_seconds,
                             Mat &Tcw,
                             const cv::Mat initialTcw,
                             float *rotAngle,
                             float *transDist,
                             float *validRatio) {
        // Initialize pose as no-transform.
        // The pose is expressed as scale (1-dim, not used), rotation (3-dim rodrigues) and translation (3-dim).
        double relRotationRodrigues[] = {0, 0, 0};
        double relTranslation[] = {0, 0, 0};

        if (!initialTcw.empty()) {
            // Initialize the relative pose according to the given initial pose.
            Mat Trel = pRefKF->GetPose() * initialTcw.inv();
            Mat rodrigues;
            Rodrigues(Trel.colRange(0, 3).rowRange(0, 3), rodrigues);
            relRotationRodrigues[0] = rodrigues.at<float>(0);
            relRotationRodrigues[1] = rodrigues.at<float>(1);
            relRotationRodrigues[2] = rodrigues.at<float>(2);
            relTranslation[0] = Trel.at<float>(0, 3);
            relTranslation[1] = Trel.at<float>(1, 3);
            relTranslation[2] = Trel.at<float>(2, 3);
        }

        Mat imGradX, imGradY;
        cv::Sobel(imColor, imGradX, CV_64FC3, 1, 0);
        cv::Sobel(imColor, imGradY, CV_64FC3, 0, 1);

        Problem problem;
        CostFunction *cost_function = new NumericDiffCostFunction<CostFunctor, RIDDERS, TRACKING_NUM_PT, 3, 3>(
                new CostFunctor(imColor, imGradX, imGradY, pRefKF, K, invK, cameraPixelNoise2));
        auto *loss_function = new LossFunctionWrapper(new HuberLoss(TRACKING_HUBER_DELTA), TAKE_OWNERSHIP);
        problem.AddResidualBlock(cost_function, loss_function, relRotationRodrigues, relTranslation);

        while (!pRefKF->mbDepthReady)
            usleep(1000);

        Solver::Options options;
        options.num_threads = thread::hardware_concurrency();   // Use all cores.
        options.max_solver_time_in_seconds = max_seconds; // Enforce real-time.
        Solver::Summary summary;
//        cout << "Solving..." << endl << flush;
        ceres::Solve(options, &problem, &summary);
        cout << "Solver finished with final cost " << summary.final_cost << "!" << endl << flush;

        Mat Rrel;
        Rodrigues(Mat(1, 3, CV_64F, relRotationRodrigues), Rrel);
        Rrel.convertTo(Rrel, CV_32F);
        Mat Trel = Mat::zeros(4, 4, CV_32F);
        Rrel.copyTo(Trel.rowRange(0, 3).colRange(0, 3));
        Trel.at<float>(0, 3) = static_cast<float>(relTranslation[0]);
        Trel.at<float>(1, 3) = static_cast<float>(relTranslation[1]);
        Trel.at<float>(2, 3) = static_cast<float>(relTranslation[2]);
        Trel.at<float>(3, 3) = 1;

        Tcw = Trel * pRefKF->GetPose();
        if (rotAngle) *rotAngle = RotationAngle(Rrel);
        if (transDist) *transDist = TranslationDist(Trel.col(3).rowRange(0, 3));
        if (validRatio) {
            // Calculate projected 2D location in the current frame
            // of the high gradient points in the reference keyframe.
            Mat proj3d =
                    (repeat(pRefKF->mHighGradPtDepth, 1, 3).mul(pRefKF->mHighGradPtHomo2dCoord)
                     * pRefKF->mInvK.t() * Rrel.t() +
                     repeat(Trel.col(3).rowRange(0, 3).t(), pRefKF->mHighGradPtDepth.rows, 1))
                    * K;
            Mat proj2d = proj3d.colRange(0, 2) / repeat(proj3d.col(2), 1, 2);
            assert(proj2d.cols == 2);
            proj2d.convertTo(proj2d, CV_32S);

            Mat valid = proj2d.col(0) >= 0;
            bitwise_and(valid, proj2d.col(0) < imColor.rows, valid, valid);
            bitwise_and(valid, proj2d.col(1) >= 0, valid, valid);
            bitwise_and(valid, proj2d.col(1) < imColor.cols, valid, valid);
            *validRatio = static_cast<float>(sum(valid)[0] / 255 / valid.rows);
        }

        return static_cast<float>(summary.final_cost) / TRACKING_NUM_PT;
    }
}