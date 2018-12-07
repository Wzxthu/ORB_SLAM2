#include "CubeSLAM.h"

using namespace std;
using namespace cv;

namespace ORB_SLAM2 {

CuboidProposal FindBestProposal(Rect bbox, float c_yaw, float c_roll, float c_pitch,
                                float shapeErrThresh, float shapeErrWeight, float alignErrWeight,
                                vector<LineSegment*> lineSegs,
                                Mat K,
                                float& bestErr,
                                float frameId,
                                Mat image)
{
    const Point2f topLeft(bbox.x, bbox.y);
    const Point2f topRight(bbox.x + bbox.width, bbox.y);
    const Point2f botLeft(bbox.x, bbox.y + bbox.height);
    const Point2f botRight(bbox.x + bbox.width, bbox.y + bbox.height);

    CuboidProposal proposal;
    vector<float> distErrs, alignErrs, shapeErrs;
    vector<CuboidProposal> candidates;
    bool isCornerVisible[8] = {true, true, true, true};
    const auto topXStep = max(2, bbox.width / 10);
    const auto yaw_start = c_yaw - static_cast<const float>(M_PI_2);
    const auto yaw_end = c_yaw + static_cast<const float>(M_PI_2);
    const auto yaw_step = static_cast<const float>(6.f / 180 * M_PI);
    const auto roll_start = c_roll - static_cast<const float>(M_PI_2 / 6);
    const auto roll_end = c_roll + static_cast<const float>(M_PI_2 / 6);
    const auto roll_step = static_cast<const float>(6.f / 180 * M_PI);
    const auto pitch_start = c_pitch - static_cast<const float>(M_PI_2 / 6);
    const auto pitch_end = c_pitch + static_cast<const float>(M_PI_2 / 6);
    const auto pitch_step = static_cast<const float>(6.f / 180 * M_PI);
    int imgIdx = 0;
    // Sample corner on the top boundary.
    for (int topX = bbox.x + (topXStep >> 1); topX < bbox.x + bbox.width - (topXStep >> 1); topX += topXStep) {
        proposal[0] = Point2f(topX, bbox.y);
        // Sample the landmark yaw in 180 degrees around the camera yaw.
        for (float l_yaw = yaw_start; l_yaw < yaw_end; l_yaw += yaw_step) {
            // Sample the landmark roll in 30 degrees around the camera roll.
            for (float l_roll = roll_start; l_roll < roll_end; l_roll += roll_step) {
                // Sample the landmark pitch in 30 degrees around the camera pitch.
                for (float l_pitch = pitch_start; l_pitch < pitch_end; l_pitch += pitch_step) {
                    // Recover rotation of the landmark.
                    Mat Rlw = RotationFromRollPitchYaw(l_roll, l_pitch, c_yaw);
                    Mat invRlw = Rlw.t();

                    // Compute the vanishing points from the pose.
                    Vec3f R1(cos(l_yaw), sin(l_yaw), 0);
                    Vec3f R2(-sin(l_yaw), cos(l_yaw), 0);
                    Vec3f R3(0, 0, 1);
                    Mat vp1 = K * invRlw * Mat(R1);
                    Mat vp2 = K * invRlw * Mat(R2);
                    Mat vp3 = K * invRlw * Mat(R3);
                    Point2f vp1_homo(vp1.at<float>(0, 0) / vp1.at<float>(2, 0),
                                     vp1.at<float>(1, 0) / vp1.at<float>(2, 0));
                    Point2f vp2_homo(vp2.at<float>(0, 0) / vp2.at<float>(2, 0),
                                     vp2.at<float>(1, 0) / vp2.at<float>(2, 0));
                    Point2f vp3_homo(vp3.at<float>(0, 0) / vp3.at<float>(2, 0),
                                     vp3.at<float>(1, 0) / vp3.at<float>(2, 0));

                    // Compute the other corners with respect to the pose, vanishing points and the bounding box.
                    if (vp3_homo.x < bbox.x || vp3_homo.x > bbox.x + bbox.width ||
                        vp3_homo.y < bbox.y + bbox.height ||
                        vp1_homo.y > bbox.y || vp2_homo.y > bbox.y) {
                        continue;
                    }
                    else if ((vp1_homo.x < bbox.x && vp2_homo.x > bbox.x + bbox.width) ||
                             (vp1_homo.x > bbox.x + bbox.width && vp2_homo.x < bbox.x)) {
                        if (vp1_homo.x > bbox.x + bbox.width && vp2_homo.x < bbox.x) {
                            swap(vp1_homo, vp2_homo);
                        }
                        // 3 faces
                        proposal[1] = LineIntersection(vp1_homo, proposal[0], topRight, botRight);
                        if (!proposal[1].inside(bbox)) {
                            continue;
                        }
                        proposal[2] = LineIntersection(vp2_homo, proposal[0], topLeft, botLeft);
                        if (!proposal[2].inside(bbox)) {
                            continue;
                        }
                        proposal[3] = LineIntersection(vp1_homo, proposal[2], vp2_homo, proposal[1]);
                        if (!proposal[3].inside(bbox)) {
                            continue;
                        }
                        proposal[4] = LineIntersection(vp3_homo, proposal[3], botLeft, botRight);
                        if (!proposal[4].inside(bbox)) {
                            continue;
                        }
                        proposal[5] = LineIntersection(vp1_homo, proposal[4], vp3_homo, proposal[2]);
                        if (!proposal[5].inside(bbox)) {
                            continue;
                        }
                        proposal[6] = LineIntersection(vp2_homo, proposal[4], vp3_homo, proposal[1]);
                        if (!proposal[6].inside(bbox)) {
                            continue;
                        }
                        proposal[7] = LineIntersection(vp1_homo, proposal[6], vp2_homo, proposal[5]);
                        if (!proposal[7].inside(bbox)) {
                            continue;
                        }
                        isCornerVisible[4] = true;
                        isCornerVisible[5] = true;
                        isCornerVisible[6] = true;
                        isCornerVisible[7] = false;
                    }
                    else if ((vp1_homo.x > bbox.x && vp1_homo.x < bbox.x + bbox.width) ||
                             (vp2_homo.x > bbox.x && vp2_homo.x < bbox.x + bbox.width)) {
                        if (vp2_homo.x > bbox.x && vp2_homo.x < bbox.x + bbox.width) {
                            swap(vp1_homo, vp2_homo);
                        }
                        if (vp2_homo.x < bbox.x) {
                            // 2 faces
                            proposal[1] = LineIntersection(vp1_homo, proposal[0], topLeft, botLeft);
                            if (!proposal[1].inside(bbox)) {
                                continue;
                            }
                            proposal[3] = LineIntersection(vp2_homo, proposal[1], topRight, botRight);
                        }
                        else if (vp2_homo.x > bbox.x + bbox.width) {
                            // 2 faces
                            proposal[1] = LineIntersection(vp1_homo, proposal[0], topRight, botRight);
                            if (!proposal[1].inside(bbox)) {
                                continue;
                            }
                            proposal[3] = LineIntersection(vp2_homo, proposal[1], topLeft, botLeft);
                            if (!proposal[3].inside(bbox)) {
                                continue;
                            }
                        }
                        else {
                            continue;
                        }
                        proposal[2] = LineIntersection(vp1_homo, proposal[3], vp2_homo, proposal[0]);
                        if (!proposal[2].inside(bbox)) {
                            continue;
                        }
                        proposal[4] = LineIntersection(vp3_homo, proposal[3], botLeft, botRight);
                        if (!proposal[4].inside(bbox)) {
                            continue;
                        }
                        proposal[5] = LineIntersection(vp1_homo, proposal[4], vp3_homo, proposal[2]);
                        if (!proposal[5].inside(bbox)) {
                            continue;
                        }
                        proposal[6] = LineIntersection(vp2_homo, proposal[4], vp3_homo, proposal[1]);
                        if (!proposal[6].inside(bbox)) {
                            continue;
                        }
                        proposal[7] = LineIntersection(vp1_homo, proposal[6], vp2_homo, proposal[5]);
                        if (!proposal[7].inside(bbox)) {
                            continue;
                        }
                        isCornerVisible[4] = true;
                        isCornerVisible[5] = false;
                        isCornerVisible[6] = true;
                        isCornerVisible[7] = false;
                    }
                    else {
                        continue;
                    }

                    // Score the proposal.
                    float distErr = 0, alignErr = 0, shapeErr = 0;

                    // Distance error
                    float weight_sum = 0;
                    distErr += 1.5 / ChamferDist(make_pair(proposal[0], proposal[1]), lineSegs);
                    distErr += 1.5 / ChamferDist(make_pair(proposal[0], proposal[2]), lineSegs);
                    distErr += 1.5 / ChamferDist(make_pair(proposal[1], proposal[3]), lineSegs);
                    distErr += 1.5 / ChamferDist(make_pair(proposal[2], proposal[3]), lineSegs);
                    distErr += 2 / ChamferDist(make_pair(proposal[1], proposal[6]), lineSegs);
                    distErr += 2 / ChamferDist(make_pair(proposal[3], proposal[4]), lineSegs);
                    distErr += 1.5 / ChamferDist(make_pair(proposal[4], proposal[6]), lineSegs);
                    weight_sum += 11.5;
                    if (isCornerVisible[5]) {
                        distErr += 2 / ChamferDist(make_pair(proposal[2], proposal[5]), lineSegs);
                        distErr += 1.5 / ChamferDist(make_pair(proposal[4], proposal[5]), lineSegs);
                        weight_sum += 3.5;
                    }
                    distErr = weight_sum / distErr;

                    // Angle alignment error.
                    float err_sum[3] = {};
                    int err_cnt[3] = {};
                    for (const auto& seg : lineSegs) {
                        float err1 = AlignmentError(Point2f(vp1.at<float>(0), vp1.at<float>(1)), *seg);
                        float err2 = AlignmentError(Point2f(vp2.at<float>(0), vp2.at<float>(1)), *seg);
                        float err3 = AlignmentError(Point2f(vp3.at<float>(0), vp3.at<float>(1)), *seg);

                        float minErr = err3;
                        int minErrIdx = 2;

                        if (err1 < 10.0 / 180 * M_PI && err1 < minErr) {
                            minErr = err1;
                            minErrIdx = 0;
                        }

                        if (err2 < 10.0 / 180 * M_PI && err2 < minErr) {
                            minErr = err2;
                            minErrIdx = 1;
                        }

                        if (minErr < 15.0 / 180 * M_PI) {
                            err_sum[minErrIdx] += minErr;
                            ++err_cnt[minErrIdx];
                        }
                    }
                    alignErr = 0;
                    for (int i = 0; i < 3; ++i) {
                        if (err_cnt[i])
                            alignErr += err_sum[i] / err_cnt[i];
                        else
                            alignErr += M_PI_2;
                    }
                    alignErr /= 3;

                    // Shape error.
                    float edgeLenSum1 = Distance(proposal[0], proposal[1])
                                        + Distance(proposal[2], proposal[3])
                                        + Distance(proposal[4], proposal[5])
                                        + Distance(proposal[6], proposal[7]);
                    float edgeLenSum2 = Distance(proposal[0], proposal[2])
                                        + Distance(proposal[1], proposal[3])
                                        + Distance(proposal[4], proposal[6])
                                        + Distance(proposal[5], proposal[7]);
                    if (edgeLenSum1 < 80 || edgeLenSum2 < 80) {
                        continue;
                    }
                    shapeErr = edgeLenSum1 > edgeLenSum2 ?
                               edgeLenSum1 / edgeLenSum2 :
                               edgeLenSum2 / edgeLenSum1;
                    shapeErr = max(shapeErr - shapeErrThresh, 0.f) * 100;

                    distErrs.push_back(distErr);
                    alignErrs.push_back(alignErr);
                    shapeErrs.push_back(shapeErr);
                    candidates.push_back(proposal);

                    if (imgIdx % 5 == 0) {
                        // draw bbox
                        rectangle(image, topLeft, botRight, Scalar(255, 0, 0), 1, CV_AA);
                        // draw cube
                        line(image, proposal[0], proposal[1], Scalar(0, 255, 0), 1, CV_AA);
                        line(image, proposal[1], proposal[3], Scalar(0, 255, 0), 1, CV_AA);
                        line(image, proposal[3], proposal[2], Scalar(0, 255, 0), 1, CV_AA);
                        line(image, proposal[2], proposal[0], Scalar(0, 255, 0), 1, CV_AA);
                        line(image, proposal[0], proposal[7], Scalar(0, 255, 0), 1, CV_AA);
                        line(image, proposal[1], proposal[6], Scalar(0, 255, 0), 1, CV_AA);
                        line(image, proposal[2], proposal[5], Scalar(0, 255, 0), 1, CV_AA);
                        line(image, proposal[3], proposal[4], Scalar(0, 255, 0), 1, CV_AA);
                        line(image, proposal[7], proposal[6], Scalar(0, 255, 0), 1, CV_AA);
                        line(image, proposal[6], proposal[4], Scalar(0, 255, 0), 1, CV_AA);
                        line(image, proposal[4], proposal[5], Scalar(0, 255, 0), 1, CV_AA);
                        line(image, proposal[5], proposal[7], Scalar(0, 255, 0), 1, CV_AA);

                        for (auto& seg : lineSegs) {
                            line(image, seg->first, seg->second, Scalar(0, 0, 255), 1, CV_AA);
                        }

                        cout << "Errors of Proposal " << imgIdx << " in Frame " << frameId
                             << ": " << distErr << ' ' << alignErr << ' ' << shapeErr << endl;
                        imwrite("Outputs/" + to_string(frameId) + "_" + to_string(imgIdx)
                                + ".jpg", image);
                    }
                    ++imgIdx;
                }
            }
        }
    }

    bestErr = -1;

    const int numErr = static_cast<const int>(distErrs.size());
    if (!numErr)
        return CuboidProposal();

    const float minDistErr = *min_element(distErrs.begin(), distErrs.end());
    const float minAlignErr = *min_element(alignErrs.begin(), alignErrs.end());
    const float maxDistErr = *max_element(distErrs.begin(), distErrs.end());
    const float maxAlignErr = *max_element(alignErrs.begin(), alignErrs.end());
    int bestProposalIdx = -1;

    for (int i = 0; i < numErr; ++i) {
        // Sum the errors by weight.
        float normDistErr = (distErrs[i] - minDistErr) / (maxDistErr - minDistErr);
        float normAlignErr = (alignErrs[i] - minAlignErr) / (maxAlignErr - minAlignErr);
        float totalErr = (normDistErr + alignErrWeight * normAlignErr) / (1 + alignErrWeight)
                         + shapeErrWeight * shapeErrs[i];
        if (totalErr < bestErr || bestErr == -1) {
            bestErr = totalErr;
            bestProposalIdx = i;
        }
    }

    return candidates[bestProposalIdx];
}

}