#include "CubeSLAM.h"
#include "ObjectDetector.h"

using namespace std;
using namespace cv;

namespace ORB_SLAM2 {

static inline float
DistanceError(const CuboidProposal& proposal, const Rect& bbox, const vector<vector<float>>& distMap)
{
    float distErr = 0;
    float weight_sum = 0;
    distErr += 1.5f / ChamferDist(make_pair(proposal.corners[0], proposal.corners[1]), bbox, distMap);
    distErr += 1.5f / ChamferDist(make_pair(proposal.corners[0], proposal.corners[2]), bbox, distMap);
    distErr += 1.5f / ChamferDist(make_pair(proposal.corners[1], proposal.corners[3]), bbox, distMap);
    distErr += 1.5f / ChamferDist(make_pair(proposal.corners[2], proposal.corners[3]), bbox, distMap);
    weight_sum += 6.f;
    if (proposal.isCornerVisible[6]) {
        distErr += 2.f / ChamferDist(make_pair(proposal.corners[1], proposal.corners[6]), bbox, distMap);
        weight_sum += 2.f;
    }
    if (proposal.isCornerVisible[4]) {
        distErr += 2.f / ChamferDist(make_pair(proposal.corners[3], proposal.corners[4]), bbox, distMap);
        weight_sum += 2.f;
    }
    if (proposal.isCornerVisible[4] && proposal.isCornerVisible[6]) {
        distErr += 1.5f / ChamferDist(make_pair(proposal.corners[4], proposal.corners[6]), bbox, distMap);
        weight_sum += 1.5f;
    }
    if (proposal.isCornerVisible[5]) {
        distErr += 2.f / ChamferDist(make_pair(proposal.corners[2], proposal.corners[5]), bbox, distMap);
        distErr += 1.5f / ChamferDist(make_pair(proposal.corners[4], proposal.corners[5]), bbox, distMap);
        weight_sum += 3.5f;
    }
    distErr = weight_sum / distErr;
    return distErr;
}

static inline float AlignmentError(const CuboidProposal& proposal, const vector<LineSegment*>& lineSegs,
                                   const cv::Point2f& vp1, const cv::Point2f& vp2, const cv::Point2f& vp3)
{
    float alignErr = 0;
    float err_sum[3] = {};
    int err_cnt[3] = {};
    for (const auto& seg : lineSegs) {
        float err1 = AlignmentError(vp1, *seg);
        float err2 = AlignmentError(vp2, *seg);
        float err3 = AlignmentError(vp3, *seg);

        float minErr = err3;
        int minErrIdx = 2;

        if (err1 < 15.f / 180 * M_PI && err1 < minErr) {
            minErr = err1;
            minErrIdx = 0;
        }

        if (err2 < 15.f / 180 * M_PI && err2 < minErr) {
            minErr = err2;
            minErrIdx = 1;
        }

        if (minErr < 20.f / 180 * M_PI) {
            err_sum[minErrIdx] += 1.f / minErr;
            ++err_cnt[minErrIdx];
        }
    }
    alignErr = 0;
    for (int i = 0; i < 3; ++i) {
        if (err_cnt[i])
            alignErr += err_cnt[i] / err_sum[i];
        else
            alignErr += M_PI_2;
    }
    alignErr /= 3;

    return alignErr;
}

static inline float ShapeError(const CuboidProposal& proposal, float shapeErrThresh)
{
    float edgeLenSum1 = Distance(proposal.corners[0], proposal.corners[1])
                        + Distance(proposal.corners[2], proposal.corners[3])
                        + Distance(proposal.corners[4], proposal.corners[5])
                        + Distance(proposal.corners[6], proposal.corners[7]);
    float edgeLenSum2 = Distance(proposal.corners[0], proposal.corners[2])
                        + Distance(proposal.corners[1], proposal.corners[3])
                        + Distance(proposal.corners[4], proposal.corners[6])
                        + Distance(proposal.corners[5], proposal.corners[7]);
    float shapeErr = edgeLenSum1 > edgeLenSum2 ?
                     edgeLenSum1 / edgeLenSum2 :
                     edgeLenSum2 / edgeLenSum1;
    shapeErr = max(shapeErr - shapeErrThresh, 0.f) * 100;
    return shapeErr;
}

CuboidProposal FindBestProposal(Rect bbox, vector<LineSegment*> lineSegs, Mat K,
                                float shapeErrThresh, float shapeErrWeight, float alignErrWeight,
                                float initRoll, float initPitch, float initYaw,
                                float frameId, Mat image, bool display, bool save)
{
    const Point2f topLeft(bbox.x, bbox.y);
    const Point2f topRight(bbox.x + bbox.width, bbox.y);
    const Point2f botLeft(bbox.x, bbox.y + bbox.height);
    const Point2f botRight(bbox.x + bbox.width, bbox.y + bbox.height);

    auto distMap = PrecomputeChamferDistMap(bbox, lineSegs);

    initRoll = 0;
    initPitch = 0;

    CuboidProposal proposal;
    vector<float> distErrs, alignErrs, shapeErrs;
    vector<CuboidProposal> candidates;
    const auto topXStep = max(2, bbox.width / 20);
    const auto topXStart = bbox.x + (topXStep >> 1);
    const auto topXEnd = bbox.x + bbox.width - (topXStep >> 1);
    const auto rollStep = static_cast<const float>(M_PI / 36);
    const auto rollStart = initRoll - static_cast<const float>(M_PI_2);
    const auto rollEnd = initRoll + static_cast<const float>(M_PI_2);
    const auto pitchStep = static_cast<const float>(M_PI / 36);
    const auto pitchStart = initPitch - static_cast<const float>(M_PI_2);
    const auto pitchEnd = initPitch + static_cast<const float>(M_PI_2);
    const auto yawStep = static_cast<const float>(M_PI / 36);
    const auto yawStart = initYaw - static_cast<const float>(M_PI_2 / 2);
    const auto yawEnd = initYaw + static_cast<const float>(M_PI_2 / 2);
    int imgIdx = 0;
    // Sample the landmark pitch in 30 degrees around the camera pitch.
    for (float l_pitch = pitchStart; l_pitch <= pitchEnd; l_pitch += pitchStep) {
        // Sample the landmark roll in 30 degrees around the camera roll.
        for (float l_roll = rollStart; l_roll <= rollEnd; l_roll += rollStep) {
            // Sample the landmark yaw in 180 degrees around the camera yaw.
            for (float l_yaw = yawStart; l_yaw <= yawEnd; l_yaw += yawStep) {
                // Recover rotation of the landmark.
                proposal.Rlc = EulerAnglesToRotationMatrix(Vec3f(l_roll, l_pitch, l_yaw));

                Mat invRlc = proposal.Rlc.t();

                // Compute the vanishing points from the pose.
                Mat vp1Homo = K * invRlc.col(0);
                Mat vp2Homo = K * invRlc.col(1);
                Mat vp3Homo = K * invRlc.col(2);
                Point2f vp1 = Point2FromHomo(vp1Homo);
                Point2f vp2 = Point2FromHomo(vp2Homo);
                Point2f vp3 = Point2FromHomo(vp3Homo);

                if (vp3.x < bbox.x || vp3.x > bbox.x + bbox.width ||
                    (vp3.y >= bbox.y && vp3.y <= bbox.y + bbox.height) ||
                    (vp1.y >= bbox.y && vp1.y <= bbox.y + bbox.height) ||
                    (vp2.y >= bbox.y && vp2.y <= bbox.y + bbox.height))
                    continue;

                bool flip = (vp1.x > bbox.x + bbox.width && vp2.x < bbox.x) ||
                            (vp2.x > bbox.x && vp2.x < bbox.x + bbox.width);
                if (flip) {
                    swap(vp1, vp2);
                }

                // Sample corner on the top boundary.
                for (int topX = topXStart; topX <= topXEnd; topX += topXStep) {
                    proposal.corners[0] = Point2f(topX, bbox.y);

                    // Compute the other corners with respect to the pose, vanishing points and the bounding box.
                    if (vp1.x < bbox.x && vp2.x > bbox.x + bbox.width) {
                        // 3 faces
                        proposal.corners[1] = LineIntersectionX(vp1, proposal.corners[0], topRight.x);
                        if (!inside(proposal.corners[1], bbox))
                            continue;
                        proposal.corners[2] = LineIntersectionX(vp2, proposal.corners[0], topLeft.x);
                        if (!inside(proposal.corners[2], bbox))
                            continue;
                        proposal.corners[3] = LineIntersection(vp1, proposal.corners[2], vp2, proposal.corners[1]);
                        if (!inside(proposal.corners[3], bbox))
                            continue;
                        proposal.corners[4] = LineIntersectionY(vp3, proposal.corners[3], botLeft.y);
                        if (!inside(proposal.corners[4], bbox))
                            continue;
                        proposal.corners[5] = LineIntersection(vp1, proposal.corners[4], vp3, proposal.corners[2]);
                        if (!inside(proposal.corners[5], bbox))
                            continue;
                        proposal.corners[6] = LineIntersection(vp2, proposal.corners[4], vp3, proposal.corners[1]);
                        if (!inside(proposal.corners[6], bbox))
                            continue;
                        proposal.corners[7] = LineIntersection(vp1, proposal.corners[6], vp2, proposal.corners[5]);
                        if (!inside(proposal.corners[7], bbox))
                            continue;

                        proposal.isCornerVisible[4] = true;
                        proposal.isCornerVisible[5] = true;
                        proposal.isCornerVisible[6] = true;
                        proposal.isCornerVisible[7] = false;
                    }
                    else if (vp1.x > bbox.x && vp1.x < bbox.x + bbox.width) {
                        if (vp2.x < bbox.x) {
                            // 2 faces
                            proposal.corners[1] = LineIntersectionX(vp1, proposal.corners[0], topLeft.x);
                            if (!inside(proposal.corners[1], bbox))
                                continue;
                            proposal.corners[3] = LineIntersectionX(vp2, proposal.corners[1], topRight.x);
                            if (!inside(proposal.corners[3], bbox))
                                continue;
                        }
                        else if (vp2.x > bbox.x + bbox.width) {
                            // 2 faces
                            proposal.corners[1] = LineIntersectionX(vp1, proposal.corners[0], topRight.x);
                            if (!inside(proposal.corners[1], bbox))
                                continue;
                            proposal.corners[3] = LineIntersectionX(vp2, proposal.corners[1], topLeft.x);
                            if (!inside(proposal.corners[3], bbox))
                                continue;
                        }
                        else
                            continue;
                        proposal.corners[2] = LineIntersection(vp1, proposal.corners[3], vp2, proposal.corners[0]);
                        if (!inside(proposal.corners[2], bbox))
                            continue;
                        proposal.corners[4] = LineIntersectionY(vp3, proposal.corners[3], botLeft.y);
                        if (!inside(proposal.corners[4], bbox))
                            continue;
                        proposal.corners[5] = LineIntersection(vp1, proposal.corners[4], vp3, proposal.corners[2]);
                        if (!inside(proposal.corners[5], bbox))
                            continue;
                        proposal.corners[6] = LineIntersection(vp2, proposal.corners[4], vp3, proposal.corners[1]);
                        if (!inside(proposal.corners[6], bbox))
                            continue;
                        proposal.corners[7] = LineIntersection(vp1, proposal.corners[6], vp2, proposal.corners[5]);
                        if (!inside(proposal.corners[7], bbox))
                            continue;

                        proposal.isCornerVisible[4] = true;
                        proposal.isCornerVisible[5] = false;
                        proposal.isCornerVisible[6] = true;
                        proposal.isCornerVisible[7] = false;
                    }
                    else
                        continue;

                    // Score the proposal.
                    distErrs.emplace_back(DistanceError(proposal, bbox, distMap));
                    alignErrs.emplace_back(AlignmentError(proposal, lineSegs, vp1, vp2, vp3));
                    shapeErrs.emplace_back(ShapeError(proposal, shapeErrThresh));

                    if (flip) {
                        swap(proposal.corners[1], proposal.corners[2]);
                        swap(proposal.corners[5], proposal.corners[6]);
                    }

                    candidates.emplace_back(proposal);

                    if (imgIdx % 1 == 0 && !image.empty()) {
                        // Draw proposal.
                        Mat canvas = image.clone();
                        rectangle(canvas, topLeft, botRight, Scalar(255, 0, 0), 1, CV_AA);
                        DrawProposal(canvas, proposal);

                        if (flip)
                            swap(vp1, vp2);

                        line(canvas, vp1, proposal.corners[0], Scalar(0, 255, 0));
                        line(canvas, vp1, proposal.corners[2], Scalar(0, 255, 0));
                        line(canvas, vp1, proposal.corners[6], Scalar(0, 0, 0));
                        line(canvas, vp2, proposal.corners[0], Scalar(0, 255, 0));
                        line(canvas, vp2, proposal.corners[1], Scalar(0, 255, 0));
                        line(canvas, vp2, proposal.corners[5], Scalar(0, 0, 0));
                        line(canvas, vp3, proposal.corners[5], Scalar(0, 255, 0));
                        line(canvas, vp3, proposal.corners[6], Scalar(0, 255, 0));

                        for (auto& seg : lineSegs) {
                            line(canvas, seg->first, seg->second, Scalar(0, 0, 255), 1, CV_AA);
                        }

                        if (display) {
                            cout << "Roll=" << l_roll * 180 / M_PI
                                 << " Pitch=" << l_pitch * 180 / M_PI
                                 << " Yaw=" << l_yaw * 180 / M_PI << endl;
                            cout << vp1Homo.t() << ' ' << vp1 << endl
                                 << vp2Homo.t() << ' ' << vp2 << endl
                                 << vp3Homo.t() << ' ' << vp3 << endl;
                            cout << proposal << endl;
                            imshow("Proposal", canvas);
                            waitKey(0);
                        }
                        if (save)
                            imwrite("Outputs/" + to_string(frameId) + "_" + to_string(imgIdx) + ".jpg", canvas);

                        if (flip)
                            swap(vp1, vp2);
                    }
                    ++imgIdx;
                }
            }
        }
    }

    const int numErr = static_cast<const int>(distErrs.size());
    if (!numErr)
        return {};  // Return an empty cuboid proposal.

    const float minDistErr = *min_element(distErrs.begin(), distErrs.end());
    const float minAlignErr = *min_element(alignErrs.begin(), alignErrs.end());
    const float maxDistErr = *max_element(distErrs.begin(), distErrs.end());
    const float maxAlignErr = *max_element(alignErrs.begin(), alignErrs.end());
    int bestProposalIdx = -1;

    float bestErr = -1;
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