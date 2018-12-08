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

CuboidProposal FindBestProposal(const Rect& bbox, const vector<LineSegment*>& lineSegs, const Mat& K,
                                float shapeErrThresh, float shapeErrWeight, float alignErrWeight,
                                float refRoll, float refPitch,
                                int frameId, int objId, const Mat& image,
                                bool display, bool save)
{
    auto distMap = PrecomputeChamferDistMap(bbox, lineSegs);

    Mat canvas;
    vector<float> distErrs, alignErrs, shapeErrs;
    vector<CuboidProposal> candidates;
    const auto topXStep = max(2, bbox.width / 20);
    const auto topXStart = bbox.x + (topXStep >> 1);
    const auto topXEnd = bbox.x + bbox.width - (topXStep >> 1);
    const auto rollStep = static_cast<const float>(2 * M_PI / 180);
    const auto rollStart = refRoll - static_cast<const float>(45 * M_PI / 180);
    const auto rollEnd = refRoll + static_cast<const float>(45 * M_PI / 180);
    const auto pitchStep = static_cast<const float>(2 * M_PI / 180);
    const auto pitchStart = refPitch - static_cast<const float>(45 * M_PI / 180);
    const auto pitchEnd = refPitch + static_cast<const float>(45 * M_PI / 180);
    const auto yawStep = static_cast<const float>(6 * M_PI / 180);
    const auto yawStart = -M_PI - static_cast<const float>(90 * M_PI / 180);
    const auto yawEnd = -M_PI + static_cast<const float>(90 * M_PI / 180);
    // Sample the landmark pitch in 30 degrees around the camera pitch.
    for (float l_roll = rollStart; l_roll <= rollEnd; l_roll += rollStep) {
        // Sample the landmark yaw in 180 degrees around the camera yaw.
        for (float l_pitch = pitchStart; l_pitch <= pitchEnd; l_pitch += pitchStep) {
            // Sample the landmark roll in 30 degrees around the camera roll.
            for (float l_yaw = yawStart; l_yaw <= yawEnd; l_yaw += yawStep) {
                // Recover rotation of the landmark.
                Mat Rlc = EulerAnglesToRotationMatrix(Vec3f(l_roll, l_pitch, l_yaw));
                Mat invRlc = Rlc.t();

                // Compute the vanishing points from the pose.
                Mat vp1Homo = K * Rlc.col(0);
                Mat vp3Homo = K * Rlc.col(1);
                Mat vp2Homo = K * Rlc.col(2);
                Point2f vp1 = Point2FromHomo(vp1Homo);
                Point2f vp2 = Point2FromHomo(vp2Homo);
                Point2f vp3 = Point2FromHomo(vp3Homo);

                if (vp3.x <= bbox.x || vp3.x >= bbox.x + bbox.width || vp3.y <= bbox.y + bbox.height)
                    continue;
                if (vp1.y > bbox.y || vp2.y > bbox.y)
                    continue;

                bool flip = (vp1.x > bbox.x + bbox.width && vp2.x < bbox.x) ||
                            (vp2.x > bbox.x && vp2.x < bbox.x + bbox.width);
                if (flip) {
                    swap(vp1, vp2);
                }

                // Sample corner on the top boundary.
                for (int topX = topXStart; topX <= topXEnd; topX += topXStep) {
                    auto proposal = GenerateCuboidProposal(bbox, topX, vp1, vp2, vp3);
                    if (!proposal.valid)
                        continue;
                    proposal.Rlc = Rlc;

                    // Score the proposal.
                    distErrs.emplace_back(DistanceError(proposal, bbox, distMap));
                    alignErrs.emplace_back(AlignmentError(proposal, lineSegs, vp1, vp2, vp3));
                    shapeErrs.emplace_back(ShapeError(proposal, shapeErrThresh));

                    if (flip) {
                        swap(proposal.corners[1], proposal.corners[2]);
                        swap(proposal.corners[5], proposal.corners[6]);
                    }

                    candidates.emplace_back(proposal);
                }
            }
        }
    }

    const int numCandidates = static_cast<const int>(candidates.size());
    if (!numCandidates)
        return {};  // Return an empty cuboid proposal.

    const float minDistErr = *min_element(distErrs.begin(), distErrs.end());
    const float maxDistErr = *max_element(distErrs.begin(), distErrs.end());
    const float minAlignErr = *min_element(alignErrs.begin(), alignErrs.end());
    const float maxAlignErr = *max_element(alignErrs.begin(), alignErrs.end());

    int bestProposalIdx = -1;
    float bestErr = -1;
    for (int i = 0; i < numCandidates; ++i) {
        // Sum the errors by weight.
        float normDistErr = (distErrs[i] - minDistErr) / (maxDistErr - minDistErr);
        float normAlignErr = (alignErrs[i] - minAlignErr) / (maxAlignErr - minAlignErr);
        float totalErr = (normDistErr + alignErrWeight * normAlignErr) / (1 + alignErrWeight)
                         + shapeErrWeight * shapeErrs[i];

        if (totalErr < bestErr || bestErr == -1) {
            bestErr = totalErr;
            bestProposalIdx = i;

            if (display || save) {
                const auto& proposal = candidates[i];

                canvas = image.clone();
                rectangle(canvas, bbox, Scalar(255, 0, 0), 1, CV_AA);
                DrawCuboidProposal(canvas, proposal, bbox, K);

                for (auto& seg : lineSegs) {
                    line(canvas, seg->first, seg->second, Scalar(0, 0, 255), 1, CV_AA);
                }

                if (display) {
                    Vec3f theta = EulerAnglesFromRotation(proposal.Rlc);
                    cout << "Roll=" << theta[0] * 180 / M_PI
                         << " Pitch=" << theta[1] * 180 / M_PI
                         << " Yaw=" << theta[2] * 180 / M_PI << endl;
                    cout << normDistErr / (1 + alignErrWeight) << ' '
                         << normAlignErr * alignErrWeight / (1 + alignErrWeight) << ' '
                         << shapeErrWeight * shapeErrs[i] << endl;

                    imshow("Proposal", canvas);
                    waitKey(0);
                }
                if (save)
                    imwrite("Outputs/" + to_string(frameId) + "_" + to_string(objId) + "_" + to_string(i) + ".jpg",
                            canvas);
            }
        }
    }

    return candidates[bestProposalIdx];
}

CuboidProposal GenerateCuboidProposal(const cv::Rect& bbox, int topX,
                                      const cv::Point2f& vp1, const cv::Point2f& vp2, const cv::Point2f& vp3)
{
    if (vp3.x <= bbox.x || vp3.x >= bbox.x + bbox.width || vp3.y <= bbox.y + bbox.height)
        return CuboidProposal();
    if (vp1.y > bbox.y || vp2.y > bbox.y)
        return CuboidProposal();

    bool flip = (vp1.x > bbox.x + bbox.width && vp2.x < bbox.x) ||
                (vp2.x > bbox.x && vp2.x < bbox.x + bbox.width);
    if (flip) {
        CuboidProposal proposal = GenerateCuboidProposal(bbox, topX, vp2, vp1, vp3);
        swap(proposal.corners[1], proposal.corners[2]);
        swap(proposal.corners[5], proposal.corners[6]);
        return proposal;
    }

    CuboidProposal proposal;

    proposal.corners[0] = Point2f(topX, bbox.y);

    // Compute the other corners with respect to the pose, vanishing points and the bounding box.
    if (vp1.x < bbox.x && vp2.x > bbox.x + bbox.width) {
        // 3 faces
        proposal.corners[1] = LineIntersectionX(vp1, proposal.corners[0], bbox.x + bbox.width);
        if (!inside(proposal.corners[1], bbox))
            return proposal;
        proposal.corners[2] = LineIntersectionX(vp2, proposal.corners[0], bbox.x);
        if (!inside(proposal.corners[2], bbox))
            return proposal;
        proposal.corners[3] = LineIntersection(vp1, proposal.corners[2], vp2, proposal.corners[1]);
        if (!inside(proposal.corners[3], bbox))
            return proposal;
        proposal.corners[4] = LineIntersectionY(vp3, proposal.corners[3], bbox.y + bbox.height);
        if (!inside(proposal.corners[4], bbox))
            return proposal;
        proposal.corners[5] = LineIntersection(vp1, proposal.corners[4], vp3, proposal.corners[2]);
        if (!inside(proposal.corners[5], bbox))
            return proposal;
        proposal.corners[6] = LineIntersection(vp2, proposal.corners[4], vp3, proposal.corners[1]);
        if (!inside(proposal.corners[6], bbox))
            return proposal;
        proposal.corners[7] = LineIntersection(vp1, proposal.corners[6], vp2, proposal.corners[5]);
        if (!inside(proposal.corners[7], bbox))
            return proposal;

        proposal.isCornerVisible[4] = true;
        proposal.isCornerVisible[5] = true;
        proposal.isCornerVisible[6] = true;
        proposal.isCornerVisible[7] = false;
    }
    else if (vp1.x > bbox.x && vp1.x < bbox.x + bbox.width) {
        if (vp2.x < bbox.x) {
            // 2 faces
            proposal.corners[1] = LineIntersectionX(vp1, proposal.corners[0], bbox.x);
            if (!inside(proposal.corners[1], bbox))
                return proposal;
            proposal.corners[3] = LineIntersectionX(vp2, proposal.corners[1], bbox.x + bbox.width);
            if (!inside(proposal.corners[3], bbox))
                return proposal;
        }
        else if (vp2.x > bbox.x + bbox.width) {
            // 2 faces
            proposal.corners[1] = LineIntersectionX(vp1, proposal.corners[0], bbox.x + bbox.width);
            if (!inside(proposal.corners[1], bbox))
                return proposal;
            proposal.corners[3] = LineIntersectionX(vp2, proposal.corners[1], bbox.x);
            if (!inside(proposal.corners[3], bbox))
                return proposal;
        }
        else
            return proposal;
        proposal.corners[2] = LineIntersection(vp1, proposal.corners[3], vp2, proposal.corners[0]);
        if (!inside(proposal.corners[2], bbox))
            return proposal;
        proposal.corners[4] = LineIntersectionY(vp3, proposal.corners[3], bbox.y + bbox.height);
        if (!inside(proposal.corners[4], bbox))
            return proposal;
        proposal.corners[5] = LineIntersection(vp1, proposal.corners[4], vp3, proposal.corners[2]);
        if (!inside(proposal.corners[5], bbox))
            return proposal;
        proposal.corners[6] = LineIntersection(vp2, proposal.corners[4], vp3, proposal.corners[1]);
        if (!inside(proposal.corners[6], bbox))
            return proposal;
        proposal.corners[7] = LineIntersection(vp1, proposal.corners[6], vp2, proposal.corners[5]);
        if (!inside(proposal.corners[7], bbox))
            return proposal;

        proposal.isCornerVisible[4] = true;
        proposal.isCornerVisible[5] = true;
        proposal.isCornerVisible[6] = false;
        proposal.isCornerVisible[7] = false;
    }
    else
        return proposal;

    proposal.valid = true;
    return proposal;
}

}