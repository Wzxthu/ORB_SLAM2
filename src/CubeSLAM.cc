#include "CubeSLAM.h"
#include "ObjectDetector.h"

#include <numeric>

using namespace std;
using namespace cv;

namespace ORB_SLAM2 {

vector<vector<float>> PrecomputeChamferDistMap(const Rect& bbox,
                                               const vector<LineSegment*>& edges);

inline float ChamferDist(const LineSegment& hypothesis,
                         const Rect& bbox,
                         const vector<vector<float>>& distMap,
                         int numSamples = 16)
{
    float dx = (hypothesis.second.x - hypothesis.first.x) / (numSamples - 1);
    float dy = (hypothesis.second.y - hypothesis.first.y) / (numSamples - 1);
    float x = hypothesis.first.x - bbox.x;
    float y = hypothesis.first.y - bbox.y;
    float chamferDist = 0;
    for (int i = 0; i < numSamples; ++i) {
        chamferDist += distMap[int(y)][int(x)];
        x += dx;
        y += dy;
    }
    return chamferDist / numSamples;
}

static inline float
DistanceError(const Cuboid2D& proposal, const Rect& bbox, const vector<vector<float>>& distMap)
{
    int horizontalCnt = 4 + proposal.isCornerVisible[5] + proposal.isCornerVisible[6];
    int verticalCnt = 1 + proposal.isCornerVisible[5] + proposal.isCornerVisible[6];
    float distErr = 0;
    float weight_sum = 0;
    distErr += verticalCnt * ChamferDist(make_pair(proposal.corners[0], proposal.corners[1]), bbox, distMap);
    distErr += verticalCnt * ChamferDist(make_pair(proposal.corners[0], proposal.corners[2]), bbox, distMap);
    distErr += verticalCnt * ChamferDist(make_pair(proposal.corners[1], proposal.corners[3]), bbox, distMap);
    distErr += verticalCnt * ChamferDist(make_pair(proposal.corners[2], proposal.corners[3]), bbox, distMap);
    distErr += horizontalCnt * ChamferDist(make_pair(proposal.corners[3], proposal.corners[4]), bbox, distMap);
    weight_sum += (verticalCnt << 2) + horizontalCnt;
    if (proposal.isCornerVisible[6]) {
        distErr += horizontalCnt * ChamferDist(make_pair(proposal.corners[1], proposal.corners[6]), bbox, distMap);
        distErr += verticalCnt * ChamferDist(make_pair(proposal.corners[4], proposal.corners[6]), bbox, distMap);
        weight_sum += verticalCnt + horizontalCnt;
    }
    if (proposal.isCornerVisible[5]) {
        distErr += horizontalCnt * ChamferDist(make_pair(proposal.corners[2], proposal.corners[5]), bbox, distMap);
        distErr += verticalCnt * ChamferDist(make_pair(proposal.corners[4], proposal.corners[5]), bbox, distMap);
        weight_sum += verticalCnt + horizontalCnt;
    }
    distErr /= weight_sum;
    return distErr;
}

static inline float AlignmentError(const Cuboid2D& proposal, const vector<LineSegment*>& lineSegs,
                                   const Point2f& vp1, const Point2f& vp2, const Point2f& vp3)
{
    float scoreSum[3] = {};
    for (const auto& seg : lineSegs) {
        float err1 = AlignmentError(vp1, *seg);
        float err2 = AlignmentError(vp2, *seg);
        float err3 = AlignmentError(vp3, *seg);

        float minErr = err3;
        int minErrIdx = 2;

        if (err1 < 15.f / 180 * M_PI_F && err1 < minErr) {
            minErr = err1;
            minErrIdx = 0;
        }

        if (err2 < 15.f / 180 * M_PI_F && err2 < minErr) {
            minErr = err2;
            minErrIdx = 1;
        }

        if (minErr < 20.f / 180 * M_PI_F) {
            float len = GetLength(*seg);
            scoreSum[minErrIdx] += len * exp(-minErr / (20.f / 180 * M_PI_F));
        }
    }
    float alignErr = 0;
    for (int i = 0; i < 3; ++i) {
        const float weight = (i == 2 ? 2 : 1);
        if (scoreSum[i] != 0)
            alignErr += weight / scoreSum[i];
        else
            alignErr += weight;
    }

    return alignErr;
}

static inline float ShapeError(const Cuboid2D& proposal, float shapeErrThresh)
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
    shapeErr = max(shapeErr - shapeErrThresh, 0.f);
    return shapeErr;
}

Cuboid2D FindBestProposal(const Rect& bbox, const vector<LineSegment*>& lineSegs, const Mat& K,
                          float shapeErrThresh, float shapeErrWeight, float alignErrWeight,
                          float refRoll, float refPitch,
                          float rollRange, float pitchRange,
                          const unsigned long frameId, int objId, const Mat& image,
                          bool display, bool save)
{
    auto distMap = PrecomputeChamferDistMap(bbox, lineSegs);

    // Sample different proposals.
    Mat canvas;
    vector<float> distErrs, alignErrs, shapeErrs;
    vector<Cuboid2D> candidates;
    {
        const auto topXStep = max(2, bbox.width / 10);
        const auto topXStart = bbox.x + (topXStep >> 1);
        const auto topXEnd = bbox.x + bbox.width - (topXStep >> 1);
        const auto rollStep = min(rollRange / 5, 3 * M_PI_F / 180);
        const auto rollStart = refRoll - rollRange;
        const auto rollEnd = refRoll + rollRange;
        const auto pitchStep = min(pitchRange / 5, 3 * M_PI_F / 180);
        const auto pitchStart = refPitch - pitchRange;
        const auto pitchEnd = refPitch + pitchRange;
        const auto yawStep = 6 * M_PI_F / 180;
        const auto yawStart = -M_PI_F - 90 * M_PI_F / 180;
        const auto yawEnd = -M_PI_F + 90 * M_PI_F / 180;
        // Sample the landmark pitch in 90 degrees around the camera pitch.
        for (float l_pitch = pitchStart; l_pitch <= pitchEnd; l_pitch += pitchStep) {
            // Sample the landmark roll in 90 degrees around the camera roll.
            for (float l_roll = rollStart; l_roll <= rollEnd; l_roll += rollStep) {
                // Sample the landmark yaw in 180 degrees around the camera yaw.
                for (float l_yaw = yawStart; l_yaw <= yawEnd; l_yaw += yawStep) {
                    // Recover rotation of the landmark.
                    Mat Rlc = EulerAnglesToRotationMatrix(Vec3f(l_roll, l_pitch, l_yaw));

                    // Compute the vanishing points from the pose.
                    Mat vp1Homo = K * Rlc.col(0);
                    Mat vp3Homo = K * Rlc.col(1);
                    Mat vp2Homo = K * Rlc.col(2);
                    Point2f vp1 = PointFrom2DHomo(vp1Homo);
                    Point2f vp2 = PointFrom2DHomo(vp2Homo);
                    Point2f vp3 = PointFrom2DHomo(vp3Homo);

                    if (vp3.x <= bbox.x || vp3.x >= bbox.x + bbox.width || vp3.y <= bbox.y + bbox.height)
                        continue;
                    if (vp1.y > bbox.y || vp2.y > bbox.y)
                        continue;

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

                        candidates.emplace_back(proposal);
                    }
                }
            }
        }
    }

    const int numCandidates = static_cast<const int>(candidates.size());
    if (!numCandidates)
        return {};  // Return an empty cuboid proposal.

    int bestProposalIdx = -1;
    // Find the best proposal.
    {
        const float distErrMean = accumulate(distErrs.begin(), distErrs.end(), 0.f) / distErrs.size();
        const float distErrStdDev = sqrtf(inner_product(distErrs.begin(), distErrs.end(), distErrs.begin(), 0.f)
                                          / distErrs.size() - distErrMean * distErrMean);
        const float alignErrMean = accumulate(alignErrs.begin(), alignErrs.end(), 0.f) / alignErrs.size();
        const float alignErrStdDev = sqrtf(inner_product(alignErrs.begin(), alignErrs.end(), alignErrs.begin(), 0.f)
                                           / alignErrs.size() - alignErrMean * alignErrMean);

        float bestErr = -1, bestNormDistErr = -1, bestNormAlignErr = -1, bestShapeErr = -1;
        for (int i = 0; i < numCandidates; ++i) {
            // Sum the errors by weight.
            float normDistErr = (distErrs[i] - distErrMean) / distErrStdDev;
            float normAlignErr = (alignErrs[i] - alignErrMean) / alignErrStdDev;
            float totalErr = normDistErr + alignErrWeight * normAlignErr + shapeErrWeight * shapeErrs[i];

            if (totalErr < bestErr || bestErr == -1) {
                bestErr = totalErr;
                bestProposalIdx = i;
                bestNormDistErr = normDistErr;
                bestNormAlignErr = normAlignErr;
                bestShapeErr = shapeErrs[i];
            }

            if ((totalErr == bestErr) && (display || save)) {
                const auto& proposal = candidates[i];

                canvas = image.clone();
                rectangle(canvas, bbox, Scalar(0, 0, 0), 1, CV_AA);
                proposal.Draw(canvas, K);

                for (auto& seg : lineSegs) {
                    line(canvas, seg->first, seg->second, Scalar(0, 0, 255), 1, CV_AA);
                }

                if (display) {
                    Vec3f theta = EulerAnglesFromRotation(proposal.Rlc);
                    cout << "Roll=" << theta[0] * 180 / M_PI_F
                         << " Pitch=" << theta[1] * 180 / M_PI_F
                         << " Yaw=" << theta[2] * 180 / M_PI_F << endl;
                    cout << "Errors:\t" << normDistErr << "\t+\t" << normAlignErr * alignErrWeight << "\t+\t"
                         << shapeErrWeight * shapeErrs[i] << "\t=\t" << totalErr << endl;
                    cout << "Best:\t" << bestNormDistErr << "\t+\t" << bestNormAlignErr * alignErrWeight << "\t+\t"
                         << shapeErrWeight * bestShapeErr << "\t=\t" << bestErr << endl;

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

Cuboid2D GenerateCuboidProposal(const Rect& bbox, int topX,
                                const Point2f& vp1, const Point2f& vp2, const Point2f& vp3)
{
    if (vp3.x <= bbox.x || vp3.x >= bbox.x + bbox.width || vp3.y <= bbox.y + bbox.height)
        return Cuboid2D();
    if (vp1.y > bbox.y || vp2.y > bbox.y)
        return Cuboid2D();

    bool flip = (vp1.x > bbox.x + bbox.width && vp2.x < bbox.x) ||
                (vp2.x > bbox.x && vp2.x < bbox.x + bbox.width);
    if (flip) {
        Cuboid2D proposal = GenerateCuboidProposal(bbox, topX, vp2, vp1, vp3);
        swap(proposal.corners[1], proposal.corners[2]);
        swap(proposal.corners[5], proposal.corners[6]);
        return proposal;
    }

    Cuboid2D proposal;

    proposal.corners[0] = Point2f(topX, bbox.y);

    // Compute the other corners with respect to the pose, vanishing points and the bounding box.
    if (vp1.x <= bbox.x && vp2.x >= bbox.x + bbox.width) {
        // 3 faces
        proposal.corners[1] = LineIntersectionX(vp1, proposal.corners[0], bbox.x + bbox.width);
        if (!Inside(proposal.corners[1], bbox))
            return proposal;
        proposal.corners[2] = LineIntersectionX(vp2, proposal.corners[0], bbox.x);
        if (!Inside(proposal.corners[2], bbox))
            return proposal;
        proposal.corners[3] = LineIntersection(vp1, proposal.corners[2], vp2, proposal.corners[1]);
        if (!Inside(proposal.corners[3], bbox))
            return proposal;
        proposal.corners[4] = LineIntersectionY(vp3, proposal.corners[3], bbox.y + bbox.height);
        if (!Inside(proposal.corners[4], bbox))
            return proposal;
        proposal.corners[5] = LineIntersection(vp1, proposal.corners[4], vp3, proposal.corners[2]);
        if (!Inside(proposal.corners[5], bbox))
            return proposal;
        proposal.corners[6] = LineIntersection(vp2, proposal.corners[4], vp3, proposal.corners[1]);
        if (!Inside(proposal.corners[6], bbox))
            return proposal;
        proposal.corners[7] = LineIntersection(vp1, proposal.corners[6], vp2, proposal.corners[5]);
        if (!Inside(proposal.corners[7], bbox))
            return proposal;

        proposal.isCornerVisible[4] = true;
        proposal.isCornerVisible[5] = true;
        proposal.isCornerVisible[6] = true;
        proposal.isCornerVisible[7] = false;
    }
    else if (vp1.x >= bbox.x && vp1.x <= bbox.x + bbox.width) {
        if (vp2.x <= bbox.x) {
            // 2 faces
            proposal.corners[1] = LineIntersectionX(vp1, proposal.corners[0], bbox.x);
            if (!Inside(proposal.corners[1], bbox))
                return proposal;
            proposal.corners[3] = LineIntersectionX(vp2, proposal.corners[1], bbox.x + bbox.width);
            if (!Inside(proposal.corners[3], bbox))
                return proposal;
        }
        else if (vp2.x >= bbox.x + bbox.width) {
            // 2 faces
            proposal.corners[1] = LineIntersectionX(vp1, proposal.corners[0], bbox.x + bbox.width);
            if (!Inside(proposal.corners[1], bbox))
                return proposal;
            proposal.corners[3] = LineIntersectionX(vp2, proposal.corners[1], bbox.x);
            if (!Inside(proposal.corners[3], bbox))
                return proposal;
        }
        else
            return proposal;
        proposal.corners[2] = LineIntersection(vp1, proposal.corners[3], vp2, proposal.corners[0]);
        if (!Inside(proposal.corners[2], bbox))
            return proposal;
        proposal.corners[4] = LineIntersectionY(vp3, proposal.corners[3], bbox.y + bbox.height);
        if (!Inside(proposal.corners[4], bbox))
            return proposal;
        proposal.corners[5] = LineIntersection(vp1, proposal.corners[4], vp3, proposal.corners[2]);
        if (!Inside(proposal.corners[5], bbox))
            return proposal;
        proposal.corners[6] = LineIntersection(vp2, proposal.corners[4], vp3, proposal.corners[1]);
        if (!Inside(proposal.corners[6], bbox))
            return proposal;
        proposal.corners[7] = LineIntersection(vp1, proposal.corners[6], vp2, proposal.corners[5]);
        if (!Inside(proposal.corners[7], bbox))
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

vector<vector<float>> PrecomputeChamferDistMap(const Rect& bbox,
                                               const vector<LineSegment*>& edges)
{
    vector<vector<float>> distMap(static_cast<unsigned long>(bbox.height + 1),
                                  vector<float>(static_cast<unsigned long>(bbox.width + 1)));
    int longerSide = bbox.width * bbox.height;
    for (int y = 0; y <= bbox.height; ++y) {
        for (int x = 0; x <= bbox.width; ++x) {
            Point2f pt(x + bbox.x, y + bbox.y);
            float& minDist = distMap[y][x];
            minDist = longerSide;
            for (auto edge : edges) {
                float dist = Distance(pt, *edge);
                if (dist < minDist)
                    minDist = dist;
            }
        }
    }
    return distMap;
}

LandmarkDimension DimensionFromProposal(const Cuboid2D& proposal, const Mat& camCoordCentroid)
{
    auto upperCenter = LineIntersection(proposal.corners[0], proposal.corners[3],
                                        proposal.corners[1], proposal.corners[2]);
    auto frontCenter = LineIntersection(proposal.corners[0], proposal.corners[5],
                                        proposal.corners[2], proposal.corners[7]);
    auto leftCenter = LineIntersection(proposal.corners[2], proposal.corners[4],
                                       proposal.corners[3], proposal.corners[5]);

    return {
            2 * DistanceToRay(camCoordCentroid, proposal.Rlc.col(0), PointToHomo(frontCenter)),
            2 * DistanceToRay(camCoordCentroid, proposal.Rlc.col(0), PointToHomo(leftCenter)),
            2 * DistanceToRay(camCoordCentroid, proposal.Rlc.col(0), PointToHomo(upperCenter)),
    };
}

}