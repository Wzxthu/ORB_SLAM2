#pragma once

#include "Thirdparty/g2o/g2o/core/base_multi_edge.h"
#include "Thirdparty/g2o/g2o/types/sba/types_six_dof_expmap.h"
#include "matrix_utils.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <math.h>
#include <algorithm>    // std::swap

namespace Eigen {

typedef Eigen::Matrix<double, 9, 1> Vector9d;
typedef Eigen::Matrix<double, 9, 9> Matrix9d;
typedef Eigen::Matrix<double, 3, 8> Matrix38d;
typedef Eigen::Matrix<double, 10, 1> Vector10d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

}

namespace g2o {

class Cuboid {
public:
    inline Cuboid() { mScale.setZero(); }

    // xyz roll pitch yaw half_scale
    inline void fromMinimalVector(const Eigen::Vector9d& v)
    {
        Eigen::Quaterniond posequat = zyx_euler_to_quat(v(3), v(4), v(5));
        mPose = SE3Quat(posequat, v.head<3>());
        mScale = v.tail<3>();
    }

    // xyz quaternion, half_scale
    inline void fromVector(const Eigen::Vector10d& v)
    {
        mPose.fromVector(v.head<7>());
        mScale = v.tail<3>();
    }

    inline const Eigen::Vector3d& getTranslation() const { return mPose.translation(); }
    inline void setTranslation(const Eigen::Vector3d& t_) { mPose.setTranslation(t_); }
    inline void setRotation(const Eigen::Quaterniond& r_) { mPose.setRotation(r_); }
    inline void setRotation(const Eigen::Matrix3d& R) { mPose.setRotation(Eigen::Quaterniond(R)); }
    inline void setScale(const Eigen::Vector3d& scale_) { mScale = scale_; }
    inline void setScale(float length, float width, float height)
    {
        mScale[0] = length;
        mScale[1] = width;
        mScale[2] = height;
    }

    // apply update to current cuboid. exponential map
    inline Cuboid expUpdate(const Eigen::Vector9d& update)
    {
        Cuboid res;
        res.mPose = this->mPose * SE3Quat::exp(update.head<6>()); // NOTE bug before. switch position
        res.mScale = this->mScale + update.tail<3>();
        return res;
    }

    // actual error between two cuboids.
    inline Eigen::Vector9d logCubeError(const Cuboid& other) const
    {
        Eigen::Vector9d res;
        SE3Quat pose_diff = other.mPose.inverse() * this->mPose;
        res.head<6>() = pose_diff.log();   //treat as se3 log error. could also just use yaw error
        res.tail<3>() = this->mScale - other.mScale;
        return res;
    }

    // function called by g2o.
    inline Eigen::Vector9d minLogError(const Cuboid& other, bool printDetails = false) const
    {
        bool whether_rotate_cubes = true;  // whether rotate cube to find smallest error
        if (!whether_rotate_cubes)
            return logCubeError(other);

        // NOTE rotating cuboid... since we cannot determine the front face consistenly, different front faces indicate different yaw, scale representation.
        // need to rotate all 360 degrees (global cube might be quite different from local cube)
        // this requires the sequential object insertion. In this case, object yaw practically should not change much. If we observe a jump, we can use code
        // here to adjust the yaw.
        Eigen::Vector4d rotate_errors_norm;
        Eigen::Vector4d rotate_angles(-1, 0, 1, 2); // rotate -90 0 90 180
        Eigen::Matrix<double, 9, 4> rotate_errors;
        for (int i = 0; i < rotate_errors_norm.rows(); i++) {
            Cuboid rotated_cuboid = other.rotateCuboid(rotate_angles(i) * M_PI / 2.0);  // rotate new cuboids
            Eigen::Vector9d cuboid_error = this->logCubeError(rotated_cuboid);
            rotate_errors_norm(i) = cuboid_error.norm();
            rotate_errors.col(i) = cuboid_error;
        }
        int min_label;
        rotate_errors_norm.minCoeff(&min_label);
        if (printDetails)
            if (min_label != 1)
                std::cout << "Rotate cube   " << min_label << std::endl;
        return rotate_errors.col(min_label);
    }

    // change front face by rotate along current body z axis. another way of representing cuboid. representing same cuboid (IOU always 1)
    inline Cuboid rotateCuboid(double yaw_angle) const // to deal with different front surface of cuboids
    {
        Cuboid res;
        SE3Quat
        rot(Eigen::Quaterniond(cos(yaw_angle *
                                   0.5), 0, 0, sin(yaw_angle * 0.5)),
            Eigen::Vector3d(0, 0, 0));   // change yaw to rotation.
        res.mPose = this->mPose * rot;
        res.mScale = this->mScale;
        if ((yaw_angle == M_PI / 2.0) || (yaw_angle == -M_PI / 2.0) || (yaw_angle == 3 * M_PI / 2.0))
            std::swap(res.mScale(0), res.mScale(1));

        return res;
    }

    // transform a local cuboid to global cuboid  Twc is camera pose. from camera to world
    inline Cuboid transformFrom(const SE3Quat& Twc) const
    {
        Cuboid res;
        res.mPose = Twc * this->mPose;
        res.mScale = this->mScale;
        return res;
    }

    // transform a global cuboid to local cuboid  Twc is camera pose. from camera to world
    inline Cuboid transformTo(const SE3Quat& Twc) const
    {
        Cuboid res;
        res.mPose = Twc.inverse() * this->mPose;
        res.mScale = this->mScale;
        return res;
    }

    // xyz roll pitch yaw half_scale
    inline Eigen::Vector9d toMinimalVector() const
    {
        Eigen::Vector9d v;
        v.head<6>() = mPose.toXYZPRYVector();
        v.tail<3>() = mScale;
        return v;
    }

    // xyz quaternion, half_scale
    inline Eigen::Vector10d toVector() const
    {
        Eigen::Vector10d v;
        v.head<7>() = mPose.toVector();
        v.tail<3>() = mScale;
        return v;
    }

    inline Eigen::Matrix4d similarityTransform() const
    {
        Eigen::Matrix4d res = mPose.to_homogeneous_matrix();
        Eigen::Matrix3d scale_mat = mScale.asDiagonal();
        res.topLeftCorner<3, 3>() = res.topLeftCorner<3, 3>() * scale_mat;
        return res;
    }

    // 8 corners 3*8 matrix, each row is x y z
    inline Eigen::Matrix3Xd compute3D_BoxCorner() const
    {
        Eigen::Matrix3Xd corners_body;
        corners_body.resize(3, 8);
        corners_body << 1, 1, -1, -1, 1, 1, -1, -1,
                1, -1, -1, 1, 1, -1, -1, 1,
                -1, -1, -1, -1, 1, 1, 1, 1;
        Eigen::Matrix3Xd corners_world = homo_to_real_coord<double>(
                similarityTransform() * real_to_homo_coord<double>(corners_body));
        return corners_world;
    }

    // get rectangles after projection  [topleft, bottomright]
    inline Eigen::Vector4d projectOntoImageRect(const SE3Quat& campose_cw, const Eigen::Matrix3d& Kalib) const
    {
        Eigen::Matrix3Xd corners_3d_world = compute3D_BoxCorner();
        Eigen::Matrix2Xd corner_2d = homo_to_real_coord<double>(Kalib * homo_to_real_coord<double>(
                campose_cw.to_homogeneous_matrix() * real_to_homo_coord<double>(corners_3d_world)));
        Eigen::Vector2d bottomright = corner_2d.rowwise().maxCoeff(); // x y
        Eigen::Vector2d topleft = corner_2d.rowwise().minCoeff();
        return {topleft(0), topleft(1), bottomright(0), bottomright(1)};
    }

    // get rectangles after projection  [center, width, height]
    inline Eigen::Vector4d projectOntoImageBbox(const SE3Quat& campose_cw, const Eigen::Matrix3d& Kalib) const
    {
        Eigen::Vector4d rect_project = projectOntoImageRect(campose_cw, Kalib);  // top_left, bottom_right  x1 y1 x2 y2
        Eigen::Vector2d rect_center = (rect_project.tail<2>() + rect_project.head<2>()) / 2;
        Eigen::Vector2d widthheight = rect_project.tail<2>() - rect_project.head<2>();
        return {rect_center(0), rect_center(1), widthheight(0), widthheight(1)};
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
    SE3Quat mPose;  // 6 dof for object, object to world by default
    Eigen::Vector3d mScale; // [length, width, height]  half!
};

class VertexCuboid : public BaseVertex<9, Cuboid>  // NOTE  this vertex stores object pose to world
{
public:
    inline VertexCuboid() = default;

    inline void setToOriginImpl() override { _estimate = Cuboid(); }

    inline void oplusImpl(const double* update_) override
    {
        Eigen::Map<const Eigen::Vector9d> update(update_);
        setEstimate(_estimate.expUpdate(update));
    }

    inline bool read(std::istream& is) override
    {
        Eigen::Vector9d est;
        for (int i = 0; i < 9; i++)
            is >> est[i];
        Cuboid cuboid;
        cuboid.fromMinimalVector(est);
        setEstimate(cuboid);
        return true;
    }

    inline bool write(std::ostream& os) const override
    {
        Eigen::Vector9d lv = _estimate.toMinimalVector();
        for (int i = 0; i < lv.rows(); i++) {
            os << lv[i] << " ";
        }
        return os.good();
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// camera -object 3D error
class EdgeSE3Cuboid : public BaseBinaryEdge<9, Cuboid, VertexSE3Expmap, VertexCuboid> {
public:
    inline EdgeSE3Cuboid() = default;

    inline bool read(std::istream& is) override
    {
        return true;
    };

    inline bool write(std::ostream& os) const override
    {
        return os.good();
    };

    inline void computeError() override
    {
        const auto* SE3Vertex = dynamic_cast<const VertexSE3Expmap*>(_vertices[0]); //  world to camera pose
        const auto* cuboidVertex = dynamic_cast<const VertexCuboid*>(_vertices[1]); //  object pose to world

        SE3Quat cam_pose_Twc = SE3Vertex->estimate().inverse();
        const Cuboid& global_cube = cuboidVertex->estimate();
        Cuboid esti_global_cube = _measurement.transformFrom(cam_pose_Twc);
        _error = global_cube.minLogError(esti_global_cube);
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// camera -object 2D projection error, rectangle difference, could also change to iou
class EdgeSE3CuboidProj : public BaseBinaryEdge<4, Eigen::Vector4d, VertexSE3Expmap, VertexCuboid> {
public:
    EdgeSE3CuboidProj() = default;

    inline bool read(std::istream& is) override
    {
        return true;
    };

    inline bool write(std::ostream& os) const override
    {
        return os.good();
    };

    inline void computeError() override
    {
        const auto* SE3Vertex = dynamic_cast<const VertexSE3Expmap*>(_vertices[0]);  //  world to camera pose
        const auto* cuboidVertex = dynamic_cast<const VertexCuboid*>(_vertices[1]);       //  object pose to world

        const SE3Quat& cam_pose_Tcw = SE3Vertex->estimate();
        const Cuboid& global_cube = cuboidVertex->estimate();

        Eigen::Vector4d rect_project = global_cube.projectOntoImageBbox(cam_pose_Tcw, Kalib); // center, width, height

        _error = rect_project - _measurement;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:
    Eigen::Matrix3d Kalib;
};

}