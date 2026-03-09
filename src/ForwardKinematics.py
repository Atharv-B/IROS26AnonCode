"""
This class specifies the forward kinematics of the robot and allows for inverse computation of points from camera frame
into the world frame.
For future reference, we have the Kinova Gen3 Ultra Lightweight 7 DoF Arm.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Cosine Helper Function
def c(t): return np.cos(t)


# Sine Helper Function
def s(t): return np.sin(t)


class ForwardKinematicsKinova:
    def __init__(self, base_ref=None):
        self.joint_angles = np.zeros(7)  # Start in the zero-angle configuration
        self.base_ref = base_ref if base_ref is not None else np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.quivers = []
        self.ref_point_in_world = [-0.5, -0.5, 0.4]
        self.draw_pc = None

    def update_joints(self, q):
        if not hasattr(q, "__len__"):
            # Q is not an iterable, return
            print("Update_Joints Called but parameter does not have the correct datatype (Iterable)")
            return
        assert len(q) == len(self.joint_angles)
        self.joint_angles = q

    def base_to_inter_frame_transform(self, track_quivers=False):

        coord_frames = []
        coord_frames.append(self.base_ref)
        coord_frames.append(np.matmul(coord_frames[-1], self.base_to_f1()))
        coord_frames.append(np.matmul(coord_frames[-1], self.f1_to_f2()))
        coord_frames.append(np.matmul(coord_frames[-1], self.f2_to_f3()))
        coord_frames.append(np.matmul(coord_frames[-1], self.f3_to_f4()))
        coord_frames.append(np.matmul(coord_frames[-1], self.f4_to_f5()))
        coord_frames.append(np.matmul(coord_frames[-1], self.f5_to_f6()))
        coord_frames.append(np.matmul(coord_frames[-1], self.f6_to_f7()))
        coord_frames.append(np.matmul(coord_frames[-1], self.f7_to_inter()))

        if track_quivers:
            self.quivers = []
            for cf in coord_frames:
                self.quivers.append(cf[:3, 3])
        return coord_frames[-1]

    def fk(self):
        return self.base_to_inter_frame_transform()[:3, 3]

    def get_depth_camera_transform(self):
        transform = self.base_to_inter_frame_transform(False)
        depth_camera = np.matmul(transform, self.inter_to_mounted_color_camera())
        return depth_camera
        # return transform

    def camera_xyz_to_world(self, camera_xyz: np.array):
        """
        Convert a matrix of points in the camera space to a matrix of points in the world space
        :param camera_xyz: An 3xN or 4xN (homogeneous) matrix of N points
        :return: A 4xN Matrix of Homogeneous Points in the World Space
        """

        assert len(camera_xyz) == 3 or len(camera_xyz) == 4
        # If the points are not homogeneous, convert them
        P = camera_xyz
        if len(camera_xyz) == 3:
            P = np.vstack((P, np.ones((1, P.shape[1]))))
        # print(P.shape)
        return np.matmul(self.get_depth_camera_transform(), P)

    def draw(self, name=None, ax=None, custom_sphere=None):
        if len(self.quivers) == 0:
            return
        self.quivers = np.array(self.quivers)
        # print(self.quivers.shape)
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        c = ["red"] + ["blue"] * (len(self.quivers) - 2) + ["green"]
        try:
            ax.plot(self.quivers[:, 0], self.quivers[:, 1], self.quivers[:, 2])
            ax.scatter(self.quivers[:, 0], self.quivers[:, 1], self.quivers[:, 2], c=c)
        except Exception as e:
            return

        if self.draw_pc is not None:
            ax.scatter(self.draw_pc[0], self.draw_pc[1], self.draw_pc[2], c="black", s=2)

        if custom_sphere is not None:
            ax.scatter([custom_sphere[0]], [custom_sphere[1]], [custom_sphere[2]], c="purple", s=20)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(0, 1)
        # ax.set_box_aspect((1, 1, 1))
        # plt.show()
        if name is not None:
            plt.savefig(f'{name}.png')

    def base_to_f1(self):
        q1 = self.joint_angles[0]  # Get the joint angle
        transform = np.array([
            [c(q1), -s(q1), 0, 0],
            [-s(q1), -c(q1), 0, 0],
            [0, 0, -1, 0.1564],
            [0, 0, 0, 1]
        ])
        return transform

    def f1_to_f2(self):
        q2 = self.joint_angles[1]
        transform = np.array([
            [c(q2), -s(q2), 0, 0],
            [0, 0, -1, 0.0054],
            [s(q2), c(q2), 0, -0.1284],
            [0, 0, 0, 1]
        ])
        return transform

    def f2_to_f3(self):
        q3 = self.joint_angles[2]
        transform = np.array([
            [c(q3), -s(q3), 0, 0],
            [0, 0, 1, -0.2104],
            [-s(q3), -c(q3), 0, -0.0064],
            [0, 0, 0, 1]
        ])
        return transform

    def f3_to_f4(self):
        q4 = self.joint_angles[3]
        transform = np.array([
            [c(q4), -s(q4), 0, 0],
            [0, 0, -1, 0.0064],
            [s(q4), c(q4), 0, -0.2104],
            [0, 0, 0, 1]
        ])
        return transform

    def f4_to_f5(self):
        q5 = self.joint_angles[4]
        transform = np.array([
            [c(q5), -s(q5), 0, 0],
            [0, 0, 1, -0.2084],
            [-s(q5), -c(q5), 0, -0.0064],
            [0, 0, 0, 1]
        ])
        return transform

    def f5_to_f6(self):
        q6 = self.joint_angles[5]
        transform = np.array([
            [c(q6), -s(q6), 0, 0],
            [0, 0, -1, 0],
            [s(q6), c(q6), 0, -0.1059],
            [0, 0, 0, 1]
        ])
        return transform

    def f6_to_f7(self):
        q7 = self.joint_angles[6]
        transform = np.array([
            [c(q7), -s(q7), 0, 0],
            [0, 0, 1, -0.1059],
            [-s(q7), -c(q7), 0, 0],
            [0, 0, 0, 1]
        ])
        return transform

    def f7_to_inter(self):
        transform = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, -0.0615],
            # [0, 0, -1, -0.1805],
            [0, 0, 0, 1]
        ])
        return transform

    # def inter_to_color_sensor(self):
    #     transform = np.array([
    #         [-1, 0, 0, 0],
    #         [0, -1, 0, 0.05639],
    #         [0, 0, 1, -0.00305],
    #         [0, 0, 0, 1]
    #     ])
    #     return transform
    #
    # def inter_to_depth_sensor(self):
    #     transform = np.array([
    #         [-1, 0, 0, 0.0275],
    #         [0, -1, 0, 0.06600],
    #         [0, 0, 1, -0.00305],
    #         [0, 0, 0, 1]
    #     ])
    #     return transform

    def inter_to_mounted_color_camera(self):
        transform = np.array([
            [-1, 0, 0, 0.038],
            [0, -1, 0, 0.05639],
            [0, 0, 1, 0.03195],
            [0, 0, 0, 1]
        ])
        return transform

    def homogeneous_inverse(self, transform):
        R = transform[:3, :3]
        t = transform[:3, 3]
        inverse_transform = np.eye(4)
        inverse_transform[:3, :3] = R.T
        inverse_transform[:3, 3] = -R.T @ t
        return inverse_transform
        # ret = np.zeros((4, 4))
        # ret[:3, :3] = R.T
        # ret[:3, 3] = np.matmul(-R.T, t)
        # ret[-1, -1] = 1
        # return ret

    def camera_to_wrold(self, camera_points):
        if camera_points.shape[0] != 4:
            ones = np.ones((1, camera_points[1]))
            camera_points = np.vstack((camera_points, ones))

        camera_to_world_transform = self.get_depth_camera_transform()
        world_points = camera_to_world_transform @ camera_points
        return world_points[:3]
    


if __name__ == "__main__":
    # fk = ForwardKinematicsKinova()
    # joint_angles = np.array([0, 0, 0, 0, 0, 0, 0, ])
    # fk.update_joints(joint_angles)
    # camera_points = np.array([[0.1, 0.1, 0.1],
    #                          [0.2, 0.2, 0.2],
    #                          [0.3, 0.3, 0.3]])
    # world_points = fk.camera_to_wrold(camera_points)
    # print(world_points)
    sample_point_cloud = np.random.random((3, 1000))
    sample_point_cloud[:2] -= 0.5
    sample_point_cloud[2] *= 0.1
    sample_point_cloud[2] += 0.15

    joint_motion = np.arange(0, -np.pi/4, -0.05)
    for i, j in enumerate(joint_motion):
        joints = np.ones(7) * j
        # joints[-3:] = 0
        fk = ForwardKinematicsKinova()
        fk.update_joints(joints)
        transform = fk.base_to_inter_frame_transform(True)
        print("EE: ", transform[:, 3])

        depth_camera = np.matmul(fk.inter_to_depth_sensor(), transform)
        fk.quivers.append(depth_camera[:3, 3])

        point_in_camera = np.matmul(fk.homogeneous_inverse(depth_camera), np.array(fk.ref_point_in_world + [1]))
        # print(point_in_camera)

        # point_in_world = np.matmul(depth_camera, point_in_camera)
        # print(point_in_world)

        point_in_world = np.matmul(depth_camera, np.array([0, 0, 0.1, 1]))
        print(point_in_world)

        new_pc = fk.camera_xyz_to_world(sample_point_cloud)
        fk.draw_pc = new_pc
        fk.draw(f"frame{i}")