#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bridge: wrist-camera detections -> environment camera bboxes (ROS Noetic)
- No tf2_geometry_msgs / PyKDL required.
- Uses numpy transform from geometry_msgs/TransformStamped.
- Publishes CentroidConfidenceArray in ENV optical frame.

Assumptions (tweak in CONFIG):
- Input JSON from your YOLO node on /detected_objects/dict_label_centroid
- Wrist depth point cloud on /camera/depth/color/points (aligned to wrist color)
- TF path exists: wrist_color_optical_frame -> scene_color_optical_frame
- Environment camera intrinsics on /scene_camera/camera/color/camera_info
"""

import json
import numpy as np
import rospy
from std_msgs.msg import String, Header
from sensor_msgs.msg import PointCloud2, CameraInfo
from geometry_msgs.msg import PointStamped
from image_geometry import PinholeCameraModel
import tf2_ros
np.float = np.float64
import ros_numpy

# ======== CONFIG (use your actual names) =========
INPUT_JSON_TOPIC      = "/detected_objects/dict_label_centroid"   # from your YOLO node
WRIST_PC2_TOPIC       = "/camera/depth/color/points"              # wrist depth cloud (aligned to color)
ENV_CAM_INFO_TOPIC    = "/scene_camera/camera/color/camera_info"  # env cam intrinsics
WRIST_OPTICAL_FRAME   = "camera_color_optical_frame"              # wrist RGB optical frame
ENV_OPTICAL_FRAME     = "static_camera_color_optical_frame"       # env RGB optical frame
OUTPUT_TOPIC          = "/goal_confidence_centroids"              # your viewer already uses this
# ================================================


# Your custom message types
from trust_and_transparency.msg import CentroidConfidence, CentroidConfidenceArray

# ROS tf quaternion -> rotation matrix (Noetic provides tf.transformations)
from tf.transformations import quaternion_matrix


def transform_point_numpy(pt_msg, T_msg):
    """
    Transform a geometry_msgs/PointStamped using a geometry_msgs/TransformStamped.
    target <- source transform: T_msg.header.frame_id is the target frame.
    """
    t = T_msg.transform.translation
    q = T_msg.transform.rotation
    R = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]  # 3x3
    p = np.array([pt_msg.point.x, pt_msg.point.y, pt_msg.point.z], dtype=np.float64)
    p_out = R.dot(p) + np.array([t.x, t.y, t.z], dtype=np.float64)

    out = PointStamped()
    out.header.stamp = pt_msg.header.stamp
    out.header.frame_id = T_msg.header.frame_id  # target frame
    out.point.x, out.point.y, out.point.z = p_out.tolist()
    return out


class WristToEnvProjector(object):
    def __init__(self):
        rospy.init_node("wrist_to_env_projector", anonymous=True)

        # TF buffer/listener
        self.tf_buf = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
        self.tf_lis = tf2_ros.TransformListener(self.tf_buf)

        # Env camera pinhole model
        self.env_cam = PinholeCameraModel()
        self.have_env_info = False

        # Latest wrist depth cloud (H×W structured np array with fields ['x','y','z'])
        self.pc_arr = None
        self.pc_w = None
        self.pc_h = None

        # Subscriptions
        rospy.Subscriber(WRIST_PC2_TOPIC, PointCloud2, self._pc_cb, queue_size=1)
        rospy.Subscriber(ENV_CAM_INFO_TOPIC, CameraInfo, self._env_info_cb, queue_size=1)
        rospy.Subscriber(INPUT_JSON_TOPIC, String, self._detections_cb, queue_size=1)

        # Publisher
        self.pub_out = rospy.Publisher(OUTPUT_TOPIC, CentroidConfidenceArray, queue_size=10)

        rospy.loginfo("wrist_to_env_projector: ready")
        rospy.spin()

    # ---- Callbacks ----
    def _env_info_cb(self, msg: CameraInfo):
        if not self.have_env_info:
            self.env_cam.fromCameraInfo(msg)
            self.have_env_info = True
            rospy.loginfo("wrist_to_env_projector: got environment CameraInfo")

    def _pc_cb(self, msg: PointCloud2):
        try:
            arr = ros_numpy.numpify(msg)
            if arr.ndim == 1:
                # Some drivers expose as (H*W,), reshape using msg.height/width
                arr = arr.reshape(msg.height, msg.width)
            self.pc_arr = arr
            self.pc_h, self.pc_w = arr.shape[0], arr.shape[1]
        except Exception as e:
            rospy.logwarn_throttle(1.0, f"PointCloud2 convert failed: {e}")

    def _detections_cb(self, msg: String):
        if self.pc_arr is None or not self.have_env_info:
            return

        try:
            dets = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn_throttle(1.0, f"Bad JSON on {INPUT_JSON_TOPIC}: {e}")
            return

        # Fetch transform once per batch
        try:
            T_env_wrist = self.tf_buf.lookup_transform(
                ENV_OPTICAL_FRAME, WRIST_OPTICAL_FRAME, rospy.Time(0), rospy.Duration(0.2)
            )
        except Exception as e:
            rospy.logwarn_throttle(1.0, f"TF lookup {ENV_OPTICAL_FRAME}<-{WRIST_OPTICAL_FRAME} failed: {e}")
            return

        out_msg = CentroidConfidenceArray()  # NOTE: no header field

        for d in dets:
            # Expect: {"id":..., "label": str, "centroid":{x,y,z}, "bounding_box":{x1,y1,x2,y2}}
            try:
                label = d.get("label", "obj")
                c = d["centroid"]
                bb = d["bounding_box"]
                x1, y1, x2, y2 = int(bb["x1"]), int(bb["y1"]), int(bb["x2"]), int(bb["y2"])
            except Exception as e:
                rospy.logwarn_throttle(1.0, f"Missing fields in detection: {e}")
                continue

            # ---- Transform centroid to ENV optical frame (numpy-based) ----
            pt_w = PointStamped()
            pt_w.header.stamp = rospy.Time(0)
            pt_w.header.frame_id = WRIST_OPTICAL_FRAME
            pt_w.point.x = float(c["x"])
            pt_w.point.y = float(c["y"])
            pt_w.point.z = float(c["z"])

            try:
                pt_env = transform_point_numpy(pt_w, T_env_wrist)
            except Exception as e:
                rospy.logwarn_throttle(1.0, f"Centroid transform failed: {e}")
                continue

            # ---- Build env-image bbox by sampling 9 pixels in the wrist bbox ----
            env_bbox = self._reproject_bbox_to_env(x1, y1, x2, y2, T_env_wrist)

            if env_bbox is None:
                # Fallback: project only the transformed centroid to pixels
                xyz = [pt_env.point.x, pt_env.point.y, pt_env.point.z]
                if xyz[2] <= 0:
                    continue
                try:
                    u_env, v_env = self.env_cam.project3dToPixel(xyz)
                    u_env = int(round(u_env)); v_env = int(round(v_env))
                    env_bbox = (u_env - 5, v_env - 5, u_env + 5, v_env + 5)
                except Exception:
                    continue

            item = CentroidConfidence()
            item.label = label
            item.centroid.x = pt_env.point.x
            item.centroid.y = pt_env.point.y
            item.centroid.z = pt_env.point.z
            item.confidence = 1.0  # set real score if available
            item.x1, item.y1, item.x2, item.y2 = env_bbox
            out_msg.items.append(item)

        if out_msg.items:
            self.pub_out.publish(out_msg)

    # ---- Helpers ----
    def _reproject_bbox_to_env(self, x1, y1, x2, y2, T_env_wrist):
        """Sample 9 wrist pixels in bbox, lift via depth cloud, transform to ENV, project to ENV pixels."""
        if self.pc_arr is None:
            return None

        # clamp bbox to wrist cloud bounds
        x1 = max(0, min(self.pc_w - 1, x1))
        x2 = max(0, min(self.pc_w - 1, x2))
        y1 = max(0, min(self.pc_h - 1, y1))
        y2 = max(0, min(self.pc_h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return None

        xc = (x1 + x2) // 2
        yc = (y1 + y2) // 2
        sample_uv = [
            (x1, y1), (xc, y1), (x2, y1),
            (x1, yc), (xc, yc), (x2, yc),
            (x1, y2), (xc, y2), (x2, y2),
        ]

        env_pixels = []
        for u, v in sample_uv:
            p = self.pc_arr[v, u]
            X, Y, Z = float(p['x']), float(p['y']), float(p['z'])
            if not np.isfinite(X) or not np.isfinite(Y) or not np.isfinite(Z) or Z <= 0:
                continue

            ps = PointStamped()
            ps.header.stamp = rospy.Time(0)
            ps.header.frame_id = WRIST_OPTICAL_FRAME
            ps.point.x, ps.point.y, ps.point.z = X, Y, Z

            try:
                ps_env = transform_point_numpy(ps, T_env_wrist)
            except Exception:
                continue

            xyz = [ps_env.point.x, ps_env.point.y, ps_env.point.z]
            if xyz[2] <= 0:
                continue

            try:
                u_env, v_env = self.env_cam.project3dToPixel(xyz)
                env_pixels.append((int(round(u_env)), int(round(v_env))))
            except Exception:
                continue

        if not env_pixels:
            return None

        xs = [p[0] for p in env_pixels]
        ys = [p[1] for p in env_pixels]
        return (min(xs), min(ys), max(xs), max(ys))


if __name__ == "__main__":
    try:
        WristToEnvProjector()
    except rospy.ROSInterruptException:
        pass
