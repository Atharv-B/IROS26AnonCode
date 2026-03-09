#!/usr/bin/env python3

import rospy
import numpy as np
import json
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32, Point
from std_msgs.msg import String
from trust_and_transparency.msg import CentroidConfidence, CentroidConfidenceArray
from sag import SAGTeleoperation
from constants import STOP_SCAN_THRESHOLD, PLACEMENT_THRESHOLDS, TOP_DOWN, SPEED_CONTROL
from cam_to_world import CameraToWorld


class VOSATeleoperation(SAGTeleoperation):
    def __init__(self):
        super().__init__()

        self.raw_objects = []
        self.intermediate_position_reached = True
        self.world_detections = []
        self.pick_set = []
        self.adjusted_z = None
        self.Z_is_updated = False
        self.ctw = CameraToWorld()
        self.confidences = []

        self.centroid_conf_pub = rospy.Publisher('/goal_confidence_centroids', CentroidConfidenceArray, queue_size=10)

        rospy.Subscriber("/clusters", PointCloud, self.centroid_callback)
        rospy.Subscriber('/detected_objects/dict_label_centroid', String, self.dict_label_centroid_callback)
        rospy.Subscriber('/centroid_to_world', String, self.centroid_to_world_callback)

        rospy.loginfo("[VOSA] Initialized with dynamic pick set subscription.")

    def centroid_callback(self, msg):
        # Similar to original: only proceed if end-effector is above threshold
        if self.ee_position[2] < 0.38:
            return
        Z_LIM = [-0.005, 0.20]
        detected = []
        for pt in msg.points:
            # Apply offset and minimum z similar to original
            # centroid = np.array([pt.x + 0.025, pt.y - 0.01, max(pt.z + 0.03, 0.10)])
            centroid = np.array([pt.x, pt.y , pt.z])

            # Filter by y thresholds as in first code
            # if pt.y > PLACEMENT_THRESHOLDS[self.task]:
            #     detected.append(centroid)
            if Z_LIM[0] < centroid[2] < Z_LIM[1]:
                detected.append(centroid)
                rospy.logdebug(f"[VOSA] Detected centroid: {centroid}")

        known_goal_pose = len(detected) > 0

        if not self.intermediate_position_reached and not known_goal_pose:
            return

        self.intermediate_position_reached = True

        self.pick_set = detected
        rospy.loginfo(f"[VOSA] Updated pick set with {len(detected)} centroids.")

    def dict_label_centroid_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            if isinstance(data, list):
                for item in data:
                    self.process_single_dict_detection(item)
            elif isinstance(data, dict):
                self.process_single_dict_detection(data)
            else:
                rospy.logwarn(f"[VOSA] Unexpected data format in dict_label_centroid_callback: {type(data)}")
        except Exception as e:
            rospy.logerr(f"[VOSA] Error in dict_label_centroid_callback: {e}")

    def process_single_dict_detection(self, data):
        try:
            label = data.get("label", "")
            cen = data.get("centroid", {})
            wx, wy, wz = cen.get("x"), cen.get("y"), cen.get("z")

            if None in (wx, wy, wz):
                rospy.logwarn(f"[VOSA] Incomplete centroid JSON for {label}")
                return

            # Remove PLACEMENT_THRESHOLDS filtering to match centroid_callback behavior
            # This ensures consistent object detection across all callbacks
            self.raw_objects.append((label, np.array([wx, wy, wz])))
            rospy.loginfo(f"[VOSA] Added raw object: {label} at ({wx:.3f}, {wy:.3f}, {wz:.3f})")
        except Exception as e:
            rospy.logerr(f"[VOSA] Error processing single detection: {e}")

    def centroid_to_world_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            if not isinstance(data, list):
                rospy.logwarn("[VOSA] Expected list format from cam_to_world")
                return

            self.world_detections = []

            for item in data:
                label = item.get("label", "")
                cen = item.get("centroid", {})
                bbox = item.get("bounding_box", {})

                wx, wy, wz = cen.get("x"), cen.get("y"), cen.get("z")
                if None in (wx, wy, wz):
                    continue

                # Remove PLACEMENT_THRESHOLDS filtering to match centroid_callback behavior
                # This ensures world_detections aligns with pick_set for proper label matching
                self.world_detections.append((label, np.array([wx, wy, wz]), bbox))

        except Exception as e:
            rospy.logerr(f"[VOSA] Error in centroid_to_world_callback: {e}")

    def infer_goal(self, ur_list):
        raw_confidences = [self.compute_confidence(ur, i) for i, ur in enumerate(ur_list)]
        soft_confidences = self.softmax(np.array(raw_confidences))
        self.confidences = soft_confidences.tolist()
        inferred_goal_index = int(np.argmax(soft_confidences))
        return inferred_goal_index, self.confidences[inferred_goal_index]

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def match_centroids_to_detections(self):
        matched_data = []

        if not self.world_detections:
            for i, centroid in enumerate(self.pick_set):
                conf = self.confidences[i] if i < len(self.confidences) else 0.0
                matched_data.append(("unknown", centroid, conf, {}))
            return matched_data



        # print(self.world_detections)
        det_labels, det_points, det_bboxes = zip(*self.world_detections)
        det_arr = np.vstack(det_points)

        for i, centroid in enumerate(self.pick_set):
            conf = self.confidences[i] if i < len(self.confidences) else 0.0
            distances = np.linalg.norm(det_arr - centroid, axis=1)
            closest_idx = int(np.argmin(distances))
            matched_data.append((det_labels[closest_idx], centroid, conf, det_bboxes[closest_idx]))

        return matched_data

    def main(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            # Y should interrupt everything (manual/wait/SA)
            if self.Y_pressed:
                rospy.loginfo("Resetting to home...")
                self.twist_command(0, 0, 0)
                rospy.sleep(0.5)
                self.reset.example_send_joint_angles(TOP_DOWN)
                rospy.sleep(6)
                self.reset.example_clear_faults()
                self.Y_pressed = False
                self.manual_mode = False
                self.wait_for_first_input = True
                rate.sleep()
                continue

            # Manual-only pass-through (after placing until Y)
            if self.manual_mode:
                self.twist_command(self.uh.linear_x * SPEED_CONTROL,
                                self.uh.linear_y * SPEED_CONTROL,
                                self.uh.linear_z * SPEED_CONTROL)
                rate.sleep()
                continue

            # First-input should be from human side (startup/after Y)
            if self.wait_for_first_input and self.input_magnitude(self.uh) <= 1e-6:
                self.twist_command(0.0, 0.0, 0.0)
                rate.sleep()
                continue
            else:
                # clear the gate on the first nonzero input
                if self.wait_for_first_input:
                    self.wait_for_first_input = False
            
            # pick mode when open; place mode when closed
            self.update_place_set_z()
            self.current_goal_set = self.place_set_mod if self.gripper_closed else self.pick_set

            # ALWAYS publish visualization data if we have goals (moved outside motion logic)
            if self.current_goal_set:
                # Compute ur & confidences over the active goal set
                ur_list = self.compute_ur_for_all_goals()
                goal_idx, confidence = self.infer_goal(ur_list)  # sets self.confidences for current set

                # Build & publish visualization message depending on gripper state
                msg = CentroidConfidenceArray()
                msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = "base_link"

                if self.gripper_closed:
                    # PLACE MODE: show placement locations with confidences
                    n = min(len(self.place_set), len(getattr(self, "confidences", [])))
                    for i in range(n):
                        px, py, pz = self.place_set_orig[i]
                        item = CentroidConfidence()
                        item.label = f"place_{i+1}"
                        item.centroid = Point(x=float(px), y=float(py), z=float(pz))
                        item.confidence = float(self.confidences[i])
                        item.x1 = item.y1 = item.x2 = item.y2 = 0
                        item.gripper_open = False
                        msg.items.append(item)
                else:
                    # PICK MODE: show pick locations with confidences
                    matched_data = self.match_centroids_to_detections()  # [(label, np3, conf, bbox), ...]
                    for label, centroid, conf, bbox in matched_data:
                        item = CentroidConfidence()
                        item.label = label
                        item.centroid = Point(x=centroid[0], y=centroid[1], z=centroid[2])
                        item.confidence = float(conf)
                        item.x1 = bbox.get('x1', 0)
                        item.y1 = bbox.get('y1', 0)
                        item.x2 = bbox.get('x2', 0)
                        item.y2 = bbox.get('y2', 0)
                        item.gripper_open = True
                        msg.items.append(item)

                # Publish visualization message
                self.centroid_conf_pub.publish(msg)

            # Handle movement commands
            if self.input_magnitude(self.uh) > 1e-6:
                self.last_nonzero_uh = self.uh
                # Human input detected - could add blending logic here if desired
                self.twist_command(self.uh.linear_x * SPEED_CONTROL,
                                self.uh.linear_y * SPEED_CONTROL,
                                self.uh.linear_z * SPEED_CONTROL)

            elif self.current_goal_set:
                # Check if robot is near the target goal and stop if close enough
                target_goal = self.current_goal_set[goal_idx]
                distance_to_goal = np.linalg.norm(self.ee_position[:3] - target_goal[:3])
                
                # Define stopping threshold (adjust as needed)
                stop_threshold = rospy.get_param("~placement_stop_threshold", 0.025)  # 2.5cm   default 

                if distance_to_goal < stop_threshold:
                    rospy.loginfo(f"[VOSA] Near placement location (dist={distance_to_goal:.3f}m), stopping robot")
                    self.twist_command(0.0, 0.0, 0.0)
                    
                    # Optional: switch to manual mode for fine positioning
                    if rospy.get_param("~manual_mode_on_arrival", False):
                        self.manual_mode = True
                        rospy.loginfo("[VOSA] Switched to manual mode for fine positioning")
                else:
                    # Autonomous assistance mode (reuse already computed values)
                    blended, alpha = self.blend_inputs(self.uh, ur_list[goal_idx], confidence)
                    cmd = np.array(blended, dtype=float)

                    if self.task == "shelving":
                        # Match Assistive behavior: big Z while placing, smaller while picking
                        z_mult_place = rospy.get_param("~z_place_mult", 15.0)
                        z_mult_pick  = rospy.get_param("~z_pick_mult",  3.0)
                        cmd[2] *= (z_mult_place if self.gripper_closed else z_mult_pick)
                    if self.task == "sorting":
                        # Match Assistive behavior: big Z while placing, smaller while picking
                        z_mult_place = rospy.get_param("~z_place_mult", 15.0)
                        z_mult_pick  = rospy.get_param("~z_pick_mult",  1.0)
                        cmd[2] *= (z_mult_place if self.gripper_closed else z_mult_pick)

                        # Normalize XYZ to length <= 1 (Assistive does this before final scale)
                        cmd[1] *= 5
                        xyz_norm = np.linalg.norm(cmd[0:3])
                        if xyz_norm > 1.0:
                            cmd[0:3] /= xyz_norm

                        # Final global scale (Assistive uses 0.5; you can use SPEED_CONTROL instead)
                        final_scale = rospy.get_param("~xyz_final_scale", 0.5)
                        cmd[0:3] *= final_scale
                    else:
                        # Non-shelving: keep previous behavior
                        cmd = cmd * SPEED_CONTROL

                    # Safety clamp
                    cmd[0:3] = np.clip(cmd[0:3], -1.0, 1.0)

                    rospy.loginfo_throttle(0.5,
                        f"[VOSA] send XYZ={cmd[0]:.3f},{cmd[1]:.3f},{cmd[2]:.3f} (placing={self.gripper_closed}, dist={distance_to_goal:.3f})")

                    self.twist_command(cmd[0], cmd[1], cmd[2])

            else:
                # no goals: pass-through teleop
                pass
                self.twist_command(self.uh.linear_x * SPEED_CONTROL,
                                self.uh.linear_y * SPEED_CONTROL,
                                self.uh.linear_z * SPEED_CONTROL)

            rate.sleep()

        rospy.logwarn("Before Shutdown Attempt to Save Trial Data to User Folder")
        self.on_shutdown()

if __name__ == "__main__":
    rospy.init_node("vosa_teleop")
    node = VOSATeleoperation()
    node.main()
