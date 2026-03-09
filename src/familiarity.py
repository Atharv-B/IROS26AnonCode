#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from sag import SAGTeleoperation
from constants import STOP_SCAN_THRESHOLD, PLACEMENT_THRESHOLDS, HOME, SPEED_CONTROL
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point
from trust_and_transparency.msg import CentroidConfidence, CentroidConfidenceArray
from std_msgs.msg import String
from cam_to_world import CameraToWorld
import json

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
        self.centroid_conf_pub = rospy.Publisher('/goal_confidence_centroids', CentroidConfidenceArray, queue_size=10)

        # Override task param
        self.task = "sorting"  # Default to "sorting" task for VOSA

        rospy.Subscriber("/clusters", PointCloud, self.centroid_callback)
        rospy.Subscriber(
            '/detected_objects/dict_label_centroid',
            String,
            self.dict_label_centroid_callback)
        
        rospy.Subscriber('/centroid_to_world', String, self.centroid_to_world_callback)
        rospy.loginfo("[VOSA] Initialized with dynamic pick set subscription.")

    def centroid_to_world_callback(self, msg: String):
        """
        msg.data is a JSON list of detections:
        [{"label": "...", "centroid": {x,y,z}, "bounding_box": {...}, "id": ...}, ...]
        """
        try:
            data = json.loads(msg.data)
            
            # Ensure data is a list
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
                    
                # Only keep detections above threshold
                if wy > PLACEMENT_THRESHOLDS[self.task]:
                    self.world_detections.append((
                        label, 
                        np.array([wx, wy, wz]),
                        bbox
                    ))
                    
            # rospy.loginfo(f"[VOSA] Updated world_detections with {len(self.world_detections)} items")
            
        except Exception as e:
            rospy.logerr(f"[VOSA] Error in centroid_to_world_callback: {e}")


    def dict_label_centroid_callback(self, msg: String):
        """
        Handle both single dict and list formats from YOLO
        msg.data can be:
        - Single dict: {"label": "...", "centroid": {x,y,z}}
        - List: [{"label": "...", "centroid": {x,y,z}}, ...]
        """
        try:
            data = json.loads(msg.data)
            
            # Handle list format (multiple detections)
            if isinstance(data, list):
                for item in data:
                    self.process_single_dict_detection(item)
            
            # Handle single dict format
            elif isinstance(data, dict):
                self.process_single_dict_detection(data)
            else:
                rospy.logwarn(f"[VOSA] Unexpected data format in dict_label_centroid_callback: {type(data)}")
                
        except Exception as e:
            rospy.logerr(f"[VOSA] Error in dict_label_centroid_callback: {e}")

    def build_place_conf_msg(self):
        """Placement locations (no bbox) with confidences (for placing)."""
        msg = CentroidConfidenceArray()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"

        # self.confidences were produced by infer_goal() over current_goal_set (== place_set here)
        n = min(len(self.place_set), len(getattr(self, "confidences", [])))
        for i in range(n):
            px, py, pz = self.place_set[i]
            item = CentroidConfidence()
            item.label = f"place_{i+1}"
            item.centroid = Point(x=float(px), y=float(py), z=float(pz))
            item.confidence = float(self.confidences[i])
            item.x1 = item.y1 = item.x2 = item.y2 = 0
            msg.items.append(item)
        return msg

    def process_single_dict_detection(self, data):
        """
        Process a single detection dict from dict_label_centroid topic
        msg.data is JSON: { "label": "...", "centroid": {x,y,z} }
        CameraToWorld has already turned it into robot-frame coords.
        Here we parse it and buffer (label, centroid_array).
        """
        try:
            label = data.get("label", "")
            cen = data.get("centroid", {})
            wx, wy, wz = cen.get("x"), cen.get("y"), cen.get("z")
            
            if None in (wx, wy, wz):
                rospy.logwarn(f"[VOSA] Incomplete centroid JSON for {label}")
                return

            # Only keep those above your y-threshold
            if wy > PLACEMENT_THRESHOLDS[self.task]:
                self.raw_objects.append((label, np.array([wx, wy, wz])))
                rospy.logdebug(f"[VOSA] Added raw object: {label} at ({wx:.3f}, {wy:.3f}, {wz:.3f})")
            else:
                rospy.logdebug(f"[VOSA] Filtered out {label} (y={wy:.3f} < threshold={PLACEMENT_THRESHOLDS[self.task]})")
                
        except Exception as e:
            rospy.logerr(f"[VOSA] Error processing single detection: {e}")

    

    def centroid_callback(self, msg):
        if self.ee_position[0] > STOP_SCAN_THRESHOLD:
            return
        Y_LIMS = [0.20, 0.60]

        detected = []
        for pt in msg.points:
            centroid = np.array([pt.x + 0.025, pt.y - 0.01, pt.z])
            
            # if pt.y > PLACEMENT_THRESHOLDS[self.task]:
            #     detected.append(centroid)
            if Y_LIMS[0] <= pt.y <= Y_LIMS[1]:
                detected.append(centroid)

        # print("We detected centroids:", detected)
        if not self.intermediate_position_reached and len(detected) == 0:
            return

        self.intermediate_position_reached = True

        self.pick_set = detected

    #defined inferred goal
    def infer_goal(self, ur_list):
        # Compute raw confidence for each goal using inherited method
        raw_confidences = [self.compute_confidence(ur, i) for i, ur in enumerate(ur_list)]
    
        # Normalize with softmax for probabilistic interpretation
        soft_confidences = self.softmax(np.array(raw_confidences))
    
        # Store the full confidence list for later publishing
        self.confidences = soft_confidences.tolist()
    
        # Choose the goal with highest confidence
        inferred_goal_index = int(np.argmax(soft_confidences))
    
        return inferred_goal_index, self.confidences[inferred_goal_index]


    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def match_centroids_to_detections(self):
        """Match pick_set centroids to world_detections and return matched data with bboxes"""
        matched_data = []
        
        if not self.world_detections:
            # No detections available, return unknown labels with empty bboxes
            for i, centroid in enumerate(self.pick_set):
                conf = self.confidences[i] if i < len(self.confidences) else 0.0
                matched_data.append(("unknown", centroid, conf, {}))
            return matched_data
        
        # Extract detection data
        det_labels, det_points, det_bboxes = zip(*self.world_detections)
        det_arr = np.vstack(det_points)
        
        # Match each centroid to closest detection
        for i, centroid in enumerate(self.pick_set):
            conf = self.confidences[i] if i < len(self.confidences) else 0.0
            
            # Find closest detection
            distances = np.linalg.norm(det_arr - centroid, axis=1)
            closest_idx = int(np.argmin(distances))
            
            matched_data.append((
                det_labels[closest_idx],
                centroid, 
                conf,
                det_bboxes[closest_idx]
            ))
        
        return matched_data

    def on_shutdown(self):
        super().on_shutdown()
        pass
    
    def main(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            # Y should interrupt everything (manual/wait/SA)
            if self.Y_pressed:
                rospy.loginfo("Resetting to home...")
                self.twist_command(-0.45, 0, 0.45)
                rospy.sleep(3)
                self.twist_command(0, 0, 0)
                rospy.sleep(0.5)
                self.reset.example_send_joint_angles(HOME)
                rospy.sleep(10.0)
                self.reset.example_clear_faults()

                self.manual_mode = False
                self.wait_for_first_input = True

                self.Y_pressed = False
                rospy.sleep(0.2)
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
                        # item.centroid = Point(x=0.0, y=0.0, z=0.0)  # No centroid for place locations
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
                # Autonomous assistance mode (reuse already computed values)
                blended, alpha = self.blend_inputs(self.uh, ur_list[goal_idx], confidence)
                cmd = np.array(blended, dtype=float)

                if self.task == "shelving":
                    # Match Assistive behavior: big Z while placing, smaller while picking
                    z_mult_place = rospy.get_param("~z_place_mult", 15.0)
                    z_mult_pick  = rospy.get_param("~z_pick_mult",  3.0)
                    cmd[2] *= (z_mult_place if self.gripper_closed else z_mult_pick)

                    # Normalize XYZ to length <= 1 (Assistive does this before final scale)
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
                    f"[VOSA] send XYZ={cmd[0]:.3f},{cmd[1]:.3f},{cmd[2]:.3f} (placing={self.gripper_closed})")

                self.twist_command(cmd[0], cmd[1], cmd[2])

            else:
                # no goals: pass-through teleop
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
 