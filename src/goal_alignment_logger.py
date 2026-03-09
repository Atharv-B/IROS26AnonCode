#!/usr/bin/env python3
# filepath: /home/kinovaresearch/catkin_workspace/src/trust_and_transparency/src/goal_alignment_logger.py

import rospy
import json
import os
import time
import csv
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Joy
from trust_and_transparency.msg import CentroidConfidenceArray
from kortex_driver.msg import BaseCyclic_Feedback
from constants import ORACLE_GOAL_SET, STOP_SCAN_THRESHOLD
from user_study.user_study import UserStudyExperiment

class GoalAlignmentLogger:
    def __init__(self):
        self.task = rospy.get_param("~task", "shelving")
        self.treatment = rospy.get_param("~treatment", "A")  
        
        # Data storage
        self.data_log = []
        self.current_robot_goal = None
        self.current_ground_truth_goal = None
        self.gripper_closed = False
        self.world_detections = []
        
        # Robot state data
        self.tool_pose = None
        self.tool_twist = None
        self.joint_angles = []
        self.joint_velocities = []
        self.joint_torques = []
        
        # Joy/controller data
        self.joy_axes = []
        self.joy_buttons = []
        
        # Track VOSA system state
        self.vosa_waiting_for_input = True  # Starts in waiting state
        self.vosa_y_pressed = False         # Track Y button state
        self.place_goal_counter = 0 
        self.pick_goal_counter = 0
        
        # Track input magnitude 
        self.current_input_magnitude = None

        # ------------------ [ADDED] Shelving sequence with HOME steps ------------------
        # Target 8-step flow:
        # 1 (open) -> 2 pick blue -> 3 place top_middle -> 4 HOME (Y) ->
        # 5 pick mustard -> 6 place bottom_left -> 7 HOME (Y) -> 8 end
        self.sequence = []
        self.seq_step = 0
        self.current_gt_mode = None  # 'pick' | 'place' | 'home'

        if self.task == 'shelving':
            shelving_cfg = ORACLE_GOAL_SET.get('shelving', {})
            obj = shelving_cfg.get('object_locations', [])
            place = shelving_cfg.get('placement_locations', [])
            TOP_LEFT_IDX = 3
            BOTTOM_RIGHT_IDX = 0
            if len(obj) >= 2 and len(place) > max(TOP_LEFT_IDX, BOTTOM_RIGHT_IDX):
                self.sequence = [
                    {'mode': 'pick',  'label': 'blue bottle', 'position': obj[0]},
                    {'mode': 'place', 'label': 'top_middle', 'position': place[TOP_LEFT_IDX]},
                    {'mode': 'home',  'label': 'home', 'position': None},  # advance on Y
                    {'mode': 'pick',  'label': 'mustard bottle', 'position': obj[1]},
                    {'mode': 'place', 'label': 'bottom_left', 'position': place[BOTTOM_RIGHT_IDX]},
                    {'mode': 'home',  'label': 'home', 'position': None},  # advance on Y
                ]
            else:
                rospy.logerr("[GoalAlignmentLogger] Shelving sequence not built: oracle points missing")
        if self.task == 'sorting':
            sorting_cfg = ORACLE_GOAL_SET.get('sorting', {})
            obj = sorting_cfg.get('object_locations', [])
            place = sorting_cfg.get('placement_locations', [])
            LEFT_BIN_IDX = 1
            RIGHT_BIN_IDX = 0
            CENTER_BIN_IDX = 2
            BLUE_BOTTLE_IDX = 1
            PASTA_BOX_IDX = 0
            SODA_CAN_IDX = 2
            if len(obj) >= 3 and len(place) > max(LEFT_BIN_IDX, RIGHT_BIN_IDX, CENTER_BIN_IDX):
                self.sequence = [
                    {'mode': 'pick',  'label': 'blue bottle', 'position': obj[PASTA_BOX_IDX]},
                    {'mode': 'place', 'label': 'left_bin', 'position': place[LEFT_BIN_IDX]},
                    {'mode': 'home',  'label': 'home', 'position': None},  # advance on Y
                    {'mode': 'pick',  'label': 'paper cup', 'position': obj[BLUE_BOTTLE_IDX]},
                    {'mode': 'place', 'label': 'right_bin', 'position': place[RIGHT_BIN_IDX]},
                    {'mode': 'home',  'label': 'home', 'position': None},  # advance on Y
                    {'mode': 'pick',  'label': 'soda can', 'position': obj[SODA_CAN_IDX]},
                    {'mode': 'place', 'label': 'center_bin', 'position': place[CENTER_BIN_IDX]},
                    {'mode': 'home',  'label': 'home', 'position': None},  # advance on Y
                ]
        # --------------------------------------------------------------------------------

        # CSV fieldnames matching robot_logger naming convention
        self.csv_fieldnames = [
            'timestamp',
            'gripper_closed',
            'robot_goal_x',
            'robot_goal_y', 
            'robot_goal_z',
            'robot_goal_confidence',
            'robot_goal_label',
            'ground_truth_goal_x',
            'ground_truth_goal_y',
            'ground_truth_goal_z',
            'ground_truth_goal_label',
            'goals_aligned',
            'input_magnitude',
            'system_state',
            'tool_pose_x',
            'tool_pose_y',
            'tool_pose_z',
            'tool_pose_theta_x',
            'tool_pose_theta_y',
            'tool_pose_theta_z',
            'tool_twist_linear_x',
            'tool_twist_linear_y',
            'tool_twist_linear_z',
            'tool_twist_angular_x',
            'tool_twist_angular_y',
            'tool_twist_angular_z'
        ]

        # ------------------ [ADDED] audit fields for the sequence ------------------
        if 'gt_mode' not in self.csv_fieldnames:
            self.csv_fieldnames += ['gt_mode', 'seq_step']
        # -------------------------------------------------------------------------

        # Add joint columns exactly as in robot_logger
        for i in range(7):
            self.csv_fieldnames.append(f'joint_{i}_angle')
        for i in range(7):
            self.csv_fieldnames.append(f'joint_{i}_velocity')
        for i in range(7):
            self.csv_fieldnames.append(f'joint_{i}_torque')
            
        # Joy columns will be added dynamically
        self.joy_axes_columns = []
        self.joy_buttons_columns = []

        # ------------------ [ADDED] robust trigger/Y edge tracking ------------------
        self.trigger_threshold = rospy.get_param("~trigger_threshold", -0.8)
        self.lt_was_pressed = False  # left trigger (close)
        self.rt_was_pressed = False  # right trigger (open)
        self.y_was_pressed  = False  # Y button edge detection
        # ---------------------------------------------------------------------------
        
        # Subscribers
        rospy.Subscriber('/goal_confidence_centroids', CentroidConfidenceArray, self.robot_goal_callback)
        rospy.Subscriber('/centroid_to_world', String, self.world_detections_callback)
        rospy.Subscriber("/my_gen3/base_feedback", BaseCyclic_Feedback, self.base_feedback_callback)
        rospy.Subscriber("/joy", Joy, self.controller_callback)  # Add joystick subscriber

        # Timer for data logging every 1000ms
        self.log_timer = rospy.Timer(rospy.Duration(0.1), self.log_data)
        
        # Initialize ground truth after a short delay to allow subscribers to connect
        rospy.Timer(rospy.Duration(2.0), self.initialize_ground_truth, oneshot=True)
        
        rospy.loginfo(f"[GoalAlignmentLogger] Initialized for task: {self.task}")

    # def input_magnitude(self, linear_x, linear_y, linear_z):
    #     """
    #     Calculate input magnitude - Euclidean norm of linear velocities
    #     Same as: np.linalg.norm(np.array([uh.linear_x, uh.linear_y, uh.linear_z]))
    #     """
    #     return np.linalg.norm(np.array([linear_x, linear_y, linear_z]))

    def convert_joy_to_linear_velocities(self, axes):
        """
        Convert joystick axes to linear velocities (same mapping as your teleop)
        Adjust these mappings based on your controller configuration
        """
        if len(axes) < 5:
            print("Insufficient axes data in joystick message")
            return 0.0, 0.0, 0.0
            
        # Xbox controller mapping
        linear_x = axes[1]  # Left stick, X (forward/backward)
        linear_y = axes[0]  # Left stick, Y (left/right)  
        linear_z = axes[4]  # Right stick, Z (up/down)
        
        return linear_x, linear_y, linear_z

    # def calculate_input_magnitude(self, controller_position):
    #     """Calculate input magnitude"""
    #     linear_x, linear_y, linear_z = self.convert_joy_to_linear_velocities(controller_position)
    #     return self.input_magnitude(linear_x, linear_y, linear_z)

    def calculate_input_magnitude(self, controller_position):
        """
        Calculate input magnitude from joystick axes.

        """
        if len(controller_position) < 5:
            rospy.logwarn_throttle(5, "[GoalAlignmentLogger] Insufficient axes data")
            return 0.0
            
        # Xbox controller mapping (same as your teleop)
        linear_x = float(controller_position[1])  # Left stick Y (forward/backward)
        linear_y = float(controller_position[0])  # Left stick X (left/right)  
        linear_z = float(controller_position[4])  # Right stick Y (up/down)
        
        # Calculate Euclidean norm directly from raw values
        print("Linear Velocities:", linear_x, linear_y, linear_z)
        return float(np.linalg.norm([linear_x, linear_y, linear_z]))

    def initialize_ground_truth(self, event):
        """Initialize ground truth goal on startup"""
        self.update_ground_truth_goal()
        
    def robot_goal_callback(self, msg):
        """Extract the robot's goal with highest confidence"""
        if not msg.items:
            self.current_robot_goal = None
            return
            
        # Find item with highest confidence
        max_conf_item = max(msg.items, key=lambda x: x.confidence)
        
        self.current_robot_goal = {
            'position': [max_conf_item.centroid.x, max_conf_item.centroid.y, max_conf_item.centroid.z],
            'confidence': max_conf_item.confidence,
            'label': max_conf_item.label,
            'is_pick_mode': max_conf_item.gripper_open 
        }
        
    def world_detections_callback(self, msg):
        """Update world detections for ground truth calculation"""
        try:
            data = json.loads(msg.data)
            if isinstance(data, list):
                self.world_detections = []
                for item in data:
                    label = item.get("label", "")
                    cen = item.get("centroid", {})
                    wx, wy, wz = cen.get("x"), cen.get("y"), cen.get("z")
                    
                    if None not in (wx, wy, wz):
                        self.world_detections.append({
                            'label': label,
                            'position': [wx, wy, wz]
                        })
            self.update_ground_truth_goal()

        except Exception as e:
            rospy.logerr(f"[GoalAlignmentLogger] Error parsing world detections: {e}")

    def controller_callback(self, data):
        """Track gripper state from joystick commands (same logic as direct_teleop.py)"""
        controller_position = data.axes
        controller_buttons = data.buttons

        # Store joy data for logging
        self.joy_axes = list(controller_position)
        self.joy_buttons = list(controller_buttons)
        
        # Update joy column names if needed (first time we get data)
        if not self.joy_axes_columns and self.joy_axes:
            self.joy_axes_columns = [f"axis_{i}" for i in range(len(self.joy_axes))]
            self.joy_buttons_columns = [f"button_{i}" for i in range(len(self.joy_buttons))]
            # Add these columns to fieldnames if not already present
            for col in self.joy_axes_columns + self.joy_buttons_columns:
                if col not in self.csv_fieldnames:
                    self.csv_fieldnames.append(col)

        self.current_input_magnitude = self.calculate_input_magnitude(controller_position)
        rospy.loginfo_throttle(1, f"[GoalAlignmentLogger] Current input magnitude: {self.current_input_magnitude:.3f}")
        
        # Ensure we have enough axes to check triggers and buttons
        if len(controller_position) < 6 or len(controller_buttons) < 4:
            return
        
        # ---------------- Y button: return to home & reset waiting state (original behavior) ----------------
        if controller_buttons[3] == 1 and not self.vosa_y_pressed:
            self.vosa_y_pressed = True
            self.vosa_waiting_for_input = True
            self.current_robot_goal = None

            # Also advance the sequence if current expected step is 'home'
            if self.sequence and self.seq_step < len(self.sequence):
                if self.sequence[self.seq_step]['mode'] == 'home':
                    self.seq_step += 1
            self.update_ground_truth_goal()

        elif controller_buttons[3] == 0:
            self.vosa_y_pressed = False
        # ----------------------------------------------------------------------------------------------

        # Check for any human input (movement commands) using SAG threshold
        if self.vosa_waiting_for_input and self.current_input_magnitude > 1e-6:
            self.vosa_waiting_for_input = False
            
        # ---------------- [ADDED] trigger edge detection with threshold ----------------
        lt_pressed = (controller_position[2] < self.trigger_threshold)   # Left trigger => CLOSE
        rt_pressed = (controller_position[5] < self.trigger_threshold)   # Right trigger => OPEN

        # LEFT TRIGGER rising edge: close gripper (confirm PICK)
        if lt_pressed and not self.lt_was_pressed and not self.gripper_closed:
            self.gripper_closed = True

            # advance sequence ONLY if expected step is 'pick'
            if self.sequence and self.seq_step < len(self.sequence):
                if self.sequence[self.seq_step]['mode'] == 'pick':
                    self.seq_step += 1

            self.update_ground_truth_goal()

        # RIGHT TRIGGER rising edge: open gripper (confirm PLACE)
        if rt_pressed and not self.rt_was_pressed and self.gripper_closed:
            self.gripper_closed = False

            # advance sequence ONLY if expected step is 'place'
            if self.sequence and self.seq_step < len(self.sequence):
                if self.sequence[self.seq_step]['mode'] == 'place':
                    self.seq_step += 1

            # keep your original counters behavior
            self.place_goal_counter += 1
            self.pick_goal_counter += 1
            self.update_ground_truth_goal()

        self.lt_was_pressed = lt_pressed
        self.rt_was_pressed = rt_pressed
        # --------------------------------------------------------------------------------

    def base_feedback_callback(self, feedback):
        """Store robot state data with exact same field names as robot_logger"""
        # Tool pose (position and orientation)
        self.tool_pose = {
            'x': feedback.base.tool_pose_x,
            'y': feedback.base.tool_pose_y,
            'z': feedback.base.tool_pose_z,
            'theta_x': feedback.base.tool_pose_theta_x,
            'theta_y': feedback.base.tool_pose_theta_y,
            'theta_z': feedback.base.tool_pose_theta_z
        }
        
        # Tool twist (linear and angular velocities)
        self.tool_twist = {
            'linear_x': feedback.base.tool_twist_linear_x,
            'linear_y': feedback.base.tool_twist_linear_y,
            'linear_z': feedback.base.tool_twist_linear_z,
            'angular_x': feedback.base.tool_twist_angular_x,
            'angular_y': feedback.base.tool_twist_angular_y,
            'angular_z': feedback.base.tool_twist_angular_z
        }
        
        # Joint states
        self.joint_angles = [feedback.actuators[i].position for i in range(len(feedback.actuators))]
        self.joint_velocities = [feedback.actuators[i].velocity for i in range(len(feedback.actuators))]
        self.joint_torques = [feedback.actuators[i].torque for i in range(len(feedback.actuators))]

    def get_system_state(self):
        """Determine current system state for logging"""
        if self.vosa_waiting_for_input:
            return "waiting_for_input"
        elif self.current_robot_goal is None:
            return "active_no_goals"
        else:
            return "active_with_goals"

    def update_ground_truth_goal(self):
        """Update ground truth goal based on current mode and task"""
        # During waiting periods, set ground truth to None
        if self.vosa_waiting_for_input:
            self.current_ground_truth_goal = None
            self.current_gt_mode = None
            rospy.loginfo_throttle(2, "[GoalAlignmentLogger] Waiting for input - ground truth set to None")
            return

        # ------------------ [ADDED] sequence-driven GT for shelving (incl. HOME) ------------------
        if self.task == 'shelving' and self.sequence:
            if self.seq_step >= len(self.sequence):
                self.current_ground_truth_goal = None
                self.current_gt_mode = None
                rospy.loginfo_throttle(5, "[GoalAlignmentLogger] Shelving sequence complete")
                return

            step = self.sequence[self.seq_step]
            self.current_gt_mode = step['mode']

            if step['mode'] == 'home':
                # During HOME: no spatial GT (blank GT fields)
                self.current_ground_truth_goal = None
                rospy.loginfo_throttle(2, "[GoalAlignmentLogger] [SEQ] home step: press Y when homed")
                return

            # pick/place → set spatial GT
            self.current_ground_truth_goal = {
                'position': step['position'],
                'label': step['label']
            }
            rospy.loginfo_throttle(2, f"[GoalAlignmentLogger] [SEQ] step {self.seq_step+1}/{len(self.sequence)}: "
                                       f"{step['mode']} -> {step['label']}")
            return
        # -------------------------------------------------------------------------------------------
            
        # Use gripper state to determine mode (fallback for other tasks)
        is_pick_mode = not self.gripper_closed
        
        rospy.loginfo_throttle(2, f"[GoalAlignmentLogger] Mode: {'PICK' if is_pick_mode else 'PLACE'} (gripper_closed: {self.gripper_closed})")
        if is_pick_mode:
            object_locations = ORACLE_GOAL_SET[self.task]['object_locations']
            if object_locations:
                if self.task == "shelving":
                    label = ["blue bottle", "mustard bottle"]
                    idx = min(self.pick_goal_counter, len(object_locations) - 1)
                    label_idx = min(self.pick_goal_counter, len(label) - 1)
                    self.current_ground_truth_goal = {
                        'position': object_locations[idx],
                        'label': label[label_idx]
                    }
                    rospy.loginfo_throttle(2, f"[GoalAlignmentLogger] Pick mode - no detections, using fallback: {object_locations[idx]}")
                
                elif self.task == "sorting":
                    label = ["pasta box", "blue bottle", "soda can"]
                    idx = min(self.pick_goal_counter, len(object_locations) - 1)
                    label_idx = min(self.pick_goal_counter, len(label) - 1)
                    target_object = object_locations[idx]
                    self.current_ground_truth_goal = {
                        'position': target_object,
                        'label': label[label_idx]
                    }
                else:
                    self.current_ground_truth_goal = None
                    rospy.logwarn_throttle(5, "[GoalAlignmentLogger] Pick mode but no objects available")
        else:
            # PLACE MODE
            placement_locations = ORACLE_GOAL_SET[self.task]['placement_locations']
            if placement_locations:
                # Define the placement order for shelving task
                if self.task == "shelving":
                    # Match the sequence order: TOP_LEFT (0), then BOTTOM_RIGHT (3)
                    placement_order = [0, 3]  # TOP_LEFT, BOTTOM_RIGHT
                    placement_labels = ['top_left', 'bottom_right']
                    
                    order_idx = min(self.place_goal_counter, len(placement_order) - 1)
                    actual_idx = placement_order[order_idx]
                    
                    if actual_idx < len(placement_locations):
                        target_placement = placement_locations[actual_idx]
                        self.current_ground_truth_goal = {
                            'position': target_placement,
                            'label': placement_labels[order_idx]
                        }
                    else:
                        self.current_ground_truth_goal = None
                        rospy.logwarn_throttle(5, f"[GoalAlignmentLogger] Invalid placement index: {actual_idx}")
                else:
                    # For other tasks, use sequential placement
                    idx = min(self.place_goal_counter, len(placement_locations) - 1)
                    target_placement = placement_locations[idx]
                    self.current_ground_truth_goal = {
                        'position': target_placement,
                        'label': f'place_{idx + 1}'
                    }
                
                if self.place_goal_counter >= len(placement_locations):
                    rospy.logwarn_throttle(5, f"[GoalAlignmentLogger] Place counter ({self.place_goal_counter}) exceeds available placements")
            else:
                self.current_ground_truth_goal = None
                rospy.logwarn_throttle(5, "[GoalAlignmentLogger] Place mode but no placement locations available")
            # # PLACE MODE
            # placement_locations = ORACLE_GOAL_SET[self.task]['placement_locations']
            # if placement_locations:
            #     idx = min(self.place_goal_counter, len(placement_locations) - 1)
            #     target_placement = placement_locations[idx]
            #     self.current_ground_truth_goal = {
            #         'position': target_placement,
            #         'label': f'place_{idx + 1}'
            #     }
            #     if self.place_goal_counter >= len(placement_locations):
            #         rospy.logwarn_throttle(5, f"[GoalAlignmentLogger] Place counter ({self.place_goal_counter}) exceeds available placements, using last one: {target_placement}")
            # else:
            #     self.current_ground_truth_goal = None
            #     rospy.logwarn_throttle(5, "[GoalAlignmentLogger] Place mode but no placement locations available")

                
    def calculate_alignment(self):
        """Calculate if robot goal and ground truth goal are aligned"""
        if self.current_robot_goal is None or self.current_ground_truth_goal is None:
            return 0
            
        robot_pos = np.array(self.current_robot_goal['position'])
        gt_pos = np.array(self.current_ground_truth_goal['position'])
        
        distance = np.linalg.norm(robot_pos - gt_pos)
        alignment_threshold = 0.07  # 7cm
        
        return 1 if distance <= alignment_threshold else 0
        
    def log_data(self, event):
        """Log data every 1000ms with robot_logger compatible field names"""
        timestamp = rospy.get_time()
        
        # Get current system state
        current_state = self.get_system_state()
        
        # Skip logging during "active_no_goals" state
        if current_state == "active_no_goals":
            return
        
        # Prepare data entry for CSV
        data_entry = {
            'timestamp': timestamp,
            'gripper_closed': self.gripper_closed,
            'goals_aligned': self.calculate_alignment(),
            'system_state': current_state,
            'input_magnitude': self.current_input_magnitude  
        }

        # ------------------ [ADDED] sequence audit fields ------------------
        if self.current_ground_truth_goal and not self.vosa_waiting_for_input:
            data_entry['gt_mode'] = self.current_gt_mode
        else:
            data_entry['gt_mode'] = self.current_gt_mode if self.current_gt_mode == 'home' else None
        data_entry['seq_step'] = self.seq_step
        # -------------------------------------------------------------------
        
        # Add robot goal data
        if self.current_robot_goal and not self.vosa_waiting_for_input:
            if self.current_gt_mode == 'home':
                # During HOME steps, blank out robot goal fields
                data_entry.update({
                    'robot_goal_x': None,
                    'robot_goal_y': None,
                    'robot_goal_z': None,
                    'robot_goal_confidence': None,
                    'robot_goal_label': None
                })
            else:
                data_entry.update({
                    'robot_goal_x': self.current_robot_goal['position'][0],
                    'robot_goal_y': self.current_robot_goal['position'][1],
                    'robot_goal_z': self.current_robot_goal['position'][2],
                    'robot_goal_confidence': self.current_robot_goal['confidence'],
                'robot_goal_label': self.current_robot_goal['label']
            })
        else:
            data_entry.update({
                'robot_goal_x': None,
                'robot_goal_y': None,
                'robot_goal_z': None,
                'robot_goal_confidence': None,
                'robot_goal_label': None
            })
            
        # Add ground truth goal data
        if self.current_ground_truth_goal and not self.vosa_waiting_for_input:
            data_entry.update({
                'ground_truth_goal_x': self.current_ground_truth_goal['position'][0],
                'ground_truth_goal_y': self.current_ground_truth_goal['position'][1],
                'ground_truth_goal_z': self.current_ground_truth_goal['position'][2],
                'ground_truth_goal_label': self.current_ground_truth_goal['label']
            })
        else:
            data_entry.update({
                'ground_truth_goal_x': None,
                'ground_truth_goal_y': None,
                'ground_truth_goal_z': None,
                'ground_truth_goal_label': None
            })
            
        # Add tool pose data (matching robot_logger field names)
        if self.tool_pose:
            data_entry.update({
                'tool_pose_x': self.tool_pose['x'],
                'tool_pose_y': self.tool_pose['y'],
                'tool_pose_z': self.tool_pose['z'],
                'tool_pose_theta_x': self.tool_pose['theta_x'],
                'tool_pose_theta_y': self.tool_pose['theta_y'],
                'tool_pose_theta_z': self.tool_pose['theta_z']
            })
        else:
            data_entry.update({
                'tool_pose_x': None,
                'tool_pose_y': None,
                'tool_pose_z': None,
                'tool_pose_theta_x': None,
                'tool_pose_theta_y': None,
                'tool_pose_theta_z': None
            })
        
        # Add tool twist data (matching robot_logger field names)
        if self.tool_twist:
            data_entry.update({
                'tool_twist_linear_x': self.tool_twist['linear_x'],
                'tool_twist_linear_y': self.tool_twist['linear_y'],
                'tool_twist_linear_z': self.tool_twist['linear_z'],
                'tool_twist_angular_x': self.tool_twist['angular_x'],
                'tool_twist_angular_y': self.tool_twist['angular_y'],
                'tool_twist_angular_z': self.tool_twist['angular_z']
            })
        else:
            data_entry.update({
                'tool_twist_linear_x': None,
                'tool_twist_linear_y': None,
                'tool_twist_linear_z': None,
                'tool_twist_angular_x': None,
                'tool_twist_angular_y': None,
                'tool_twist_angular_z': None
            })
        
        # Add joint data (matching robot_logger field names)
        for i in range(7):
            if i < len(self.joint_angles):
                data_entry[f'joint_{i}_angle'] = self.joint_angles[i]
            else:
                data_entry[f'joint_{i}_angle'] = None
                
        for i in range(7):
            if i < len(self.joint_velocities):
                data_entry[f'joint_{i}_velocity'] = self.joint_velocities[i]
            else:
                data_entry[f'joint_{i}_velocity'] = None
                
        for i in range(7):
            if i < len(self.joint_torques):
                data_entry[f'joint_{i}_torque'] = self.joint_torques[i]
            else:
                data_entry[f'joint_{i}_torque'] = None
        
        # Add joy data (matching joy_logger field names)
        for i, col in enumerate(self.joy_axes_columns):
            if i < len(self.joy_axes):
                data_entry[col] = self.joy_axes[i]
            else:
                data_entry[col] = None
                
        for i, col in enumerate(self.joy_buttons_columns):
            if i < len(self.joy_buttons):
                data_entry[col] = self.joy_buttons[i]
            else:
                data_entry[col] = None
        
        self.data_log.append(data_entry)
        
        # Optional: Log status for debugging
        if len(self.data_log) % 5 == 0:  # Every 5 seconds
            input_mag = self.current_input_magnitude if self.current_input_magnitude is not None else 0.0
            rospy.loginfo(f"[GoalAlignmentLogger] State: {data_entry['system_state']}, "
                          f"Goals aligned: {data_entry['goals_aligned']}, "
                          f"Gripper closed: {self.gripper_closed}, "
                          f"Input magnitude: {input_mag:.3f}, "
                          f"Data points: {len(self.data_log)}")
                
    def save_data(self):
        """Save logged data to CSV file"""
        if not self.data_log:
            rospy.logwarn("[GoalAlignmentLogger] No data to save")
            return
            
        # Get user directory
        exp = UserStudyExperiment()
        user_dir = exp.get_user_dir(self.task, self.treatment)
        
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
            
        # Save to CSV
        filename = os.path.join(user_dir, "goal_alignment_data.csv")
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.csv_fieldnames)
            writer.writeheader()
            
            for row in self.data_log:
                # Ensure all fieldnames are present in row
                complete_row = {field: row.get(field, None) for field in self.csv_fieldnames}
                writer.writerow(complete_row)
                
        rospy.loginfo(f"[GoalAlignmentLogger] Saved {len(self.data_log)} data points to: {filename}")
            
    def save_summary(self, user_dir):
        """Save summary statistics to JSON file"""
        if not self.data_log:
            return
            
        # Calculate summary statistics
        total_logs = len(self.data_log)
        aligned_count = sum(1 for entry in self.data_log if entry.get('goals_aligned') == 1)
        alignment_percentage = (aligned_count / total_logs * 100) if total_logs > 0 else 0
        
        # Count system states
        state_counts = {}
        for entry in self.data_log:
            state = entry.get('system_state', 'unknown')
            state_counts[state] = state_counts.get(state, 0) + 1
            
        summary = {
            'task': self.task,
            'treatment': self.treatment,
            'total_data_points': total_logs,
            'aligned_goals': aligned_count,
            'alignment_percentage': alignment_percentage,
            'state_distribution': state_counts,
            'pick_goal_counter': self.pick_goal_counter,
            'place_goal_counter': self.place_goal_counter
        }
        
        summary_file = os.path.join(user_dir, "goal_alignment_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        rospy.loginfo(f"[GoalAlignmentLogger] Summary: {alignment_percentage:.1f}% alignment ({aligned_count}/{total_logs})")
            
    def on_shutdown(self):
        """Handle shutdown - save data"""
        rospy.loginfo("[GoalAlignmentLogger] Shutting down, saving data...")
        self.save_data()
        
        # Also save summary
        exp = UserStudyExperiment()
        user_dir = exp.get_user_dir(self.task, self.treatment)
        self.save_summary(user_dir)

def main():
    rospy.init_node('goal_alignment_logger')
    logger = GoalAlignmentLogger()
    
    # Register shutdown callback
    rospy.on_shutdown(logger.on_shutdown)
    
    rospy.loginfo("[GoalAlignmentLogger] Data logging started")
    rospy.spin()

if __name__ == '__main__':
    main()
