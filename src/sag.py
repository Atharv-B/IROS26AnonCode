#!/usr/bin/env python3

import rospy
import time
import math
import numpy as np
import torch
import copy
import torch.nn.functional as F
from direct_teleop import DirectTeleoperation
from constants import ORACLE_GOAL_SET,HOME, GOAL_POP_THRESHOLD, SPEED_CONTROL
from user_study.goal_logger import GoalLogger
from user_study.user_study import UserStudyExperiment

class SAGTeleoperation(DirectTeleoperation):
    def __init__(self):
        super().__init__()

        self.ee_position = np.array([0.0, 0.0, 0.0])
        self.last_nonzero_uh = self.uh

        if self.task not in ORACLE_GOAL_SET:
            rospy.logerr(f"Invalid task: {self.task}. Available: {list(ORACLE_GOAL_SET.keys())}")
            raise ValueError(f"Task {self.task} not recognized.")

        self.pick_set = ORACLE_GOAL_SET[self.task]["object_locations"]
        self.place_set = ORACLE_GOAL_SET[self.task]["placement_locations"]
        self.current_goal_set = self.pick_set  # Initially go for picking
        self.goal_logger = GoalLogger()
        self.place_set_template = copy.deepcopy(self.place_set)
        self.place_set_orig = copy.deepcopy(self.place_set)   # never touched (viz)
        self.place_set_mod  = copy.deepcopy(self.place_set)   # used for motion/conf
        rospy.loginfo(f"[SAG] Task: {self.task}, Pick: {len(self.pick_set)}, Place: {len(self.place_set)}")

    def compute_ur_for_all_goals(self):
        ur_list = []
        for goal in self.current_goal_set:
            direction = np.array(goal) - self.ee_position
            norm = np.linalg.norm(direction)
            if norm < 1e-2:
                ur_list.append(np.zeros(3))
            else:
                ur_list.append(direction / norm)
        # Log the current goal state here
        # print(f"[SAG] Current Goal Set: {self.current_goal_set}")
        self.goal_logger.log_goal_event(rospy.get_time(), self.current_goal_set)
        return ur_list

    def infer_goal(self, ur_list):
        confidences = []
        for i, ur in enumerate(ur_list):
            conf = self.compute_confidence(ur, i)
            confidences.append(conf)
        softmax_conf = F.softmax(torch.tensor(confidences), dim=0).numpy()
        best_goal = int(np.argmax(softmax_conf))
        return best_goal, softmax_conf[best_goal]

    def compute_confidence(self, ur, idx):
        w1 = 0.3
        w2 = 1.0 - w1
        dist = np.linalg.norm(self.ee_position - self.current_goal_set[idx])
        human_dir = np.array([self.last_nonzero_uh.linear_x,
                              self.last_nonzero_uh.linear_y,
                              self.last_nonzero_uh.linear_z])
        dot = np.dot(human_dir, ur)
        return w1 * dot + w2 * math.exp(-dist)

    def blend_inputs(self, uh, ur, conf, alpha_min=0.0, alpha_max=0.8):
        h_vec = np.array([uh.linear_x, uh.linear_y, uh.linear_z])
        alpha = alpha_min + (alpha_max - alpha_min) * conf if np.linalg.norm(h_vec) > 1e-3 else 0.8 
        return (alpha * ur + (1 - alpha) * h_vec), alpha

    def update_place_set_z(self):
        """
        Update Z value of the place set using current end-effector Z position.
        """
        
        if self.gripper_closed and not self.Z_is_updated:
            self.adjusted_z = self.ee_position[2]
            for i in range(len(self.place_set_mod)):
                base_z = self.place_set_orig[i][2]        # original Z as baseline
                self.place_set_mod[i][2] = base_z + self.adjusted_z
            self.Z_is_updated = True
            rospy.loginfo(f"[VOSA] Updated place_set Z (motion) with adjusted_z: {self.adjusted_z:.3f}")
        if not self.gripper_closed:
            self.Z_is_updated = False


    def drop_reached_place_goals(self):
        """
        Drop any goal from place_set that is close to the end-effector when gripper is open.
        """
        if not self.gripper_closed: #and self.current_goal_set == self.place_set:
            new_place_set = []
            for goal in self.place_set:
                dist = np.linalg.norm(self.ee_position - goal)
                if dist >= 0.05:
                    new_place_set.append(goal)
                else:
                    rospy.loginfo(f"[SAG] Removed placed goal: {goal}")
            self.place_set = new_place_set
            rospy.loginfo(f"[SAG] Remaining place goals: {len(self.place_set)}")


    def main(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            # --- 0) Y should interrupt everything (manual/wait/SA) ---
            if self.Y_pressed:
                rospy.loginfo("Resetting to home...")
                self.twist_command(-0.45, 0, 0.45)
                time.sleep(3)
                self.twist_command(0, 0, 0)
                time.sleep(0.5)
                self.reset.example_send_joint_angles(HOME)
                time.sleep(6.0)
                self.reset.example_clear_faults()

                # leave manual-only and re-arm first-nudge gate
                self.manual_mode = False
                self.wait_for_first_input = True

                self.Y_pressed = False
                time.sleep(0.2)
                rate.sleep()
                continue

            # --- 1) Manual-only pass-through (after placing until Y) ---
            if self.manual_mode:
                self.twist_command(self.uh.linear_x * SPEED_CONTROL,
                                self.uh.linear_y * SPEED_CONTROL,
                                self.uh.linear_z * SPEED_CONTROL)
                rate.sleep()
                continue

            # --- 2) First-nudge gate (startup/after Y) ---
            if self.wait_for_first_input and self.input_magnitude(self.uh) <= 1e-6:
                self.twist_command(0.0, 0.0, 0.0)
                rate.sleep()
                continue
            else:
                # clear the gate on the first nonzero input
                if self.wait_for_first_input:
                    self.wait_for_first_input = False
            self.update_place_set_z()
            self.current_goal_set = self.place_set_mod if self.gripper_closed else self.pick_set
            # self.drop_reached_place_goals()
            # print(f"[SAG] Current place set: {self.place_set}")

            if self.input_magnitude(self.uh) > 1e-6:
                self.last_nonzero_uh = self.uh
                
            elif self.current_goal_set:
                ur_list = self.compute_ur_for_all_goals()
                goal_idx, confidence = self.infer_goal(ur_list)

                blended, alpha = self.blend_inputs(self.uh, ur_list[goal_idx], confidence)
                # ----- Make it behave like AssistiveTeleoperation -----
                cmd = np.array(blended, dtype=float)

                if self.task == "shelving":
                    # Match Assistive: big Z when placing, smaller Z otherwise
                    z_mult_place = rospy.get_param("~z_place_mult", 15.0)
                    z_mult_pick  = rospy.get_param("~z_pick_mult",  3.0)
                    cmd[2] *= (z_mult_place if self.gripper_closed else z_mult_pick)

                    # Normalize XYZ length to ≤ 1 (Assistive does this)
                    xyz_norm = np.linalg.norm(cmd[0:3])
                    if xyz_norm > 1.0:
                        cmd[0:3] /= xyz_norm

                    # Final global scale (Assistive uses 0.5)
                    final_scale = rospy.get_param("~xyz_final_scale", 0.5)  # or use SPEED_CONTROL if you prefer
                    cmd[0:3] *= final_scale
                else:
                    # Non-shelving behavior unchanged (optional: keep your previous scaling)
                    cmd *= 1.0  # no-op; or cmd[0:3] *= SPEED_CONTROL

                # Safety clamp and send
                cmd[0:3] = np.clip(cmd[0:3], -1.0, 1.0)
                rospy.loginfo_throttle(0.5, f"[SAG] send XYZ={cmd[0]:.3f},{cmd[1]:.3f},{cmd[2]:.3f} (grip={self.gripper_closed})")
                self.twist_command(cmd[0], cmd[1], cmd[2])
            else:
                self.twist_command(self.uh.linear_x * 0.5,
                                   self.uh.linear_y * 0.5,
                                   self.uh.linear_z * 0.5)
            rate.sleep()

        rospy.logwarn("Before Shutdown Attempt to Save Trial Data to User Folder")
        self.on_shutdown()

    def input_magnitude(self, uh):
        return np.linalg.norm(np.array([uh.linear_x, uh.linear_y, uh.linear_z]))

    def on_shutdown(self):
        super().on_shutdown()
        user_dir = UserStudyExperiment().get_user_dir(self.task, self.treatment)
        rospy.loginfo("Saving goal events...")
        self.goal_logger.save_goal_events(user_dir)
        # Put any additional shutdown steps here, such as saving data

if __name__ == "__main__":
    rospy.init_node("sag_teleop")
    node = SAGTeleoperation()
    node.main()
