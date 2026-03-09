#!/usr/bin/env python3

import time
import rospy
import numpy as np
import json
from sensor_msgs.msg import Joy
from kortex_driver.msg import Twist, BaseCyclic_Feedback, TwistCommand, Finger, GripperCommand, GripperMode
from kortex_driver.srv import SendGripperCommand, SendGripperCommandRequest, SendTwistJoystickCommand
from kortex_api.Exceptions.KException import KException
from constants import SPEED_CONTROL, HOME, TOP_DOWN
from example_full_arm_movement import ExampleFullArmMovement
from user_study.joy_logger import JoyLogger
from user_study.robot_logger import RobotLogger
from user_study.user_study import UserStudyExperiment
import shutil
import os

class DirectTeleoperation:
    def __init__(self):
        self.uh = Twist()
        self.last_nonzero_uh = Twist()
        self.human_input = False
        self.Y_pressed = False
        self.gripper_closed = False
        self.wait_for_first_input = True
        self.manual_mode = False
        self.joy_logger = JoyLogger()
        self.robot_logger = RobotLogger()

        self.ee_position = np.array([0.0, 0.0, 0.0])
        self.task = rospy.get_param("~task", "shelving")  # Default to "shelving" task
        self.treatment = rospy.get_param("~treatment", "A") # Default to "A" treatment
        
        # Match goal_alignment_logger trigger detection for consistency
        self.trigger_threshold = rospy.get_param("~trigger_threshold", -0.8)
        self.lt_was_pressed = False  # left trigger (close gripper)
        self.rt_was_pressed = False  # right trigger (open gripper)
        
        # Track input magnitude (same as goal_alignment_logger)
        self.current_input_magnitude = 0.0
        self.cumulative_input_magnitude = 0.0  # Sum of all input magnitudes
        self.input_sample_count = 0  # Count of samples for averaging
        
        # Track task timing for summary statistics
        self.task_start_time = None
        self.task_end_time = None
        self.first_input_received = False

        rospy.Subscriber("/joy", Joy, self.controller_callback)
        rospy.Subscriber("/my_gen3/base_feedback", BaseCyclic_Feedback, self.base_feedback_callback)

        rospy.wait_for_service("/my_gen3/base/send_gripper_command")
        self.gripper_command = rospy.ServiceProxy("/my_gen3/base/send_gripper_command", SendGripperCommand)

        rospy.wait_for_service("/my_gen3/base/send_twist_joystick_command")
        self.send_twist_command = rospy.ServiceProxy("/my_gen3/base/send_twist_joystick_command", SendTwistJoystickCommand)

        self.reset = ExampleFullArmMovement()

    def calculate_input_magnitude(self, controller_position):
        """
        Calculate input magnitude from joystick axes.
        Same calculation as goal_alignment_logger for consistency.
        """
        if len(controller_position) < 5:
            return 0.0
            
        # Xbox controller mapping (same as goal_alignment_logger)
        linear_x = float(controller_position[1])  # Left stick Y (forward/backward)
        linear_y = float(controller_position[0])  # Left stick X (left/right)  
        linear_z = float(controller_position[4])  # Right stick Y (up/down)
        
        # Calculate Euclidean norm directly from raw values
        return float(np.linalg.norm([linear_x, linear_y, linear_z]))


    def controller_callback(self, data):
        ts = rospy.get_time()
        controller_position = data.axes
        controller_buttons = data.buttons
        self.joy_logger.log_joy_event(ts, controller_position, controller_buttons)

        # Calculate input magnitude (matching goal_alignment_logger)
        self.current_input_magnitude = self.calculate_input_magnitude(controller_position)
        
        # Accumulate input magnitude for cumulative tracking
        if self.first_input_received:
            self.cumulative_input_magnitude += self.current_input_magnitude
            self.input_sample_count += 1
        
        # Track first human input for task timing
        if not self.first_input_received and self.current_input_magnitude > 1e-6:
            self.first_input_received = True
            self.task_start_time = rospy.get_time()
            rospy.loginfo(f"[Teleop] Task started at {self.task_start_time}")

        # RESET TO POSITION (Only Reset if the gripper is open):
        if controller_buttons[3] == 1 and not self.gripper_closed:
            self.Y_pressed = True
            print("RESET ROBOT")

        # Use threshold-based trigger detection (matching goal_alignment_logger)
        # This ensures both nodes agree on gripper state
        lt_pressed = (len(controller_position) > 2 and controller_position[2] < self.trigger_threshold)   # Left trigger => CLOSE
        rt_pressed = (len(controller_position) > 5 and controller_position[5] < self.trigger_threshold)   # Right trigger => OPEN

        # RIGHT TRIGGER: Open gripper (rising edge detection)
        if rt_pressed and not self.rt_was_pressed and self.gripper_closed:
            self.send_gripper_command(0.0)
            self.gripper_closed = False
            self.manual_mode = True
            self.wait_for_first_input = False  # allow immediate manual control
            rospy.loginfo("[Teleop] Gripper opened — MANUAL-ONLY until Y.")

        # LEFT TRIGGER: Close gripper (rising edge detection)
        elif lt_pressed and not self.lt_was_pressed and not self.gripper_closed:
            self.send_gripper_command(0.45)
            self.gripper_closed = True
            self.last_nonzero_uh = Twist()

        # Update trigger states for edge detection
        self.lt_was_pressed = lt_pressed
        self.rt_was_pressed = rt_pressed

        twist = Twist()
        linear = np.array([
            float(controller_position[1]), # Controller X
            float(controller_position[0]), # Controller Y
            float(controller_position[4]), # Controller Z
            ])
        
        # print("Linear Twist Command:", linear)
        
        #Normalize Input
        if np.linalg.norm(linear) > 1e-9:
            linear /= np.linalg.norm(linear)

        twist.linear_x, twist.linear_y, twist.linear_z = linear
        self.uh = twist

        if self.wait_for_first_input and np.linalg.norm(linear) > 1e-3:
            self.wait_for_first_input = False
            rospy.loginfo("[Teleop] First human input received — enabling motion.")

        if controller_position[1] == 0 and controller_position[0] == 0 and controller_position[4] == 0:
            self.human_input = False
        else:
            self.human_input = True
            self.last_nonzero_uh = twist

    def base_feedback_callback(self, feedback):
        ts = rospy.get_time()
        if self.robot_logger:
            self.robot_logger.log_robot_event(ts, feedback)
        self.ee_position[0] = feedback.base.tool_pose_x
        self.ee_position[1] = feedback.base.tool_pose_y
        self.ee_position[2] = feedback.base.tool_pose_z

    def send_gripper_command(self, value):
        req = SendGripperCommandRequest()
        finger = Finger()
        finger.finger_identifier = 0
        finger.value = value
        req.input.gripper.finger.append(finger)
        req.input.mode = GripperMode.GRIPPER_POSITION
        try:
            self.gripper_command(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to send gripper command")

    def twist_command(self, x, y, z, angle_x=0, angle_y=0, angle_z=0):
        twist = Twist()
        twist.linear_x = float(x)
        twist.linear_y = float(y)
        twist.linear_z = float(z)
        
        twist_cmd = TwistCommand()
        twist_cmd.duration = 0
        twist_cmd.reference_frame = 1
        twist_cmd.twist = twist

        try:
            self.send_twist_command(twist_cmd)
        except rospy.ServiceException as e:
            rospy.logerr("Failed to send twist command")
        except KException as ke:
            rospy.logerr("Kortex exception")
            rospy.loginfo(f"KINOVA exception error code: {ke.get_error_code()}")
            rospy.loginfo(f"KINOVA exception error sub code: {ke.get_error_sub_code()}")
            rospy.loginfo(f"KINOVA exception description: {ke.what()}")

    def on_shutdown(self):
        # Put any additional shutdown steps here, such as saving data
        user_dir = UserStudyExperiment().get_user_dir(self.task, self.treatment)
        rospy.loginfo("Saving teleoperation history...")
        self.joy_logger.save_joy_events(user_dir)
        rospy.loginfo("Saving robot state history...")
        self.robot_logger.save_robot_events(user_dir)
        
        # Save task timing summary
        if self.task_start_time is not None:
            self.task_end_time = rospy.get_time()
            task_duration = self.task_end_time - self.task_start_time
            
            # Calculate average input magnitude
            avg_input_magnitude = (self.cumulative_input_magnitude / self.input_sample_count 
                                  if self.input_sample_count > 0 else 0.0)
            
            summary = {
                'task': self.task,
                'treatment': self.treatment,
                'task_start_time': self.task_start_time,
                'task_end_time': self.task_end_time,
                'task_duration_seconds': task_duration,
                'task_duration_minutes': task_duration / 60.0,
                'final_input_magnitude': self.current_input_magnitude,
                'cumulative_input_magnitude': self.cumulative_input_magnitude,
                'average_input_magnitude': avg_input_magnitude,
                'input_sample_count': self.input_sample_count
            }
            
            summary_file = os.path.join(user_dir, "direct_teleop_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            rospy.loginfo(f"[Teleop] Task completed in {task_duration:.2f} seconds ({task_duration/60.0:.2f} minutes)")
            rospy.loginfo(f"[Teleop] Final input magnitude: {self.current_input_magnitude:.3f}")
            rospy.loginfo(f"[Teleop] Cumulative input magnitude: {self.cumulative_input_magnitude:.3f}")
            rospy.loginfo(f"[Teleop] Average input magnitude: {avg_input_magnitude:.3f}")
            rospy.loginfo(f"[Teleop] Summary saved to: {summary_file}")
        
        # shutil.move("end_effector_perspective.mov", os.path.join(user_dir, "end_effector_perspective.mov"))
        # shutil.move("env_camera_perspective.mov", os.path.join(user_dir, "env_camera_perspective.mov"))

    def main(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.Y_pressed:
                rospy.loginfo("Resetting to home position...")
                if self.task == "shelving":
                    rospy.loginfo("Resetting to home...")
                    self.twist_command(-0.45, 0, 0.45)
                    rospy.sleep(3)
                    self.twist_command(0, 0, 0)
                    rospy.sleep(0.5)
                    self.reset.example_send_joint_angles(HOME)
                    rospy.sleep(8.0)
                    # self.reset.example_clear_faults()
                if self.task == "sorting":
                    rospy.loginfo("Resetting to home...")
                    self.twist_command(-0.45, 0, 0.45)
                    rospy.sleep(3)
                    self.twist_command(0, 0, 0)
                    rospy.sleep(0.5)
                    self.reset.example_send_joint_angles(TOP_DOWN)
                    rospy.sleep(8.0)
                self.reset.example_clear_faults()
                self.Y_pressed = False
                self.manual_mode = False
                self.wait_for_first_input = True
                rospy.loginfo("[Teleop] Reset done — SA re-armed; waiting for first nudge.")

                time.sleep(0.2)
            else:
                DT_SPEED_CONTROL = 0.6
                self.twist_command(self.uh.linear_x * DT_SPEED_CONTROL, self.uh.linear_y * DT_SPEED_CONTROL, self.uh.linear_z * DT_SPEED_CONTROL)
            rate.sleep()

        rospy.logwarn("Before Shutdown Attempt to Save Trial Data to User Folder")
        self.on_shutdown()

if __name__ == "__main__":
    rospy.init_node("direct_teleop")
    node = DirectTeleoperation()
    node.main()
