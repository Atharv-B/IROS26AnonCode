import pandas as pd
import os

class RobotLogger:
    def __init__(self):
        self.robot_events = []

    def log_robot_event(self, ts, feedback):
        if self.robot_events and self.robot_events[-1][0] > ts - 0.5:
            # Skip logging if the last event is too recent
            # Feel free to adjust the threshold to increase resolution
            return
        tool_position = [
            feedback.base.tool_pose_x,
            feedback.base.tool_pose_y,
            feedback.base.tool_pose_z,
            feedback.base.tool_pose_theta_x,
            feedback.base.tool_pose_theta_y,
            feedback.base.tool_pose_theta_z
        ]
        tool_twist = [
            feedback.base.tool_twist_linear_x,
            feedback.base.tool_twist_linear_y,
            feedback.base.tool_twist_linear_z,
            feedback.base.tool_twist_angular_x,
            feedback.base.tool_twist_angular_y,
            feedback.base.tool_twist_angular_z
        ]
        joint_angles = [feedback.actuators[i].position for i in range(len(feedback.actuators))]
        joint_velocities = [feedback.actuators[i].velocity for i in range(len(feedback.actuators))]
        joint_torques = [feedback.actuators[i].torque for i in range(len(feedback.actuators))]

        row = [ts] + tool_position + tool_twist + joint_angles + joint_velocities + joint_torques
        self.robot_events.append(row)

    def get_robot_events(self):
        return self.robot_events

    def save_robot_events(self, _dir):
        columns_raw = ["timestamp"] + [
            "tool_pose_x", "tool_pose_y", "tool_pose_z",
            "tool_pose_theta_x", "tool_pose_theta_y", "tool_pose_theta_z",
            "tool_twist_linear_x", "tool_twist_linear_y", "tool_twist_linear_z",
            "tool_twist_angular_x", "tool_twist_angular_y", "tool_twist_angular_z"
        ] + [f"joint_{i}_angle" for i in range(7)] + [f"joint_{i}_velocity" for i in range(7)] + [f"joint_{i}_torque" for i in range(7)]
        df = pd.DataFrame(self.robot_events, columns=columns_raw)
        df.to_csv(os.path.join(_dir, "robot_state.csv"), index=False)

