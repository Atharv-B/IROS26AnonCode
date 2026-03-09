#!/usr/bin/env python
import rospy
import torch
import torch.nn as nn
import numpy as np

class ActuatorModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.actuatorDataDict = {}
        self.toolPoseDataDict = {}
        self.demoCount = 0

    def setActuatorData(self, pos0, pos1, pos2, pos3, pos4, pos5, pos6):
        self.position_0 = pos0
        self.position_1 = pos1
        self.position_2 = pos2
        self.position_3 = pos3
        self.position_4 = pos4
        self.position_5 = pos5
        self.position_6 = pos6
        # self.gripper = gripper

        data = [self.position_0, self.position_1, self.position_2, self.position_3, self.position_4, self.position_5, self.position_6]
        np_array = np.array(data)
        self.actuatorDataDict[self.demoCount] = torch.from_numpy(np_array)
        self.demoCount += 1
        return self.get_position()

    def setToolPoseData(self, x, y, z, theta_x, theta_y, theta_z):
        self.tool_x = x
        self.tool_y = y
        self.tool_z = z
        self.tool_theta_x = theta_x
        self.tool_theta_y = theta_y
        self.tool_theta_z = theta_z
        self.gripper = gripper

        data = [self.tool_x, self.tool_y, self.tool_z, self.tool_theta_x, self.tool_theta_y, self.tool_theta_z]
        np_array = np.array(data)
        self.toolPoseDataDict[self.demoCount] = torch.from_numpy(np_array)
        self.demoCount += 1

    def get_position(self):
        data = [self.position_0, self.position_1, self.position_2, self.position_3, self.position_4, self.position_5, self.position_6]
        np_array = np.array(data)

        #self.actuatorDataDict[self.demoCount] = torch.from_numpy(np_array)
        #self.demoCount += 1
        return np_array