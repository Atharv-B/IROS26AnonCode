#!/usr/bin/env python3
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class CameraRecorder:
    def __init__(self, topic, out_filename):
        self.bridge = CvBridge()
        self.writer = None
        self.out_filename = out_filename
        rospy.Subscriber(topic, Image, self.callback)

    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if self.writer is None:
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(self.out_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        self.writer.write(frame)

    def shutdown(self):
        if self.writer:
            self.writer.release()

if __name__ == '__main__':
    rospy.init_node("camera_recorder")
    end_effector = CameraRecorder("/247122071167/color/image_raw", "end_effector_perspective.mov")
    external = CameraRecorder("/env_cam/color/image_raw", "env_camera_perspective.mov")
    rospy.on_shutdown(lambda: (end_effector.shutdown(), external.shutdown()))
    rospy.spin()
