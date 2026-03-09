#!/usr/bin/env python3
import rospy
import cv2
import os
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from user_study.user_study import UserStudyExperiment

class CameraRecorder:
    def __init__(self):
        rospy.init_node("camera_recorder")
        
        # Get parameters from launch file
        self.task = rospy.get_param("~task", "shelving")
        self.treatment = rospy.get_param("~treatment", "C")
        
        # Get the output directory from user study structure
        self.experiment = UserStudyExperiment()
        self.output_dir = self.experiment.get_user_dir(self.task, self.treatment)
        
        rospy.loginfo(f"[CameraRecorder] Recording to: {self.output_dir}")
        rospy.loginfo(f"[CameraRecorder] Task: {self.task}, Treatment: {self.treatment}")
        
        self.bridge = CvBridge()
        self.writer = None
        
        # Output filename with serial number
        self.out_filename = os.path.join(self.output_dir, "realsense_246422070578.mp4")
        
        # Subscribe to the specific camera topic
        # Assuming the camera will be launched with namespace or with serial number in topic
        self.camera_topic = rospy.get_param("~camera_topic", "/camera_246422070578/color/image_raw")
        
        rospy.loginfo(f"[CameraRecorder] Subscribing to: {self.camera_topic}")
        rospy.Subscriber(self.camera_topic, Image, self.callback)
        
        rospy.on_shutdown(self.shutdown)

    def callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            if self.writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 30
                self.writer = cv2.VideoWriter(self.out_filename, fourcc, fps, (w, h))
                rospy.loginfo(f"[CameraRecorder] Started recording: {self.out_filename} ({w}x{h} @ {fps}fps)")
            
            self.writer.write(frame)
        except Exception as e:
            rospy.logerr(f"[CameraRecorder] Error in callback: {e}")

    def shutdown(self):
        if self.writer:
            self.writer.release()
            rospy.loginfo(f"[CameraRecorder] Recording saved to: {self.out_filename}")

if __name__ == '__main__':
    try:
        print("Starting Camera Recorder Node...")
        recorder = CameraRecorder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
