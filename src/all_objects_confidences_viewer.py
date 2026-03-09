#!/usr/bin/env python3

import rospy
import json
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from trust_and_transparency.msg import CentroidConfidenceArray
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from cam_to_world import CameraToWorld
from constants import PLACEMENT_THRESHOLDS

class AllObjectsConfidenceViewer:
    def __init__(self):
        rospy.init_node('all_objects_confidences_viewer', anonymous=True)
        self.bridge = CvBridge()

        self.current_image = None
        self.centroids_confidences = []

        # Subscribe to topics
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback, queue_size=10)
        rospy.Subscriber('/goal_confidence_centroids', CentroidConfidenceArray, self.confidence_callback)
        rospy.loginfo("Initialized viewer with colored bounding boxes and confidence values.")
        

    def image_callback(self, msg):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"Image conversion failed: {e}")
    
    def confidence_callback(self, msg):
        # Clear previous data and store new data with bounding boxes
        self.centroids_confidences.clear()
        
        for item in msg.items:
            label = item.label
            centroid = item.centroid
            confidence = item.confidence
            bbox = {'x1': item.x1, 'y1': item.y1, 'x2': item.x2, 'y2': item.y2}
            
            self.centroids_confidences.append((centroid, confidence, bbox, label))
        
        self.draw_all_objects()

    def confidence_to_color(self, conf):
        """Map confidence to color from red (low) to green (high)"""
        if not self.centroids_confidences:
            return (255, 0, 0)
        
        all_confidences = [confidence for _, confidence, _, _ in self.centroids_confidences]
        
        if len(all_confidences) == 1:
            return (0, 255, 0)
        
        min_conf = min(all_confidences)
        max_conf = max(all_confidences)
        
        if max_conf == min_conf:
            return (0, 255, 0)

        normalized_conf = (conf - min_conf) / (max_conf - min_conf)
        red = int((1.0 - normalized_conf) * 255)
        green = int(normalized_conf * 255)
        
        return (0, green, red)

    def draw_all_objects(self):
        if self.current_image is None:
            return

        image = self.current_image.copy()

        for centroid, confidence, bbox, label in self.centroids_confidences:
            color = self.confidence_to_color(confidence)
            
            # Draw bounding box if coordinates are valid
            if bbox['x2'] > bbox['x1'] and bbox['y2'] > bbox['y1']:
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f"{label}: {confidence:.2f}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                rospy.logwarn(f"Invalid bbox for {label}: {bbox}")
        
        # Create a publisher for the processed image
        pub = rospy.Publisher('/all_objects_confidences_image', Image, queue_size=10)
        try:
            image_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            pub.publish(image_msg)
        except Exception as e:
            rospy.logerr(f"Failed to publish image: {e}")

        cv2.imshow("All Objects with Confidence", image)
        cv2.waitKey(1)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        viewer = AllObjectsConfidenceViewer()
        viewer.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()

