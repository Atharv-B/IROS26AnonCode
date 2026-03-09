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

class HighestConfidenceViewer:
    def __init__(self):
        rospy.init_node('highest_confidence_viewer', anonymous=True)
        self.bridge = CvBridge()

        self.current_image = None
        self.centroids_confidences = []
        #publishers
        self.image_pub = rospy.Publisher('/highest_confidence_object/image', Image, queue_size=10)
        # Subscribe to topics
        rospy.Subscriber('camera/color/image_raw', Image, self.image_callback, queue_size=10)
        rospy.Subscriber('/goal_confidence_centroids', CentroidConfidenceArray, self.confidence_callback)
        rospy.loginfo("Initialized viewer with colored bounding boxes and confidence values.")

    def image_callback(self, msg):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"Image conversion failed: {e}")
    
    def confidence_callback(self, msg):
        # build list of (centroid, conf, bbox, label)
        dets = [
            (
                item.centroid,
                item.confidence,
                {'x1':item.x1, 'y1':item.y1, 'x2':item.x2, 'y2':item.y2},
                item.label
            )
            for item in msg.items
        ]
        if not dets or self.current_image is None:
            rospy.logwarn("No detections or image available.")
            return

        # pick the best detection
        _, best_conf, best_bbox, best_label = max(dets, key=lambda t: t[1])
        x1, y1, x2, y2 = best_bbox['x1'], best_bbox['y1'], best_bbox['x2'], best_bbox['y2']

        # sanity‐check & clamp to image bounds
        # h, w = self.current_image.shape[:2]
        # x1, x2 = max(0, x1), min(w, x2)
        # y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            rospy.logwarn(f"Invalid best bbox: {best_bbox}")
            return

        # crop and show only that patch
        crop = self.current_image[y1:y2, x1:x2]
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(crop, encoding='bgr8'))
        rospy.loginfo(f"Showing highest confidence object: {best_label} with confidence {best_conf:.2f}")
        # cv2.imshow("Highest-Confidence Object", crop)
        # cv2.waitKey(1)

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
        #Create a publisher to show the image with all objects
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(image, encoding='bgr8'))

        cv2.imshow("All Objects with Confidence", image)
        cv2.waitKey(1)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        viewer = HighestConfidenceViewer()
        viewer.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()