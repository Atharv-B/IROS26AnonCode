#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from trust_and_transparency.msg import CentroidConfidenceArray
from cv_bridge import CvBridge
import cv2
import numpy as np

class AllObjectsConfidenceViewer:
    def __init__(self):
        rospy.init_node('all_objects_confidences_viewer', anonymous=True)
        self.bridge = CvBridge()

        # Mode: "all" or "highest"
        # - set at launch with:  rosparam set /all_objects_confidences_viewer/mode highest
        # - or in a launch file via <param name="mode" value="highest"/>
        # Runtime toggle: publish "all" or "highest" to /viewer_mode
        self.mode = rospy.get_param('~mode', 'all').strip().lower()
        if self.mode not in ('all', 'highest'):
            rospy.logwarn("Unknown ~mode '%s', defaulting to 'all'", self.mode)
            self.mode = 'all'

        # Optional: show window (disable for headless)
        self.show_window = rospy.get_param('~show_window', True)

        self.current_image = None
        # Store list of tuples: (centroid, confidence, bbox_dict, label)
        self.centroids_confidences = []

        # I/O
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback, queue_size=10)
        rospy.Subscriber('/goal_confidence_centroids', CentroidConfidenceArray, self.confidence_callback, queue_size=10)
        rospy.Subscriber('/viewer_mode', String, self.mode_callback, queue_size=1)

        self.pub = rospy.Publisher('/all_objects_confidences_image', Image, queue_size=10)

        rospy.loginfo("Initialized viewer (mode=%s). Toggling at /viewer_mode with 'all' or 'highest'.", self.mode)

        if self.show_window:
            cv2.namedWindow("Objects Confidence Viewer", cv2.WINDOW_NORMAL)

    # ------------------------------ Callbacks ------------------------------

    def image_callback(self, msg):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr("Image conversion failed: %s", e)

    def mode_callback(self, msg):
        new_mode = msg.data.strip().lower()
        if new_mode in ('all', 'highest'):
            if new_mode != self.mode:
                rospy.loginfo("Viewer mode changed: %s -> %s", self.mode, new_mode)
            self.mode = new_mode
        else:
            rospy.logwarn("Ignoring unknown viewer mode: '%s' (use 'all' or 'highest')", msg.data)

    def confidence_callback(self, msg):
        # Refresh detections
        self.centroids_confidences = []
        for item in msg.items:
            bbox = {'x1': item.x1, 'y1': item.y1, 'x2': item.x2, 'y2': item.y2}
            self.centroids_confidences.append((item.centroid, item.confidence, bbox, item.label))

        # Render based on mode
        if self.mode == 'highest':
            self.draw_highest_only()
        else:
            self.draw_all_objects()

    # ------------------------------ Utilities ------------------------------

    def confidence_to_color(self, conf):
        """Map confidence to color (red->green) for 'all' mode."""
        if not self.centroids_confidences:
            return (0, 0, 255)  # red in BGR

        all_confidences = [c for _, c, _, _ in self.centroids_confidences]
        if len(all_confidences) == 1 or max(all_confidences) == min(all_confidences):
            return (0, 255, 0)  # green

        mn, mx = min(all_confidences), max(all_confidences)
        normalized = (conf - mn) / (mx - mn + 1e-12)
        red = int((1.0 - normalized) * 255)
        green = int(normalized * 255)
        # BGR
        return (0, green, red)

    def valid_bbox(self, bbox, h, w):
        x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
        if x2 <= x1 or y2 <= y1:
            return False
        # Optional: clamp to image bounds
        return (0 <= x1 < w) and (0 <= x2 <= w) and (0 <= y1 < h) and (0 <= y2 <= h)

    # ------------------------------ Drawing ------------------------------

    def draw_all_objects(self):
        if self.current_image is None:
            return

        image = self.current_image.copy()
        h, w = image.shape[:2]

        any_drawn = False
        for _, confidence, bbox, label in self.centroids_confidences:
            if not self.valid_bbox(bbox, h, w):
                rospy.logwarn("Invalid bbox for %s: %s", label, bbox)
                continue

            x1, y1, x2, y2 = map(int, (bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']))
            color = self.confidence_to_color(confidence)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            # show label + confidence in 'all' mode
            cv2.putText(image, f"{label}: {confidence:.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            any_drawn = True

        if not any_drawn:
            rospy.logdebug("No valid boxes to draw in 'all' mode.")

        self.publish_and_show(image, window_title="Objects Confidence Viewer (all)")

    def draw_highest_only(self):
        if self.current_image is None:
            return

        image = self.current_image.copy()
        h, w = image.shape[:2]

        if not self.centroids_confidences:
            rospy.logdebug("No detections available for 'highest' mode.")
            self.publish_and_show(image, window_title="Objects Confidence Viewer (highest)")
            return

        # Choose highest confidence
        best = max(self.centroids_confidences, key=lambda t: t[1])  # (centroid, conf, bbox, label)
        _, _, bbox, _ = best

        if not self.valid_bbox(bbox, h, w):
            rospy.logwarn("Highest-confidence bbox invalid: %s", bbox)
            self.publish_and_show(image, window_title="Objects Confidence Viewer (highest)")
            return

        x1, y1, x2, y2 = map(int, (bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']))
        # Draw rectangle ONLY (no label or score)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        self.publish_and_show(image, window_title="Objects Confidence Viewer (highest)")

    # ------------------------------ Output ------------------------------

    def publish_and_show(self, image, window_title):
        # Publish to SAME topic regardless of mode
        try:
            image_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            self.pub.publish(image_msg)
        except Exception as e:
            rospy.logerr("Failed to publish image: %s", e)

        if self.show_window:
            try:
                cv2.imshow(window_title, image)
                cv2.waitKey(1)
            except Exception as e:
                rospy.logwarn("imshow failed (headless env?): %s", e)

    # ------------------------------ Run ------------------------------

    def run(self):
        rospy.spin()
        if self.show_window:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        viewer = AllObjectsConfidenceViewer()
        viewer.run()
    except rospy.ROSInterruptException:
        pass
