#!/usr/bin/env python3
import json
import rospy
import tf2_ros
import numpy as np
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, Point
from sensor_msgs.msg import Image, CameraInfo
from image_geometry import PinholeCameraModel
from trust_and_transparency.msg import CentroidConfidenceArray, CentroidConfidence
from std_msgs.msg import String, Bool
from constants import ORACLE_GOAL_SET

class TransformExample:
    def __init__(self):
        rospy.init_node('scene_bbox_viz', anonymous=True)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Initialize camera model and image handling
        self.bridge = CvBridge()
        self.cam_model = PinholeCameraModel()
        self.have_cam_info = False
        self.cam_frame = None
        self.latest_image = None
        self.centroids = []
        self.gripper_open = True  # Assume gripper is open initially
        
        # Centroid stabilization
        self.stable_centroids = {}  # Dictionary to store stable centroid positions
        self.stable_pixels = {}     # Dictionary to store stable pixel positions
        self.movement_threshold = rospy.get_param('~movement_threshold', 0.03)  # 3cm threshold
        self.pixel_threshold = rospy.get_param('~pixel_threshold', 1)  # 5 pixel threshold
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.3)  # Only stabilize confident detections
        
        # Calibration offsets (adjust these to fine-tune centroid positions)
        self.offset_x = rospy.get_param('~offset_x', 0.15)
        self.offset_y = rospy.get_param('~offset_y', -0.01)
        self.offset_z = rospy.get_param('~offset_z', 0.0)
        
        # Publishers and subscribers
        self.overlay_pub = rospy.Publisher('/scene_camera/centroid_overlay', Image, queue_size=1)
        self.sparse_pub = rospy.Publisher('/scene_camera/highest_confidence_overlay', Image, queue_size=1)  # New publisher

        # NEW: Publisher for highest confidence label
        self.highest_conf_label_pub = rospy.Publisher('/highest_confidence_label', String, queue_size=10)

        # NEW: Publisher for all objects (for verbal_rich feedback)
        self.all_objects_pub = rospy.Publisher('/all_objects_labels_confidences', String, queue_size=10)

        self.raw_bb = []
        
        # Subscribe to camera info and image
        rospy.Subscriber('/scene_camera/camera/color/camera_info', CameraInfo, self.camera_info_callback)
        rospy.Subscriber('/scene_camera/camera/color/image_raw', Image, self.image_callback)


        # Subscribe to scene YOLO detections
        rospy.Subscriber('/scene_camera/detected_objects/dict_label_centroid', String, self.dict_label_centroid_callback)
        
        # Subscribe to centroids
        rospy.Subscriber('/goal_confidence_centroids', CentroidConfidenceArray, self.centroids_callback)
        
        rospy.loginfo("Scene Centroid Visualizer initialized")
        
        # Give the transform listener some time to populate the buffer
        rospy.loginfo("Waiting for transform data...")

    def camera_info_callback(self, msg):
        """Callback for camera info"""
        if self.have_cam_info:
            return
        self.cam_model.fromCameraInfo(msg)
        self.cam_frame = msg.header.frame_id
        self.have_cam_info = True
        rospy.loginfo(f"Camera info received for frame: {self.cam_frame}")

    def dict_label_centroid_callback(self, msg: String):
        """
        Handle both single dict and list formats from YOLO
        msg.data can be:
        - Single dict: {"label": "...", "centroid": {x,y,z}}
        - List: [{"label": "...", "centroid": {x,y,z}}, ...]
        """
        try:
            self.raw_bb = []
            data = json.loads(msg.data)
            
            # Handle list format (multiple detections)
            if isinstance(data, dict):
                self.raw_bb = [data]
            elif isinstance(data, list):
                self.raw_bb = data
            else:
                rospy.logwarn(f"[VOSA] Unexpected data format in dict_label_centroid_callback: {type(data)}")
                
        except Exception as e:
            rospy.logerr(f"[VOSA] Error in dict_label_centroid_callback: {e}")

    def image_callback(self, msg):
        """Callback for camera images"""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # self.draw_and_publish()
        except Exception as e:
            rospy.logwarn(f"Image conversion failed: {e}")

    def centroids_callback(self, msg):
        """Callback for centroid confidence array"""
        self.centroids = []
        for item in msg.items:
            centroid_data = {
                'centroid': item.centroid,
                'confidence': item.confidence,
                'label': item.label
            }
            self.gripper_open = item.gripper_open

            # Apply stabilization to reduce jitter
            stabilized_centroid = self.stabilize_centroid(centroid_data)
            self.centroids.append(stabilized_centroid)

        rospy.loginfo(f"Received {len(self.centroids)} centroids")
        self.draw_and_publish()

    def stabilize_centroid(self, centroid_data):
        """Stabilize centroid position to reduce jitter"""
        label = centroid_data['label']
        current_pos = centroid_data['centroid']
        confidence = centroid_data['confidence']
        
        # Only apply stabilization to confident detections
        if confidence < self.confidence_threshold:
            return centroid_data
        
        # Check if we have a previous stable position for this label
        if label in self.stable_centroids:
            stable_pos = self.stable_centroids[label]
            
            # Calculate distance between current and stable position
            dx = current_pos.x - stable_pos.x
            dy = current_pos.y - stable_pos.y
            dz = current_pos.z - stable_pos.z
            distance = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            # If movement is below threshold, use stable position
            if distance < self.movement_threshold:
                # rospy.logdebug(f"Using stable position for {label} (movement: {distance:.3f}m)")
                centroid_data['centroid'] = stable_pos
                return centroid_data
            else:
                pass
                # rospy.loginfo(f"Object {label} moved {distance:.3f}m, updating stable position")
        
        # Update stable position for this label
        self.stable_centroids[label] = current_pos
        return centroid_data

    def stabilize_pixel_position(self, label, u, v, confidence):
        """Stabilize pixel position to reduce visual jitter"""
        # Only apply stabilization to confident detections
        if confidence < self.confidence_threshold:
            return u, v
        
        # Check if we have a previous stable pixel position for this label
        if label in self.stable_pixels:
            stable_u, stable_v = self.stable_pixels[label]
            
            # Calculate pixel distance between current and stable position
            pixel_distance = np.sqrt((u - stable_u)**2 + (v - stable_v)**2)
            
            # If movement is below pixel threshold, use stable position
            if pixel_distance < self.pixel_threshold:
                # rospy.logdebug(f"Using stable pixel position for {label} (movement: {pixel_distance:.1f}px)")
                return stable_u, stable_v
            else:
                pass
                # rospy.loginfo(f"Object {label} moved {pixel_distance:.1f}px, updating stable pixel position")
        
        # Update stable pixel position for this label
        self.stable_pixels[label] = (u, v)
        return u, v

    def confidence_to_color(self, confidence, min_conf=None, max_conf=None):
        """Convert confidence to BGR color (red=low, green=high)"""
        if min_conf is None or max_conf is None or min_conf == max_conf:
            # Fallback to absolute confidence mapping
            green = int(confidence * 255)
            red = int((1.0 - confidence) * 255)
            return (0, green, red)  # BGR format
        
        # Normalize confidence relative to min/max in current batch
        normalized_conf = (confidence - min_conf) / (max_conf - min_conf)
        green = int(normalized_conf * 255)
        red = int((1.0 - normalized_conf) * 255)
        return (0, green, red)  # BGR format
    
    def draw_and_publish(self):
        """Transform centroids and draw them on the image"""
        if not (self.have_cam_info and self.latest_image is not None and self.centroids):
            rospy.loginfo_throttle(2.0, f"Waiting for: cam_info={self.have_cam_info}, image={self.latest_image is not None}, centroids={len(self.centroids) if self.centroids else 0}")
            return
            
        image = self.latest_image.copy()
        sparse_image = self.latest_image.copy()  # New image for sparse feedback
        h, w = image.shape[:2]
        
        # Calculate min and max confidence for relative color mapping
        confidences = [centroid_data['confidence'] for centroid_data in self.centroids]
        min_conf = min(confidences) if confidences else 0.0
        max_conf = max(confidences) if confidences else 1.0
        highest_conf_matched_label = None  # NEW: Store the actual matched bbox label
        
        # Find highest confidence object
        highest_conf_data = None
        if self.centroids:
            highest_conf_data = max(self.centroids, key=lambda x: x['confidence'])
            # rospy.loginfo(f"Highest confidence object: {highest_conf_data['label']} with confidence {highest_conf_data['confidence']:.2f}")
        # NEW: Dictionary to store matched labels for each centroid
        matched_labels = {}  # key: centroid_data id, value: matched label
        points_plotted = 0
        highest_conf_plotted = False
        
        for centroid_data in self.centroids:
            try:
                # Create PointStamped from centroid
                point_stamped = PointStamped()
                point_stamped.header.frame_id = "base_link"  # Centroids should be in base_link after cam_to_world transformation
                point_stamped.header.stamp = rospy.Time.now()
                point_stamped.point = centroid_data['centroid']
                
                # Apply calibration offsets for fine-tuning
                point_stamped.point.x += 0
                point_stamped.point.y += 0
                point_stamped.point.z += 0
                
                # Look up transform
                transform = self.tf_buffer.lookup_transform(
                    self.cam_frame,
                    "base_link",
                    rospy.Time(0),
                    rospy.Duration(1.0)
                )
                
                # Transform point manually
                transformed_point = self.transform_point_manually(point_stamped, transform)
                # rospy.loginfo(f"Original point: {point_stamped.point}")
                # rospy.loginfo(f"Transformed point: {transformed_point}")
                
                # Project to image coordinates
                X = transformed_point.point.x 
                Y = transformed_point.point.y
                Z = transformed_point.point.z
                
                # rospy.loginfo(f"Camera coordinates: X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}")
                
                if Z <= 0:  # Behind camera
                    rospy.logwarn(f"Point behind camera: Z={Z:.3f}")
                    continue
                    
                u, v = self.cam_model.project3dToPixel((X, Y, Z))
                u, v = int(round(u)), int(round(v))
                
                # Apply pixel-level stabilization
                u, v = self.stabilize_pixel_position(centroid_data['label'], u, v, centroid_data['confidence'])
                
                # rospy.loginfo(f"Projected to pixel: u={u}, v={v}")
                
                (px, py) = (u, v)

                # Find all candidate bounding boxes that contain this point
                bbox_candidates = []
                for bb_candidate in self.raw_bb:
                    bounding_box = bb_candidate["bounding_box"]
                    (x1, y1, x2, y2) = (bounding_box["x1"], bounding_box["y1"], bounding_box["x2"], bounding_box["y2"])
                    if (x1 <= px <= x2 and y1 <= py <= y2):
                        bbox_candidates.append(bb_candidate)

                # Initialize bbox and label
                bbox = None
                label = None

                # First, try to match by label if confidence is high enough
                expected_label = centroid_data['label']
                if centroid_data['confidence'] > 0.5 and len(bbox_candidates) > 0:
                    for candidate in bbox_candidates:
                        if candidate["label"] == expected_label:
                            bbox = candidate["bounding_box"]
                            label = candidate["label"]
                            # rospy.loginfo(f"Matched by label priority: {label}")
                            break

                # If no label match, fall back to closest center or single match
                if bbox is None and len(bbox_candidates) == 1:
                    bbox = bbox_candidates[0]["bounding_box"]
                    label = bbox_candidates[0]["label"]
                    # rospy.loginfo(f"Single bbox match: {label}")
                elif bbox is None and len(bbox_candidates) > 1:
                    # Multiple matches - choose closest bbox center to centroid
                    min_distance = float('inf')
                    for candidate in bbox_candidates:
                        bb = candidate["bounding_box"]
                        center_x = (bb["x1"] + bb["x2"]) / 2.0
                        center_y = (bb["y1"] + bb["y2"]) / 2.0
                        distance = np.sqrt((center_x - px)**2 + (center_y - py)**2)
                        
                        if distance < min_distance:
                            min_distance = distance
                            bbox = bb
                            label = candidate["label"]
                    
                    # rospy.loginfo(f"Centroid at ({px}, {py}) matched to {label} by distance (among {len(bbox_candidates)} candidates), distance={min_distance:.1f}px")
                elif len(bbox_candidates) == 0:
                    rospy.logwarn(f"Centroid at ({px}, {py}) for {centroid_data['label']} not inside any bounding box")

                # NEW: Store the matched label (or use centroid label as fallback)
                centroid_id = id(centroid_data)  # Use object id as key
                matched_labels[centroid_id] = label if label is not None else centroid_data['label']


                # Check if gripper is closed - if so, clear bbox
                if not self.gripper_open:
                    bbox = None
                    # rospy.loginfo("Gripper closed - clearing bounding boxes")
                
                # Check if this is the highest confidence object
                is_highest_conf = (highest_conf_data and 
                                centroid_data['label'] == highest_conf_data['label'] and 
                                centroid_data['confidence'] == highest_conf_data['confidence'])
                
                # NEW: If this is the highest confidence, store the MATCHED label
                if is_highest_conf and label is not None:
                    highest_conf_matched_label = label

                

                # If the centroid is not inside any bounding box, skip drawing it
                DISPLAY_CENTROIDS_ANYWAY = True
                if bbox is None and DISPLAY_CENTROIDS_ANYWAY:
                    # Check if point is within image bounds
                    if 0 <= u < w and 0 <= v < h:
                        color = self.confidence_to_color(centroid_data['confidence'], min_conf, max_conf)
                        
                        # Draw circle at centroid location (ALL OBJECTS)
                        cv2.circle(image, (u, v), 5, color, -1, cv2.LINE_AA)
                        
                        # Draw confidence text (ALL OBJECTS)
                        text = f"{centroid_data['label']}: {centroid_data['confidence']:.2f}"
                        cv2.putText(image, text, (u - 15, v - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                        
                        # Draw only highest confidence object on sparse image
                        if is_highest_conf:
                            text_sparse = f"{centroid_data['label']}"
                            cv2.circle(sparse_image, (u, v), 6, color, -1, cv2.LINE_AA)
                            cv2.putText(sparse_image, text_sparse, (u + 15, v - 15), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                            highest_conf_plotted = True
                            # NEW: Store label even without bbox
                            if highest_conf_matched_label is None:
                                highest_conf_matched_label = centroid_data['label']
                        
                        points_plotted += 1
                        # rospy.loginfo(f"Successfully plotted point at ({u}, {v})")
                    continue

                # Otherwise, draw the bounding box and label onto the image
                color = self.confidence_to_color(centroid_data['confidence'], min_conf, max_conf)
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                
                # Draw on full overlay (ALL OBJECTS)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                # also plot centroid for debugging
                # cv2.circle(image, (u, v), 5, color, -1, cv2.LINE_AA)
                # label_text = f"{centroid_data['label']}: {centroid_data['confidence']:.2f}"
                label_text = f"{label}: {centroid_data['confidence']:.2f}"  # Use 'label' from bbox, not centroid_data['label']
                cv2.putText(image, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw only highest confidence object on sparse image
                if is_highest_conf:
                    cv2.rectangle(sparse_image, (x1, y1), (x2, y2), color, 2)  
                    # add label for debugging
                    # label_text = f"{centroid_data['label']}: {centroid_data['confidence']:.2f}"
                    # label_text = f"{label}: {centroid_data['confidence']:.2f}"  # Use 'label' from bbox, not centroid_data['label']
                    # cv2.putText(sparse_image, label_text, (x1, y1 - 10), 
                    #         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    highest_conf_plotted = True
                    # rospy.loginfo(f"Plotted highest confidence bbox: {centroid_data['label']} at ({x1},{y1})-({x2},{y2})")

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn_throttle(2.0, f"Transform failed: {e}")
                continue
        
        if highest_conf_data and highest_conf_matched_label:
            highest_label_msg = String()
            highest_label_msg.data = json.dumps({
                'label': highest_conf_matched_label,  # Use matched bbox label, not centroid label
                'confidence': highest_conf_data['confidence'],
                'gripper_open': self.gripper_open,
                'centroid': {
                    'x': highest_conf_data['centroid'].x,
                    'y': highest_conf_data['centroid'].y,
                    'z': highest_conf_data['centroid'].z
                }
            })
            self.highest_conf_label_pub.publish(highest_label_msg)

        # NEW: Publish all objects' labels and confidences
        all_objects_list = []
        for centroid_data in self.centroids:
            centroid_id = id(centroid_data)  # Get the ID for this centroid
            matched_label = matched_labels.get(centroid_id, centroid_data['label'])  # Look up the matched label
    
            all_objects_list.append({
                # 'label': centroid_data['label'],
                'label': matched_label,
                'confidence': centroid_data['confidence'],
                'gripper_open': self.gripper_open,
                'centroid': {
                    'x': centroid_data['centroid'].x,
                    'y': centroid_data['centroid'].y,
                    'z': centroid_data['centroid'].z
                }
            })
        if all_objects_list:
            all_objects_msg = String()
            all_objects_msg.data = json.dumps(all_objects_list)
            self.all_objects_pub.publish(all_objects_msg)
            
        
        # Publish overlay image (ALL OBJECTS)
        try:
            self.overlay_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
            # rospy.loginfo(f"Published overlay with {points_plotted}/{len(self.centroids)} centroids")
        except Exception as e:
            rospy.logwarn(f"Failed to publish overlay: {e}")
        
        # Publish sparse image (HIGHEST CONFIDENCE ONLY)
        try:
            self.sparse_pub.publish(self.bridge.cv2_to_imgmsg(sparse_image, "bgr8"))
            # rospy.loginfo(f"Published sparse overlay with highest confidence object: {highest_conf_plotted}")
        except Exception as e:
            rospy.logwarn(f"Failed to publish sparse overlay: {e}")
        

    def transform_point_manually(self, point_stamped, transform):
        """
        Manually transform a PointStamped using the transform matrix
        """
        # Extract translation and rotation from transform
        t = transform.transform.translation
        r = transform.transform.rotation
        
        # Convert quaternion to rotation matrix
        x, y, z, w = r.x, r.y, r.z, r.w
        # Normalize quaternion
        norm = np.sqrt(x*x + y*y + z*z + w*w)
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
        
        # Create rotation matrix from quaternion
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        
        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [t.x, t.y, t.z]
        
        # Transform the point
        original_point_vec = np.array([
            point_stamped.point.x,
            point_stamped.point.y,
            point_stamped.point.z,
            1.0
        ])
        
        transformed_vec = T @ original_point_vec
        
        # Create transformed PointStamped
        transformed_point = PointStamped()
        transformed_point.header.frame_id = transform.header.frame_id
        transformed_point.header.stamp = point_stamped.header.stamp
        transformed_point.point.x = transformed_vec[0]
        transformed_point.point.y = transformed_vec[1]
        transformed_point.point.z = transformed_vec[2]
        
        return transformed_point

if __name__ == '__main__':
    try:
        TransformExample()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass




