#!/usr/bin/env python3

import rospy
import tf2_ros
import numpy as np
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, Point
from sensor_msgs.msg import Image, CameraInfo
from image_geometry import PinholeCameraModel
from trust_and_transparency.msg import CentroidConfidenceArray
from std_msgs.msg import String
from constants import ORACLE_GOAL_SET

class TransformExample:
    def __init__(self):
        rospy.init_node('scene_centroid_viz', anonymous=True)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Initialize camera model and image handling
        self.bridge = CvBridge()
        self.cam_model = PinholeCameraModel()
        self.have_cam_info = False
        self.cam_frame = None
        self.latest_image = None
        self.centroids = []
        
        # Centroid stabilization
        self.stable_centroids = {}  # Dictionary to store stable centroid positions
        self.stable_pixels = {}     # Dictionary to store stable pixel positions
        self.movement_threshold = rospy.get_param('~movement_threshold', 0.03)  # 3cm threshold
        self.pixel_threshold = rospy.get_param('~pixel_threshold', 5)  # 5 pixel threshold
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.3)  # Only stabilize confident detections
        
        # Calibration offsets (adjust these to fine-tune centroid positions)
        self.offset_x = rospy.get_param('~offset_x', 0.15)
        self.offset_y = rospy.get_param('~offset_y', -0.01)
        self.offset_z = rospy.get_param('~offset_z', 0.0)
        
        # Publishers and subscribers
        self.overlay_pub = rospy.Publisher('/scene_camera/centroid_overlay', Image, queue_size=1)
        
        # Subscribe to camera info and image
        rospy.Subscriber('/scene_camera/camera/color/camera_info', CameraInfo, self.camera_info_callback)
        rospy.Subscriber('/scene_camera/camera/color/image_raw', Image, self.image_callback)
        
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
                rospy.logdebug(f"Using stable position for {label} (movement: {distance:.3f}m)")
                centroid_data['centroid'] = stable_pos
                return centroid_data
            else:
                rospy.loginfo(f"Object {label} moved {distance:.3f}m, updating stable position")
        
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
                rospy.logdebug(f"Using stable pixel position for {label} (movement: {pixel_distance:.1f}px)")
                return stable_u, stable_v
            else:
                rospy.loginfo(f"Object {label} moved {pixel_distance:.1f}px, updating stable pixel position")
        
        # Update stable pixel position for this label
        self.stable_pixels[label] = (u, v)
        return u, v

    def confidence_to_color(self, confidence):
        """Convert confidence to BGR color (red=low, green=high)"""
        # Map confidence [0,1] to color
        green = int(confidence * 255)
        red = int((1.0 - confidence) * 255)
        return (0, green, red)  # BGR format

    def draw_and_publish(self):
        """Transform centroids and draw them on the image"""
        if not (self.have_cam_info and self.latest_image is not None and self.centroids):
            rospy.loginfo_throttle(2.0, f"Waiting for: cam_info={self.have_cam_info}, image={self.latest_image is not None}, centroids={len(self.centroids) if self.centroids else 0}")
            return
            
        image = self.latest_image.copy()
        h, w = image.shape[:2]
        
        points_plotted = 0
        for centroid_data in self.centroids:
            try:
                # Create PointStamped from centroid
                point_stamped = PointStamped()
                point_stamped.header.frame_id = "base_link"  # Centroids should be in base_link after cam_to_world transformation
                point_stamped.header.stamp = rospy.Time.now()
                point_stamped.point = centroid_data['centroid']
                
                # Apply calibration offsets for fine-tuning
                point_stamped.point.x += self.offset_x
                point_stamped.point.y += self.offset_y
                point_stamped.point.z += self.offset_z
                
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
                    # rospy.logwarn(f"Point behind camera: Z={Z:.3f}")
                    continue
                    
                u, v = self.cam_model.project3dToPixel((X, Y, Z))
                u, v = int(round(u)), int(round(v))
                
                # Apply pixel-level stabilization
                u, v = self.stabilize_pixel_position(centroid_data['label'], u, v, centroid_data['confidence'])
                
                # rospy.loginfo(f"Projected to pixel: u={u}, v={v}")
                
                # Check if point is within image bounds
                if 0 <= u < w and 0 <= v < h:
                    color = self.confidence_to_color(centroid_data['confidence'])
                    
                    # Draw circle at centroid location
                    cv2.circle(image, (u, v), 12, color, -1, cv2.LINE_AA)
                    
                    # Draw confidence text
                    text = f"{centroid_data['label']}: {centroid_data['confidence']:.2f}"
                    cv2.putText(image, text, (u + 15, v - 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                    
                    points_plotted += 1
                    rospy.loginfo(f"Successfully plotted point at ({u}, {v})")
                else:
                    rospy.logwarn(f"Point outside image bounds: ({u}, {v}) - image size: {w}x{h}")
                    
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn_throttle(2.0, f"Transform failed: {e}")
                continue
        
        # Publish overlay image
        try:
            self.overlay_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
            rospy.loginfo(f"Published overlay with {points_plotted}/{len(self.centroids)} centroids")
        except Exception as e:
            rospy.logwarn(f"Failed to publish overlay: {e}")
        
        # Display image (optional, for debugging)
        cv2.imshow("Scene Centroid Overlay", image)
        # cv2.waitKey(1)

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





# import rospy, cv2, numpy as np
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image, CameraInfo
# from image_geometry import PinholeCameraModel
# import tf2_ros

# from trust_and_transparency.msg import CentroidConfidenceArray

# def quat_to_rot(qx,qy,qz,qw):
#     n = np.sqrt(qx*qx+qy*qy+qz*qz+qw*qw) or 1.0
#     qx,qy,qz,qw = qx/n,qy/n,qz/n,qw/n
#     xx,yy,zz = qx*qx,qy*qy,qz*qz
#     xy,xz,yz = qx*qy,qx*qz,qy*qz
#     wx,wy,wz = qw*qx,qw*qy,qw*qz
#     return np.array([
#         [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
#         [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
#         [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
#     ], dtype=np.float64)

# def T_from_tf(tfmsg):
#     t,q = tfmsg.transform.translation, tfmsg.transform.rotation
#     T = np.eye(4, dtype=np.float64)
#     T[:3,:3] = quat_to_rot(q.x,q.y,q.z,q.w)
#     T[:3, 3] = [t.x, t.y, t.z]
#     return T

# class SceneCentroidViz:
#     def __init__(self):
#         rospy.init_node("scene_centroid_viz", anonymous=True)
#         self.src_frame = rospy.get_param("~src_frame", "base_link")
#         self.show_window = rospy.get_param("~show_window", True)

#         self.bridge = CvBridge()
#         self.cam_model = PinholeCameraModel()
#         self.have_cam_info = False
#         self.cam_frame = None
#         self.last_img = None
#         self.last_img_stamp = rospy.Time(0)
#         self.items = []

#         self.tfbuf = tf2_ros.Buffer(rospy.Duration(10.0))
#         self.tfl  = tf2_ros.TransformListener(self.tfbuf)

#         self.pub = rospy.Publisher("/scene_camera/centroid_overlay", Image, queue_size=1)
#         rospy.Subscriber("/scene_camera/camera/color/camera_info", CameraInfo, self._info_cb,  queue_size=1)
#         rospy.Subscriber("/scene_camera/camera/color/image_raw",  Image,      self._image_cb, queue_size=1)
#         rospy.Subscriber("/goal_confidence_centroids", CentroidConfidenceArray, self._cent_cb, queue_size=1)
        
#         # Add debug timer
#         rospy.Timer(rospy.Duration(10.0), self._debug_status)

#     def _debug_status(self, event):
#         """Debug method to log current status"""
#         rospy.loginfo("Status: cam_info=%s, cam_frame=%s, items=%d, last_img=%s", 
#                      self.have_cam_info, self.cam_frame, len(self.items), 
#                      self.last_img is not None)
        
#         if self.items:
#             rospy.loginfo("First centroid: [%.3f, %.3f, %.3f]", 
#                          self.items[0].centroid.x, self.items[0].centroid.y, self.items[0].centroid.z)

#     def _info_cb(self, msg):
#         if self.have_cam_info: return
#         self.cam_model.fromCameraInfo(msg)
#         self.cam_frame = msg.header.frame_id   # "static_camera_color_optical_frame"
#         self.have_cam_info = True
#         rospy.loginfo("Projecting into %s", self.cam_frame)

#     def _image_cb(self, msg):
#         try:
#             self.last_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
#             self.last_img_stamp = msg.header.stamp
#             self._draw_and_publish()
#         except Exception as e:
#             rospy.logwarn("Image convert failed: %s", e)

#     def _cent_cb(self, msg):
#         self.items = msg.items
        
#         # Get the frame from the message header
#         if hasattr(msg, 'header') and hasattr(msg.header, 'frame_id'):
#             self.centroid_msg_frame = msg.header.frame_id
#             rospy.loginfo("Centroids received in frame: %s", self.centroid_msg_frame)
#         else:
#             rospy.logwarn("CentroidConfidenceArray message has no header.frame_id!")
            
#         if self.items:
#             rospy.loginfo("Received %d centroids. First: [%.3f, %.3f, %.3f], Last: [%.3f, %.3f, %.3f]", 
#                          len(self.items), 
#                          self.items[0].centroid.x, self.items[0].centroid.y, self.items[0].centroid.z,
#                          self.items[-1].centroid.x, self.items[-1].centroid.y, self.items[-1].centroid.z)


#     @staticmethod
#     def _color(c):  # green→red by confidence
#         c = max(0.0, min(1.0, float(c)))
#         return (0, int(c*255), int((1.0-c)*255))

#     def _lookup_T_cam_from_centroids(self):
#         if not self.cam_frame:
#             raise RuntimeError("CameraInfo not received")
            
#         # Use the frame from message header
#         actual_centroid_frame = getattr(self, 'centroid_msg_frame', self.src_frame)
            
#         rospy.loginfo("Looking up transform: %s -> %s", actual_centroid_frame, self.cam_frame)
        
#         try:
#             tfmsg = self.tfbuf.lookup_transform(self.cam_frame, actual_centroid_frame,
#                                                 rospy.Time(0), rospy.Duration(1.0))
#             T = T_from_tf(tfmsg)
#             rospy.loginfo("Transform matrix (first row): [%.3f, %.3f, %.3f, %.3f]", 
#                          T[0,0], T[0,1], T[0,2], T[0,3])
#             return T
#         except Exception as e:
#             rospy.logerr("Transform lookup failed: %s", e)
#             raise

#     def _draw_and_publish(self):
#         if not (self.have_cam_info and self.last_img is not None and self.items):
#             rospy.loginfo_throttle(5.0, "Waiting for: cam_info=%s, img=%s, items=%d", 
#                                  self.have_cam_info, self.last_img is not None, len(self.items))
#             return
#         img = self.last_img.copy()
#         h, w = img.shape[:2]
#         rospy.loginfo("Image size: %dx%d", w, h)

#         try:
#             T = self._lookup_T_cam_from_centroids()
#         except Exception as e:
#             rospy.logerr_throttle(2.0, "TF failed: %s", e)
#             return

#         points_plotted = 0
#         for i, it in enumerate(self.items):
#             rospy.loginfo("=== Item %d ===", i)
#             rospy.loginfo("Original centroid: [%.3f, %.3f, %.3f]", 
#                           it.centroid.x, it.centroid.y, it.centroid.z)
            
#             p = np.array([it.centroid.x, it.centroid.y, it.centroid.z, 1.0], dtype=np.float64)
#             X, Y, Z = (T @ p)[:3]
            
#             rospy.loginfo("Transformed to camera: [%.3f, %.3f, %.3f]", X, Y, Z)
            
#             if not np.isfinite([X,Y,Z]).all():
#                 rospy.logwarn("Non-finite coordinates after transform")
#                 continue
                
#             if Z <= 0:
#                 rospy.loginfo("Behind camera (Z=%.3f)", Z)
#                 continue
                
#             u, v = self.cam_model.project3dToPixel((X, Y, Z))
#             u, v = int(round(u)), int(round(v))
            
#             rospy.loginfo("Projected to pixel: [%d, %d]", u, v)
            
#             if 0 <= u < w and 0 <= v < h:
#                 color = self._color(getattr(it, "confidence", 0.0))
#                 cv2.circle(img, (u, v), 12, color, -1, cv2.LINE_AA)  # Made circle bigger
#                 cv2.putText(img, f"{getattr(it,'confidence',0.0):.2f}",
#                             (u+15, max(0, v-15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
#                 points_plotted += 1
#                 rospy.loginfo("SUCCESS: Plotted at [%d, %d]", u, v)
#             else:
#                 rospy.loginfo("REJECTED: Outside bounds [%d, %d] (image: %dx%d)", u, v, w, h)

#         rospy.loginfo("=== SUMMARY: Plotted %d/%d centroids ===", points_plotted, len(self.items))

#         try:
#             self.pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
#             rospy.loginfo("Published overlay image with %d objects", points_plotted)
#         except Exception as e:
#             rospy.logwarn("Publish failed: %s", e)

#         if self.show_window:
#             cv2.imshow("Scene Centroid Overlay", img)
#             cv2.waitKey(1)

#     def spin(self):
#         rospy.spin()
#         if self.show_window: cv2.destroyAllWindows()

#     def inverse_T(self,T):
#         """
#         Compute the inverse of a 4x4 homogeneous transformation matrix.
        
#         For a transformation matrix T = [R t; 0 1] where R is 3x3 rotation and t is 3x1 translation:
#         T^-1 = [R^T -R^T*t; 0 1]
        
#         Args:
#             T: 4x4 numpy array representing homogeneous transformation matrix
            
#         Returns:
#             T_inv: 4x4 numpy array representing the inverse transformation
#         """
#         T_inv = np.eye(4, dtype=np.float64)
        
#         # Extract rotation matrix (3x3) and translation vector (3x1)
#         R = T[:3, :3]
#         t = T[:3, 3]
        
#         # Inverse rotation is transpose for orthogonal matrices
#         R_inv = R.T
        
#         # Inverse translation is -R^T * t
#         t_inv = -R_inv @ t
        
#         # Construct inverse transformation matrix
#         T_inv[:3, :3] = R_inv
#         T_inv[:3, 3] = t_inv
        
#         return T_inv
# if __name__ == "__main__":
#     SceneCentroidViz().spin()
