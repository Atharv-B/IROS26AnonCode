#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
np.float = np.float64  # temp fix for following import, revert when ros_numpy repo is updated
import ros_numpy as rnp
from kortex_driver.msg import BaseCyclic_Feedback
from ActuatorModel import ActuatorModel
from ForwardKinematics import ForwardKinematicsKinova
from std_msgs.msg import Header
from geometry_msgs.msg import Point32
from sensor_msgs.msg import PointCloud
from std_msgs.msg import String
import json


def convert_to_world(cloud_msg, fk):
    xyz = np.zeros((len(cloud_msg.points), 3), dtype=np.float64)
    for i, pt in enumerate(cloud_msg.points):
        xyz[i] = [pt.x, pt.y, pt.z]
    xyz = xyz.T  # Transpose to match expected shape (3, N)

    # Convert NP Tuple Array to Matrix for transformation
    world_coords = fk.camera_xyz_to_world(xyz)

    new_cloud_msg = PointCloud()
    new_cloud_msg.header = Header(stamp=rospy.Time.now(), frame_id="base_link")
    new_cloud_msg.points = [Point32(x=pt[0], y=pt[1], z=pt[2]) for pt in world_coords.T]
    
    print("publishing point cloud with {} points".format(len(new_cloud_msg.points)))
    return new_cloud_msg


class CameraToWorld:
    def __init__(self):
        self.curr_pos = None
        self.subscriber = rospy.Subscriber('/clusters_camera_frame', PointCloud, self.point_cloud_callback)
        self.publisher = rospy.Publisher('/clusters', PointCloud, queue_size=10)
        self.cam_to_word_for_trust_pub = rospy.Publisher('/centroid_to_world', String, queue_size=10)
        self.basefeedback = rospy.Subscriber("/" + 'my_gen3' + "/base_feedback", BaseCyclic_Feedback,
                                             self.base_feedback_callback, buff_size=1)
        
        self.label_centroid_sub = rospy.Subscriber(
            '/detected_objects/dict_label_centroid',
            String,
            self.dict_label_centroid_callback) 
        self.world_label_dict = {}
        self.publish_timer = rospy.Timer(
            rospy.Duration(0.1), self._publish_and_clear)
        
        self.positions = []
        self.fk_kinova = ForwardKinematicsKinova()
        rospy.Timer(rospy.Duration(0.5), self.update_joints)

    def base_feedback_callback(self, feedback):
        get_state = ActuatorModel()
        get_state.position_0 = feedback.actuators[0].position
        get_state.position_1 = feedback.actuators[1].position
        get_state.position_2 = feedback.actuators[2].position
        get_state.position_3 = feedback.actuators[3].position
        get_state.position_4 = feedback.actuators[4].position
        get_state.position_5 = feedback.actuators[5].position
        get_state.position_6 = feedback.actuators[6].position

        self.positions.append(get_state)
        self.curr_pos = get_state.get_position()


    def dict_label_centroid_callback(self, msg: String):
        """
        Handle both single dict and list formats from YOLO
        """
        try:
            data = json.loads(msg.data)
            
            # Handle list format (multiple detections)
            if isinstance(data, list):
                results = []
                for item in data:
                    result = self.process_single_detection(item)
                    if result:
                        results.append(result)
                
                # Publish the list of transformed detections
                if results:
                    output_msg = String(data=json.dumps(results))
                    self.cam_to_word_for_trust_pub.publish(output_msg)  # Fixed typo here
            
            # Handle single dict format
            elif isinstance(data, dict):
                result = self.process_single_detection(data)
                if result:
                    output_msg = String(data=json.dumps([result]))  # Wrap in list for consistency
                    self.cam_to_word_for_trust_pub.publish(output_msg)  # Fixed typo here
                    
        except Exception as e:
            rospy.logerr(f"[CamToWorld] Error processing detection: {e}")


    def process_single_detection(self, data):
        """Process a single detection dict"""
        try:
            label = data.get("label", "")
            cen = data.get("centroid", {})
            bbox = data.get("bounding_box", {})
            detection_id = data.get("id", 0)
            
            cam_x = cen.get("x")
            cam_y = cen.get("y") 
            cam_z = cen.get("z")
            
            if None in (cam_x, cam_y, cam_z):
                rospy.logwarn(f"[CamToWorld] Incomplete centroid for {label}")
                return None
                
            # Transform coordinates
            cam_pt = np.array([[cam_x], [cam_y], [cam_z]])
            world_pt = self.fk_kinova.camera_xyz_to_world(cam_pt).T[0]
            
            # Apply offsets
            wx = float(world_pt[0] + 0.025)
            wy = float(world_pt[1] - 0.01) 
            wz = float(world_pt[2])
            
            
            return {
                "label": label,
                "centroid": {"x": wx, "y": wy, "z": wz},
                "bounding_box": bbox,
                "id": detection_id
            }
            
        except Exception as e:
            rospy.logerr(f"[CamToWorld] Failed to transform detection: {e}")
            return None
    
    def _publish_and_clear(self, event):
        if not self.world_label_dict:
            return
        # publish the snapshot
        j = json.dumps(self.world_label_dict)
        self.cam_to_word_for_trust_pub.publish(String(data=j))
        # now clear for the next frame
        self.world_label_dict.clear()

    def point_cloud_callback(self, cloud_msg):
        """
        Callback Function for point cloud messages.
        """
        filtered_cloud = convert_to_world(cloud_msg, self.fk_kinova)
        self.publisher.publish(filtered_cloud)

    def get_curr_pos(self):
        if len(self.positions) == 0:
            return 0

        actuatorModel = ActuatorModel()
        curr_pos = self.positions.pop(len(self.positions)-1)
        joint_angles = actuatorModel.setActuatorData(curr_pos.position_0, curr_pos.position_1,
                                            curr_pos.position_2, curr_pos.position_3,
                                            curr_pos.position_4, curr_pos.position_5,
                                            curr_pos.position_6)

        return joint_angles

    def publish_world_coordinates(self, coords):
        centroid_cloud = PointCloud()
        centroid_cloud.header = Header(stamp=rospy.Time.now(), frame_id="base_link")
        centroid_cloud.points = [Point32(x=c[0], y=c[1], z=c[2]) for c in coords]
        # self.publisher.publish(centroid_cloud)
        self.cam_to_word_for_trust_pub.publish(centroid_cloud)


    def update_joints(self, event):
        joint_angles_deg = self.get_curr_pos()
        joint_angles = np.radians(joint_angles_deg)
        # print("Joint Angles ", joint_angles)
        self.fk_kinova.update_joints(joint_angles)


if __name__ == '__main__':
    rospy.init_node('camera_to_world')
    bg_filter = CameraToWorld()
    rospy.spin()
