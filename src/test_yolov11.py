#!/usr/bin/env python3

import json
import rospy
from sensor_msgs.msg import Image
import numpy as np
import cv2
import torch
from std_msgs.msg import String, Int32
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
from ultralytics import YOLO
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
np.float = np.float64
import ros_numpy


class ObjectDetection:
    def __init__(self):

        self.subscriber_topics = {
           "color": "/camera/color/image_raw",
           "depth": "/camera/depth/color/points"
        }

        self.model = self.load_model()
        self.classes = self.model.names
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(self.device)

        self.publisher = rospy.Publisher('/num_objects', Int32, queue_size=10)
        self.image_pub = rospy.Publisher('/detected_objects/image', ROSImage, queue_size=10)
        self.label_pub = rospy.Publisher('/detected_objects/label', String, queue_size=10)
        self.latest_depth_pc = None

        rospy.Subscriber(self.subscriber_topics["color"], Image, self.get_frame)
        rospy.Subscriber(self.subscriber_topics["depth"], PointCloud2, self.depth_callback)

        self.full_image_pub = rospy.Publisher('/detected_objects/full_image', ROSImage, queue_size=10)
        self.dict_bb_pub = rospy.Publisher('/detected_objects/dict_label_centroid', String, queue_size=10)
        self.centroid_pub = rospy.Publisher('/detected_objects/centroid', Point, queue_size=10)
        self.bridge = CvBridge()
        self.detection_id_counter = 0
        
    def load_model(self):
        # Load YOLOv11n
        return YOLO("/home/kinovaresearch/catkin_workspace/src/trust_and_transparency/yolocustom.pt")

    def get_frame(self, data):
        # Use CvBridge for simplicity
        frame = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        self.process_image(frame)

    def depth_callback(self, msg: PointCloud2):
        self.latest_depth_pc = msg

    def process_image(self, frame):
        results = self.model(frame)  # YOLOv11 inference
        dets = results[0].boxes
        plot_labels = [
            "orange bottle", "blue bottle", "banana", "scissors", "ball", "paper cup", "mustard bottle", "soup can", "soda can", "pasta box"
        ]

        labels, cords = [], []
        for box in dets:
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
            conf = float(box.conf.cpu().numpy()[0])
            cls  = int(box.cls.cpu().numpy()[0])
            if self.class_to_label(cls) not in plot_labels:
                continue
            cords.append([x1 / frame.shape[1], y1 / frame.shape[0],
                          x2 / frame.shape[1], y2 / frame.shape[0],
                          conf])
            labels.append(cls)

        # Filter out boxes with strong overlap
        filtered_boxes = []
        filtered_labels = []
        for i, box in enumerate(cords):
            include = True
            for existing_box in filtered_boxes:
                if self.bb_intersection_over_union(box, existing_box) > 0.90:
                    include = False
                    break
            if include:
                filtered_boxes.append(box)
                filtered_labels.append(labels[i])

        frame_annotated, object_count = self.full_plot_boxes(filtered_labels, filtered_boxes, frame)

        # Show annotated frame
        cv2.imshow("Full Annotated Image", frame_annotated)
        cv2.waitKey(1)
        self.publisher.publish(object_count)


    def full_plot_boxes(self, labels, cords, frame):
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        object_count = 0

        try:
            depth_pc2 = self.latest_depth_pc
            if not depth_pc2:
                rospy.logwarn("[YOLO] No depth cloud received yet")
                return frame, object_count
        except rospy.ROSException:
            rospy.logwarn("[YOLO] Timeout waiting for depth cloud")
            return frame, object_count

        pc_arr = ros_numpy.numpify(depth_pc2)
        x_mat = pc_arr['x']
        y_mat = pc_arr['y']
        z_mat = pc_arr['z']
        all_info = []

        for i, row in enumerate(cords):
            if row[4] < 0.2:
                continue

            label = self.class_to_label(labels[i])
            if "table" in label.lower():
                continue

            object_count += 1
            detection_id = self.detection_id_counter
            self.detection_id_counter += 1

            x1 = int(row[0] * x_shape)
            y1 = int(row[1] * y_shape)
            x2 = int(row[2] * x_shape)
            y2 = int(row[3] * y_shape)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            xs = x_mat[y1:y2, x1:x2]
            ys = y_mat[y1:y2, x1:x2]
            zs = z_mat[y1:y2, x1:x2]
            valid = ~np.isnan(xs)
            if not np.any(valid):
                rospy.logwarn(f"No valid depth for detection {i}, skipping")
                continue

            pts3d = np.column_stack((xs[valid], ys[valid], zs[valid]))
            cam_centroid = np.round(pts3d.mean(axis=0), 3)

            info = {
                "id": detection_id,
                "label": label,
                "centroid": {
                    "x": float(cam_centroid[0]),
                    "y": float(cam_centroid[1]),
                    "z": float(cam_centroid[2])
                },
                "bounding_box": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                }
            }

            all_info.append(info)

        ros_full_img = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.full_image_pub.publish(ros_full_img)
        if all_info:
            self.dict_bb_pub.publish(String(data=json.dumps(all_info)))
        return frame, object_count
    

    def bb_intersection_over_union(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
        boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
        denom = boxAArea + boxBArea - interArea
        return interArea / denom if denom > 0 else 0.0

    def class_to_label(self, x):
        return self.classes[int(x)]

    def start(self):
        rospy.init_node('object_detection_node', anonymous=True)
        rospy.spin()
      

if __name__ == '__main__':
    try:
        od = ObjectDetection()
        od.start()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
