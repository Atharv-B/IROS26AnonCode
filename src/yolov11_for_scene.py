#!/usr/bin/env python3

import json
import rospy
from sensor_msgs.msg import Image
import numpy as np
import cv2
import torch
from std_msgs.msg import String
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
from std_msgs.msg import Int32
from ultralytics import YOLO
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
np.float = np.float64
import ros_numpy


class ObjectDetection:
    def __init__(self):

        self.subscriber_topics = {
           "color": "/scene_camera/camera/color/image_raw",
           "depth": "/scene_camera/camera/depth/color/points"
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
        # Move model to device once
        self.model.to(self.device)

        self.publisher = rospy.Publisher('/scene_camera/num_objects', Int32, queue_size=10)
        self.image_pub = rospy.Publisher('/scene_camera/detected_objects/image', ROSImage, queue_size=10)
        self.label_pub = rospy.Publisher('/scene_camera/detected_objects/label', String, queue_size=10)
        self.latest_depth_pc = None

        rospy.Subscriber(self.subscriber_topics["color"], Image, self.get_frame)
        rospy.Subscriber(self.subscriber_topics["depth"], PointCloud2, self.depth_callback)

        self.full_image_pub = rospy.Publisher('/scene_camera/detected_objects/full_image', ROSImage, queue_size=10)
        self.dict_bb_pub = rospy.Publisher('/scene_camera/detected_objects/dict_label_centroid', String, queue_size=10)
        self.centroid_pub = rospy.Publisher('/scene_camera/detected_objects/centroid', Point, queue_size=10)
        self.bridge = CvBridge()
        self.detection_id_counter = 0
        
    def load_model(self):
        return YOLO("/home/kinovaresearch/catkin_workspace/src/trust_and_transparency/yolocustom.pt")  # Load your custom-trained YOLOv11 model

    def get_frame(self, data):
        # Keep your original manual conversion path
        if data.encoding == "rgb8":
            conversion_format = "RGB8"
        elif data.encoding == "bgr8":
            conversion_format = "BGR8"
        else:
            raise ValueError(f"Unsupported encoding {data.encoding}")

        np_arr = np.frombuffer(data.data, dtype=np.uint8)
        image_np = np_arr.reshape(data.height, data.width, -1)

        if conversion_format == "RGB8":
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        self.process_image(image_np)

    def depth_callback(self, msg: PointCloud2):
        self.latest_depth_pc = msg

    def process_image(self, frame):
        # YOLOv11 inference
        results = self.model(frame)
        boxes = results[0].boxes

        # Build labels and normalized coords (x1,y1,x2,y2,conf) to match your downstream logic
        labels, cords = [], []
        h, w = frame.shape[0], frame.shape[1]
        for b in boxes:
            x1, y1, x2, y2 = b.xyxy.cpu().numpy()[0]
            conf = float(b.conf.cpu().numpy()[0])
            cls  = int(b.cls.cpu().numpy()[0])
            # normalize to 0..1 like your original code expected from xyxyn
            cords.append([x1 / w, y1 / h, x2 / w, y2 / h, conf])
            labels.append(cls)

        # Filter out boxes with strong overlap (keep your logic)
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

        cv2.imshow("Full Annotated Image", frame_annotated)
        cv2.waitKey(1)
        self.publisher.publish(object_count)


    def full_plot_boxes(self, labels, cords, frame):
        """
        Draw 2D boxes, compute true 3D centroids from the depth cloud,
        and publish JSON on self.dict_pub.
        """
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        object_count = 0

        # 1) grab a single depth PointCloud2
        try:
            depth_pc2 = self.latest_depth_pc
            if not depth_pc2:
                rospy.logwarn("[YOLO] No depth cloud received yet")
                return frame, object_count
        except rospy.ROSException:
            rospy.logwarn("[YOLO] Timeout waiting for depth cloud")
            return frame, object_count

        # 2) convert to structured array H×W with fields ['x','y','z']
        pc_arr = ros_numpy.numpify(depth_pc2)
        x_mat = pc_arr['x']
        y_mat = pc_arr['y']
        z_mat = pc_arr['z']
        all_info = []

        for i, row in enumerate(cords):
            # row = [x_min_norm, y_min_norm, x_max_norm, y_max_norm, score]
            if row[4] < 0.2:
                continue
                
            # Filter out labels containing selected words
            filter_words = ["table", "chair", "airplane", "sink", "laptop", "keyboard", "mouse",
                            "tvmonitor", "cell phone", "book", "traffic light", "toilet", "tv"]
            label = self.class_to_label(labels[i])
            if any(word in label.lower() for word in filter_words):
                continue
                
            object_count += 1
            detection_id = self.detection_id_counter
            self.detection_id_counter += 1

            # 3) 2D box coords (convert normalized → pixels)
            x1 = int(row[0] * x_shape)
            y1 = int(row[1] * y_shape)
            x2 = int(row[2] * x_shape)
            y2 = int(row[3] * y_shape)

             # --- NEW: drop abnormally large boxes for small objects like cups ---
            box_area = (x2 - x1) * (y2 - y1)
            frame_area = x_shape * y_shape
            area_ratio = box_area / frame_area

            # Only apply to cup-like labels
            if "cup" in label and area_ratio > 0.12:  # 12% of image area → too big
                rospy.loginfo(f"Skipping oversized '{label}' box (area ratio {area_ratio:.2f})")
                continue
            # --------------------------------------------------------------------

            # 4) draw box & label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            # 5) slice out the 3D crop
            xs = x_mat[y1:y2, x1:x2]
            ys = y_mat[y1:y2, x1:x2]
            zs = z_mat[y1:y2, x1:x2]
            valid = ~np.isnan(xs)
            if not np.any(valid):
                rospy.logwarn(f"No valid depth for detection {i}, skipping")
                continue

            pts3d = np.column_stack((xs[valid], ys[valid], zs[valid]))  # N×3
            cam_centroid = np.round(pts3d.mean(axis=0), 3)

            # 6) publish the JSON
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

        # Publish the full annotated image
        ros_full_img = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.full_image_pub.publish(ros_full_img)
        if all_info:
            self.dict_bb_pub.publish(String(data=json.dumps(all_info)))
        return frame, object_count
    

    def bb_intersection_over_union(self, boxA, boxB):
        # Inputs are normalized coords; no +1 terms
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
        boxBArea = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))
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
