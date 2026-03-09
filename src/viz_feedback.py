#!/usr/bin/env python3

import time
import threading
import argparse

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from trust_and_transparency.msg import CentroidConfidenceArray, CentroidConfidence
from auditory.tts import TextToSpeech
from std_msgs.msg import String 
import json
from enum import Enum
from user_study.audio_logger import AudioLogger
from highest_confidence_object_viewer import HighestConfidenceViewer
from user_study.user_study import UserStudyExperiment
# from auditory.auditory_rich import VerbalFeedback

class FEEDBACK(Enum):
    VISUAL_SPARSE = 0
    VISUAL_RICH = 1
    VERBAL_SPARSE = 2
    VERBAL_RICH = 3
    
class VERBAL_GRANULARITY(Enum):
    DING = 0
    NUM_OBJECTS = 1
    CONFIDENCE_ONLY = 2
    CONFIDENCE_GATED_ASSERTION = 3
    CONFIDENCE_GATED_INTENT_DECLARATION = 4
    FULL_VERBOSE = 5
    LABEL_ONLY = 6  # Added new granularity level

class FeedbackVisualizer:
    def __init__(self):
        rospy.init_node('visual_feedback_mode', anonymous=True)
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Visual Feedback Visualizer Node')
        # parser.add_argument('--feedback_type', type=str, default='VISUAL_RICH',
        #                 choices=['VISUAL_SPARSE', 'VISUAL_RICH', 'VERBAL_SPARSE', 'VERBAL_RICH'],
        #                 help='Type of feedback to provide')
        
        self.task = rospy.get_param("~task", "shelving")  # Default to "shelving" task
        self.treatment = rospy.get_param("~treatment", "A") # Default to "A" treatment
        
        # Parse known args to allow ROS arguments to pass through
        args, unknown = parser.parse_known_args()
        
        # Convert string to enum
        feedback_type_map = {
            'VISUAL_SPARSE': FEEDBACK.VISUAL_SPARSE,
            'VISUAL_RICH': FEEDBACK.VISUAL_RICH,
            'VERBAL_SPARSE': FEEDBACK.VERBAL_SPARSE,
            'VERBAL_RICH': FEEDBACK.VERBAL_RICH
        }
        # self.feedback_type = feedback_type_map[args.feedback_type]
        self.feedback_type = feedback_type_map[rospy.get_param("~feedback_type", "VISUAL_RICH")]
        rospy.loginfo(f"Feedback type set to: {self.feedback_type.name}")
        
        self.last_audio_message = None
        self.sparse_verbal_granularity = VERBAL_GRANULARITY.LABEL_ONLY
        self.rich_verbal_granularity = VERBAL_GRANULARITY.LABEL_ONLY

        self.subscriber = rospy.Subscriber('/goal_confidence_centroids', CentroidConfidenceArray, self.callback)
        self.subscriber = rospy.Subscriber(
            '/goal_label_confidence', 
            String, 
            self.goal_label_callback
        )
        self.latest_feedback_time = 0
        self.is_speaking = False
        self.overlay_pub = rospy.Publisher('/all_objects_confidences_image', Image, queue_size=1)


        self.bridge = CvBridge()
        self.latest_image = None
        self.audio_logger = AudioLogger()
        self.all_labels = []
        self.all_confidences = []
        
        # Label freezing mechanism
        self.frozen_labels = {}  # position_key -> {'label': str, 'confidence': float, 'freeze_time': float}
        self.position_tolerance = 0.05  # 5cm tolerance for position matching
        self.freeze_threshold = 0.4  # Only freeze labels with high confidence
        self.last_freeze_time = 0
        self.freeze_cooldown = 1.0  # Wait 1 second between freezes

        # IMPORTANT: Use the same image source as scene_bb_from_centroids.py
        rospy.Subscriber('/scene_camera/camera/color/image_raw', Image, self.image_callback)
        
        self.image_pub = rospy.Publisher('/visual_sparse/highest_confidence_object/image', Image, queue_size=10)

        # Store highest confidence label from scene_bb_from_centroids
        self.scene_highest_conf_label = None
        self.scene_highest_conf_value = 0.0
        self.scene_gripper_open = True
        self.scene_highest_centroid = None
        self.all_data = None
        
        # Subscribe to highest confidence label from scene_bb_from_centroids
        rospy.Subscriber('/highest_confidence_label', String, self.highest_conf_label_callback)

        # Subscribe to all confidence labels for verbal rich feedback
        rospy.Subscriber('/all_objects_labels_confidences', String, self.all_objects_labels_callback)
    
    def all_objects_labels_callback(self, msg: String):
        """Receive all object labels and confidences from scene_bb_from_centroids.py"""
        try:
            self.data = json.loads(msg.data)
            self.all_labels = [it['label'] for it in self.data]
            self.all_confidences = [it['confidence'] for it in self.data]

            # rospy.loginfo_throttle(5.0, f"Received all labels: {self.all_labels} with confidences: {self.all_confidences}")
        except json.JSONDecodeError as e:
            rospy.logwarn(f"Failed to parse JSON from /all_objects_labels_confidences: {e}")

    def highest_conf_label_callback(self, msg: String):
        """Receive highest confidence label from scene_bb_from_centroids.py"""
        try:
            data = json.loads(msg.data)
            self.scene_highest_conf_label = data.get('label')
            self.scene_highest_conf_value = data.get('confidence', 0.0)
            self.scene_gripper_open = data.get('gripper_open', True)
            self.scene_highest_centroid = data.get('centroid')

            print(f"all labels: {data}")
            
            rospy.loginfo_throttle(1.0, f"Received highest conf from scene_bb: {self.scene_highest_conf_label} ({self.scene_highest_conf_value:.2f})")
        except json.JSONDecodeError as e:
            rospy.logwarn(f"Failed to parse JSON from /highest_confidence_label: {e}")

    def get_position_key(self, centroid):
        """Create a position-based key for spatial lookup"""
        # Use 10cm grid for position matching
        return f"{int(centroid.x / self.position_tolerance)}_{int(centroid.y / self.position_tolerance)}_{int(centroid.z / self.position_tolerance)}"

    def freeze_high_confidence_labels(self, centroids, confidences, labels):
        """Freeze labels for high-confidence detections"""
        current_time = rospy.get_time()
        
        # Only freeze periodically to avoid constant updates
        if current_time - self.last_freeze_time < self.freeze_cooldown:
            return
        
        for centroid, confidence, label in zip(centroids, confidences, labels):
            # if confidence > self.freeze_threshold:
            # Freeze based on argmax confidence
            if confidence:

                pos_key = self.get_position_key(centroid)
                
                # Only freeze if we don't have this position or if new confidence is much higher
                if (pos_key not in self.frozen_labels or 
                    confidence > self.frozen_labels[pos_key]['confidence'] + 0.15):
                    
                    self.frozen_labels[pos_key] = {
                        'label': label,
                        'confidence': confidence,
                        'centroid': centroid,
                        'freeze_time': current_time
                    }
                    # rospy.loginfo(f"Froze label '{label}' at position {pos_key} with confidence {confidence:.2f}")
        
        self.last_freeze_time = current_time

    def get_stable_label_for_highest_confidence(self, centroids, confidences, labels):
        """Get stable label for the highest confidence detection"""
        if not centroids:
            return None, None
        
        # Find highest confidence detection
        highest_conf = max(confidences)
        highest_idx = confidences.index(highest_conf)
        highest_centroid = centroids[highest_idx]
        current_label = labels[highest_idx]
        
        # Look for nearby frozen labels
        best_match = None
        min_distance = float('inf')
        
        for frozen_key, frozen_data in self.frozen_labels.items():
            # Calculate distance to frozen position
            frozen_centroid = frozen_data['centroid']
            dist = np.sqrt(
                (highest_centroid.x - frozen_centroid.x)**2 +
                (highest_centroid.y - frozen_centroid.y)**2 +
                (highest_centroid.z - frozen_centroid.z)**2
            )
            
            # If within tolerance and closer than previous matches
            if dist < self.position_tolerance * 1.5 and dist < min_distance:
                min_distance = dist
                best_match = frozen_data
        
        if best_match:
            # rospy.loginfo(f"Using frozen label '{best_match['label']}' for current detection '{current_label}' (distance: {min_distance:.3f}m)")
            return best_match['label'], highest_conf
        else:
            return current_label, highest_conf

    def ee_cam_image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"Image conversion failed: {e}")

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"Image conversion failed: {e}")

    def confidence_to_color(self, conf):
        green = int(conf * 255)
        red = int((1.0 - conf) * 255)
        return (0, green, red)  # BGR
    

    def confidence_to_color_normalized(self, conf, all_confidences):
        """Map confidence to color with normalization across all current confidences"""
        if len(all_confidences) <= 1:
            return (0, 255, 0)  # Green if only one object
        
        min_conf = min(all_confidences)
        max_conf = max(all_confidences)
        
        if max_conf == min_conf:
            return (0, 255, 0)  # Green if all same confidence
        
        # Normalize confidence relative to current range
        normalized_conf = (conf - min_conf) / (max_conf - min_conf)
        red = int((1.0 - normalized_conf) * 255)
        green = int(normalized_conf * 255)
        
        return (0, green, red)  # BGR

    def goal_label_callback(self, msg: String):
        try:
            items = json.loads(msg.data)
        except json.JSONDecodeError:
            rospy.logwarn("Failed to parse JSON from /goal_label_confidence")
            return

        n = len(items)
        # print(f"Received feedback for human with {n} items.")

        # pull out labels & confidences
        labels      = [it['label']      for it in items]
        confidences = [it['confidence'] for it in items]

    def callback(self, msg):
        # Extract centroids and confidences
        centroids = [item.centroid for item in msg.items]
        confidences = [item.confidence for item in msg.items]
        labels = [item.label for item in msg.items]
        gripper_states = [item.gripper_open for item in msg.items]
        bboxes = [{'x1': item.x1, 'y1': item.y1, 'x2': item.x2, 'y2': item.y2} for item in msg.items]

        # Dispatch based on selected feedback type
        if self.feedback_type == FEEDBACK.VISUAL_SPARSE:
            self.visual_sparse(centroids, confidences, labels, bboxes)
            # rospy.loginfo("Running visual sparse feedback.")
        elif self.feedback_type == FEEDBACK.VISUAL_RICH:
            self.visual_rich(centroids, confidences, labels, bboxes)
            # rospy.loginfo("Running visual rich feedback.")
        elif self.feedback_type == FEEDBACK.VERBAL_SPARSE:
            self.verbal_sparse(centroids, confidences, labels, gripper_states)
            # rospy.loginfo("Running verbal sparse feedback.")
        elif self.feedback_type == FEEDBACK.VERBAL_RICH:
            self.verbal_rich(centroids, confidences, labels, gripper_states)
            # rospy.loginfo("Running verbal rich feedback.")

    def visual_sparse(self, centroids, confidences, labels, bboxes):
        
        if self.latest_image is None:
            rospy.logwarn("No image available for visual sparse feedback.")
            return

        dets = list(zip(centroids, confidences, labels, bboxes))
        if not dets:
            rospy.logwarn("No detections available for visual sparse feedback.")
            return

        # Find the object with highest confidence
        best_centroid, best_conf, best_label, best_bbox = max(dets, key=lambda t: t[1])

        # Start with a clean image (same as scene_bb_from_centroids.py uses)
        image = self.latest_image.copy()
        
        # Use the same color method as visual_rich for consistency
        color = self.confidence_to_color_normalized(best_conf, confidences)
        
        # Draw bounding box using the same logic as scene_bb_from_centroids.py
        if best_bbox and best_bbox['x2'] > best_bbox['x1'] and best_bbox['y2'] > best_bbox['y1']:
            # print(f"Drawing bbox for highest confidence {best_label} with confidence {best_conf:.2f}")
            x1, y1, x2, y2 = best_bbox['x1'], best_bbox['y1'], best_bbox['x2'], best_bbox['y2']
            
            # Draw bounding box (thicker line for prominence)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            
            # Draw confidence and label
            text = f"{best_label}: {best_conf:.2f}"
            cv2.putText(image, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        else:
            rospy.logwarn(f"Invalid bbox for highest confidence {best_label}: {best_bbox}")
            return

        # Publish the sparse feedback image with only one bounding box
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(image, encoding='bgr8'))
            # rospy.loginfo(f"Published sparse feedback with highest confidence object: {best_label} with confidence {best_conf:.2f}")
        except Exception as e:
            rospy.logerr("Failed to publish visual sparse image: %s", e)

        # cv2.imshow("Visual Sparse Feedback", image)
        # cv2.waitKey(1)

    def visual_rich(self, centroids, confidences, labels, bboxes):
        
        if self.latest_image is None:
            rospy.logwarn("No image available for visual rich feedback.")
            return

        image = self.latest_image.copy()
        
        # Use normalized colors for better contrast
        for idx, (centroid, confidence, label, bbox) in enumerate(zip(centroids, confidences, labels, bboxes)):
            color = self.confidence_to_color_normalized(confidence, confidences)
            
            # Draw bounding box if coordinates are valid
            if bbox['x2'] > bbox['x1'] and bbox['y2'] > bbox['y1']:
                # print(f"Drawing bbox for {label} with confidence {confidence:.2f}")
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Draw confidence and label
                text = f"{label}: {confidence:.2f}"
                cv2.putText(image, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
            else:
                rospy.logwarn(f"Invalid bbox for {label}: {bbox}")

        # Publish the rich feedback image with all bounding boxes
        try:
            self.overlay_pub.publish(self.bridge.cv2_to_imgmsg(image, encoding='bgr8'))
            # rospy.loginfo(f"Published overlay image with {len(centroids)} objects.")
        except Exception as e:
            rospy.logerr("Failed to publish /all_objects_confidences_image: %s", e)

        # cv2.imshow("Visual Rich Feedback", image)
        # cv2.waitKey(1)

    def verbal_sparse(self, centroids, confidences, labels, gripper_states):
        # Check if speech is already playing to prevent overlap
        if self.is_speaking:
            return
        
        # Use the label from scene_bb_from_centroids (already has correct bbox matching and stabilization)
        if self.scene_highest_conf_label is None:
            rospy.logwarn_throttle(2.0, "No highest confidence label available from scene_bb_from_centroids")
            return
        
        predicted_object = self.scene_highest_conf_label
        highest_conf = self.scene_highest_conf_value
        
        # Debug output
        print(f"\n=== Verbal Sparse Using Scene BB Label ===")
        print(f"Highest confidence: {predicted_object} ({highest_conf:.2f})")
        print(f"Gripper open: {self.scene_gripper_open}")
        if self.scene_highest_centroid:
            print(f"Centroid position: x={self.scene_highest_centroid['x']:.3f}, "
                  f"y={self.scene_highest_centroid['y']:.3f}, "
                  f"z={self.scene_highest_centroid['z']:.3f}")
        print("=" * 40 + "\n")
        SUFFICIENT_CONFIDENCE = 0.2
        speech = None
        
        # Just a ding sound if confidence is high enough
        if self.sparse_verbal_granularity == VERBAL_GRANULARITY.DING:
            if highest_conf > SUFFICIENT_CONFIDENCE:
                speech = "Ding!"

        # Simply say the object label with highest confidence
        elif self.sparse_verbal_granularity == VERBAL_GRANULARITY.LABEL_ONLY:
            if highest_conf > SUFFICIENT_CONFIDENCE:
                speech = f"{predicted_object}."
                print(f"Generated sparse verbal feedback: {predicted_object} (conf: {highest_conf:.2f})")

        # Simply state the number of detected objects in the scene
        elif self.sparse_verbal_granularity == VERBAL_GRANULARITY.NUM_OBJECTS:
            num_objects = len(centroids) if centroids else 0
            speech = f"I see {num_objects} objects in the scene."

        # Convey the exact confidence level of the highest-confidence object
        elif self.sparse_verbal_granularity == VERBAL_GRANULARITY.CONFIDENCE_ONLY:
            if highest_conf > SUFFICIENT_CONFIDENCE:
                speech = f"I'm quite confident about what you are trying to do, at {highest_conf*100:.1f} percent."
            else:
                speech = f"I'm currently only {highest_conf*100:.1f} percent confident about what you are trying to do."
        
        # Assert that you know what the human wants if confidence is high enough
        elif self.sparse_verbal_granularity == VERBAL_GRANULARITY.CONFIDENCE_GATED_ASSERTION:
            if highest_conf > SUFFICIENT_CONFIDENCE:
                speech = f"Oh, I think I know what you want to do, let me help you with that!"

        # Specifically mention the item that you think the human wants if confidence is high enough
        elif self.sparse_verbal_granularity == VERBAL_GRANULARITY.CONFIDENCE_GATED_INTENT_DECLARATION:
            if highest_conf > SUFFICIENT_CONFIDENCE:
                verb = "pick up the" if self.scene_gripper_open else "place the object on"
                speech = f"Oh, I think you are trying to {verb} {predicted_object}, let me help you with that!"

        # Be as verbose as possible about all detected objects
        elif self.sparse_verbal_granularity == VERBAL_GRANULARITY.FULL_VERBOSE:
            speech = f"The highest confidence object is {predicted_object} at {int(highest_conf*100)} percent confidence."
            if centroids and len(centroids) > 1:
                speech += f" I also see {len(centroids) - 1} other objects in the scene."

        # Catch-all
        else:
            speech = "Oops, something went wrong with verbal sparse feedback on my end. Please debug."

        if speech is None:
            return

        # If the thing you are about to say was the same as last time, skip it or only play every so often
        COOLDOWN_TIME = 4  # Interval between verbal feedbacks in seconds
        REPEAT_EVER = False
        if speech == self.last_audio_message:
            if not REPEAT_EVER:
                return
            elif time.time() - self.latest_feedback_time < COOLDOWN_TIME:
                return

        print("Generated sparse verbal feedback:", speech)

        # Log the audio feedback
        ts = rospy.get_time()
        self.audio_logger.log_audio_event(ts, speech)

        # Set flag to prevent overlapping speech
        self.is_speaking = True
        self.last_audio_message = speech
        if gripper_states and gripper_states[0] == False and (self.task == "sorting" or self.task == "familiarity"):
            placement_locations = ['place_1', 'place_2', 'place_3']
            placement_locations_dict = {'place_1': 'Right Bin', 'place_3': 'Middle Bin', 'place_2': 'Left Bin'}
            if self.scene_highest_conf_label in placement_locations:
                speech = f"{placement_locations_dict[self.scene_highest_conf_label]}"
            else:
                # self.is_speaking = False
                return
        tts = TextToSpeech()
        
        # Use a threaded approach to reset the flag when speech completes
        def speak_and_reset():
            try:
                tts.speak_sync(speech)
            finally:
                self.is_speaking = False
        
        # Start speech in a separate thread
        threading.Thread(target=speak_and_reset, daemon=True).start()
        self.latest_feedback_time = time.time()

    def verbal_rich(self, centroids, confidences, labels, gripper_states):
        COOLDOWN_TIME = 10 # Interval between verbal feedbacks in seconds
        if time.time() - self.latest_feedback_time < COOLDOWN_TIME:
            return
        
        # Check if speech is already playing to prevent overlap
        if self.is_speaking:
            return

        # if not hasattr(self, 'all_data') or self.all_data is None:
        #     rospy.logwarn_throttle(5.0, "No data available for verbal rich feedback.")
        #     return
        
        # if not hasattr(self, 'all_labels') or not hasattr(self, 'all_confidences'):
        #     rospy.logwarn_throttle(5.0, "No labels/confidences available for verbal rich feedback.")
        #     return
        
        # Extract labels and confidences from scene_bb_from_centroids
        scene_labels = self.all_labels
        scene_confidences = self.all_confidences
        # scene_gripper_open = self.all_data[0].get('gripper_open', True) if self.all_data else True
        scene_gripper_open = gripper_states

        print(f"\n=== Verbal Rich Feedback ===")
        print(f"All labels: {scene_labels}")
        print(f"All confidences: {scene_confidences}")
        print(f"Gripper open: {scene_gripper_open}")
        print("=" * 40 + "\n")

        if scene_gripper_open[0] and 'place_1' in scene_labels:
            return

        
        # Hardcoded rich verbal feedback instead of VLM
        if not scene_labels:
            return
        speech = self.generate_rich_hardcoded_feedback_from_scene_data(scene_labels, scene_confidences, scene_gripper_open)
            
        
        print("Generated rich verbal feedback:", speech)

        # Set flag to prevent overlapping speech
        self.is_speaking = True
        tts = TextToSpeech()
        
        # Use a threaded approach to reset the flag when speech completes
        def speak_and_reset():
            try:
                tts.speak_sync(speech)  # Use sync to know when it's done
            finally:
                self.is_speaking = False
        
        # Start speech in a separate thread
        threading.Thread(target=speak_and_reset).start()
        self.latest_feedback_time = time.time()

    def generate_rich_hardcoded_feedback_from_scene_data(self, labels, confidences, gripper_open):
        """Generate rich verbal feedback using data from scene_bb_from_centroids"""
        
        if not labels:
            return "I don't see any objects in the scene right now."
        
        # Find highest confidence object
        highest_conf = max(confidences)
        highest_idx = confidences.index(highest_conf)
        highest_label = labels[highest_idx]

        
        # Extract gripper state - it's a list, so get the first element or the one at highest_idx
        gripper_state = gripper_open[0] if isinstance(gripper_open, list) and len(gripper_open) > 0 else gripper_open
        
        # Generate rich feedback based on scene analysis
        if highest_conf:
            action = "pick up" if gripper_state else "place something on"
            conf_percent = int(highest_conf * 100)
            
            if len(labels) > 1:
                # Get all objects except the highest confidence one (by index)
                other_objects = [labels[i] for i in range(len(labels)) if i != highest_idx]
                if other_objects:
                    # Count duplicates for better description
                    from collections import Counter
                    object_counts = Counter(other_objects)
                    
                    # Create descriptive list
                    object_descriptions = []
                    for obj, count in object_counts.items():
                        if count > 1:
                            object_descriptions.append(f"{count} {obj}s")
                        else:
                            object_descriptions.append(f"a {obj}")
                    
                    
                    # Single natural sentence combining all information
                    other_items = ', '.join(object_descriptions[:2])
                    speech = f"I'm {conf_percent}% confident you want to {action} the {highest_label}, and I can also see {other_items} nearby."
                else:
                    speech = f"I'm {conf_percent}% confident you want to {action} the {highest_label}."
            else:
                speech = f"I'm {conf_percent}% confident you want to {action} the {highest_label}."
        else:
            speech = "I can see some objects but I'm not quite sure what you're trying to do yet."
        

        return speech
    
    def generate_rich_hardcoded_feedback(self, centroids, confidences, labels, gripper_states):
        """Generate rich verbal feedback using hardcoded sentence templates"""
        
        if not centroids:
            return "I don't see any objects in the scene right now."
        
        # Find highest confidence object
        highest_conf = max(confidences)
        highest_idx = confidences.index(highest_conf)
        highest_label = labels[highest_idx]
        highest_gripper_state = gripper_states[highest_idx]
        
        # Generate rich feedback based on scene analysis
        if highest_conf:
            action = "pick up" if highest_gripper_state else "place something on"
            conf_percent = int(highest_conf * 100)
            
            if len(centroids) > 1:
                # Get all objects except the highest confidence one
                other_objects = [labels[i] for i in range(len(labels)) if i != highest_idx]
                if other_objects:
                    # Count duplicates for better description
                    from collections import Counter
                    object_counts = Counter(other_objects)
                    
                    # Create descriptive list
                    object_descriptions = []
                    for obj, count in object_counts.items():
                        if count > 1:
                            object_descriptions.append(f"{count} {obj}s")
                        else:
                            object_descriptions.append(f"a {obj}")
                    
                    # Single natural sentence combining all information
                    other_items = ', '.join(object_descriptions[:2])
                    speech = f"I'm {conf_percent}% confident you want to {action} the {highest_label}, and I can also see {other_items} nearby."
                else:
                    speech = f"I'm {conf_percent}% confident you want to {action} the {highest_label}."
            else:
                speech = f"I'm {conf_percent}% confident you want to {action} the {highest_label}."
        else:
            speech = "I can see some objects but I'm not quite sure what you're trying to do yet."

        return speech
        
    # def generate_rich_hardcoded_feedback(self, centroids, confidences, labels, gripper_states):
    #     """Generate rich verbal feedback using hardcoded sentence templates"""
    #
    #     if not centroids:
    #         return "I don't see any objects in the scene right now."
        
    #     # Find highest confidence object
    #     highest_conf = max(confidences)
    #     highest_idx = confidences.index(highest_conf)
    #     highest_label = labels[highest_idx]
    #     highest_gripper_state = gripper_states[highest_idx]
        
    #     # Count objects by confidence levels
    #     high_conf_objects = [label for conf, label in zip(confidences, labels) if conf > 0.7]
    #     medium_conf_objects = [label for conf, label in zip(confidences, labels) if 0.3 < conf <= 0.7]
    #     low_conf_objects = [label for conf, label in zip(confidences, labels) if conf <= 0.3]
        
    #     # Generate rich feedback based on scene analysis
    #     if highest_conf > 0.5:
    #         action = "pick up" if highest_gripper_state else "place something on"
            
    #         # Convert confidence to percentage for more natural speech
    #         conf_percent = int(highest_conf * 100)
            
    #         # Option 1: Include confidence in the main sentence
    #         # speech = f"I can clearly see you're working with the {highest_label} with {conf_percent}% confidence. "
    #         # speech += f"It looks like you want to {action} it. "
            
    #         # Option 2: Alternative phrasing based on confidence level
    #         # if highest_conf > 0.8:
    #         #     speech = f"I'm very confident at {conf_percent}% that you're working with the {highest_label}. "
    #         # elif highest_conf > 0.6:
    #         #     speech = f"I'm reasonably confident at {conf_percent}% that you're working with the {highest_label}. "
    #         # else:
    #         #     speech = f"I think with {conf_percent}% confidence that you're working with the {highest_label}. "
    #         # speech += f"It looks like you want to {action} it. "
            
    #         if len(centroids) > 1:
    #             # Get all objects except the specific highest confidence one (by index, not label)
    #             other_objects = [labels[i] for i in range(len(labels)) if i != highest_idx]
    #             if other_objects:
    #                 # Count duplicates for better description
    #                 from collections import Counter
    #                 object_counts = Counter(other_objects)
                    
    #                 # Create descriptive list
    #                 object_descriptions = []
    #                 for obj, count in object_counts.items():
    #                     if count > 1:
    #                         object_descriptions.append(f"{count} {obj}s")
    #                     else:
    #                         object_descriptions.append(f"a {obj}")
                    
    #                 speech += f"I also notice {', '.join(object_descriptions[:2])} in the workspace. "

    #     return speech


    def main(self):
        rospy.loginfo("Feedback Visualizer Node Started")
        rospy.loginfo(f"Feedback Type: {self.feedback_type.name}")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

        rospy.logwarn("Shutting down feedback visualizer due to ROS interrupt. Saving important logs")
        vis.shutdown()  # important to save logs

    def shutdown(self):
        user_dir = UserStudyExperiment().get_user_dir(self.task,self.treatment)
        if self.feedback_type == FEEDBACK.VERBAL_RICH or self.feedback_type == FEEDBACK.VERBAL_SPARSE:
            rospy.loginfo("Logging Audio Feedback")
            self.audio_logger.save_audio_events(user_dir)

if __name__ == '__main__':
    vis = FeedbackVisualizer()
    try:
        vis.main()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        pass
