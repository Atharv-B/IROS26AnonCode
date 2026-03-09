#!/usr/bin/env python3

'''
This module provides verbal feedback based on the detected objects in the scene.
It uses OpenAI's API to generate natural language descriptions of the objects
and their confidences. Different types of verbal feedback can be specified. 

short: Naming the object(s) being picked up
Concise: A more detailed description that includes confidence level
Polite: A more human-friendly and polite description of what the robot is "thinking"
verbose: A detailed description of all objects and their confidence levels'''

import base64
from openai import OpenAI
import os
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
#create branch called: all_feedback_types
class Style():
  RED = "\033[31m"
  RESET = "\033[0m"

class VerbalFeedback:
    def __init__(self, option="verbose"):
        self.client = OpenAI()
        self.client.api_key = os.getenv("OPENAI_API_KEY")
        if not self.client.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        print("VerbalFeedback initialized with OpenAI client.")
        
        self.bridge = CvBridge()
        self.current_image = None  # Initialize current_image to None

        self.image_subscriber = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)

        self.example_type = option # Options: "short", "concise", "polite" "verbose"

        if self.example_type == "short":
            self.examples =    '''
                - "Grabbing bottle"
                - "Grabbing smartphone"
                - "Grabbing spoon"
            '''
        elif self.example_type == "concise":
            self.examples = '''
                - "Grabbing the red bottle on the top shelf with 75 percent confidence"
                - "Picking up the smartphone that is on top of the table with 60 percent confidence"
                - "Grabbing the small spoon next to the white bowl with 94 percent confidence"      
            '''
        elif self.example_type == "polite":
            self.examples = '''
                - "You probably want the red bottle off the top shelf. Let me grab that for you."
                - "I somewhat believe you want assistance in getting your smartphone that's on top of the table."
                - "I'm confident you want the spoon that's sitting next to the white bowl."
                - "I think you are trying to reach for the bottle which is on the left side of the cup. I am unsure about this action."
            '''
        elif self.example_type == "verbose":
            self.examples = '''
                - "I think you are trying to reach for the red bottle which is on the top shelf. I have a confidence of 75 percent for this action. There is also a smaller bottle to its right, but I have 20 percent confidence for this action"
                - "I am 30 percent confident that you want the napkin on top of the table. I believe you want to pick up the smartphone that is next to the napkin. My confidence for this action is 60 percent."
                - "It seems like you are trying to grab the small spoon next to the white bowl. I am 94 percent confident about this action. It is unlikely that you want the large spoon next to it, as I have only 10 percent confidence for that action."
            '''

        if self.example_type == "verbose":
            self.describe_system_prompt = f'''
                Always start with
                You are a system that is interpreting the intent of a robotic arm from an image. Your task is to describe what objects the robotic arm can pick up given the set of coordinates to various objects in the image. You are also given the labels of each object at the coordinates and their confidence levels.
                Based on the analysis of an image and given coordinates, describe the various objects you believe the arm could be picking up and the confidence of that action. Use the percent confidence given to you to describe how certain you are about each object in the scene. 
                Focus on objects with a matching label. Do not describe the background or unrelated objects. Do not include objects without a label in your response. Do not make up labels and confidence levels for objects that do not have them.
                "Please be more human in your responses."
                "The robot is on the right hand side of the camera view. The closer the object to the robot (closer to the right) the higher confidence it will have."
                '''
        else:
            self.describe_system_prompt = f'''
                You are a system that is interpreting the intent of a robotic arm from an image. Your task is to describe what object/item the robotic arm intends to pick up given the coordinates to the object in the image. You are also given the label of the object at the coordinates and its confidence level.
                Based on the analysis of an image and given coordinates within that image, describe the object you believe you are picking up. Use the percent confidence given to you to describe how certain you are that this is the correct object to pick up. 
                Focus on the object with the highest confidence that has a corresponding label. Do not describe the background or unrelated objects. Do not include objects of lower confidence in your response.
                '''

        self.describe_system_prompt += f'''
            If there is ambiguity in the intended object, describe the most likely target that the robotic arm will pick up.
            Be clear, specific, and concise. Use natural, accessible language that helps the user understand what the arm is trying to pick up. Restrain from answering with more than four sentences. Do not include the coordinates in your responses.

            Here are example outputs:
            {self.examples}'''


        print(f"{Style.RED}Response Type{Style.RESET}", self.example_type)
        print(f"{Style.RED}System Prompt{Style.RESET}", self.describe_system_prompt)
        
        
    def image_callback(self, msg):
        try:
            # Convert ROS image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Encode image as JPEG and convert to base64
            success, buffer = cv2.imencode('.jpg', cv_image)
            if not success:
                rospy.logerr("Failed to encode image as JPEG")
                return
                
            # Convert to base64 string
            self.current_image = base64.b64encode(buffer).decode('utf-8')
            
        except Exception as e:
            rospy.logerr(f"Image conversion failed: {e}")
            self.current_image = None

    def describe_image(self, centroids, confidences, labels, gripper_states):
        # Check if image is available
        if self.current_image is None:
            rospy.logwarn("No image available yet. Waiting for camera data...")
            return "No camera image available for analysis."

        highest_confidence = max(confidences) if confidences else 0
        most_confident_centroid = centroids[confidences.index(highest_confidence)]
        most_confident_label = labels[confidences.index(highest_confidence)]
        user_prompt = f""
        if self.example_type == "verbose":
            for i, (centroid, confidence, label, gripper_state) in enumerate(zip(centroids, confidences, labels, gripper_states)):
                if i > 0:
                    user_prompt += "\n"
                user_prompt += f"Object {i+1}:\n"
                user_prompt += f"Coordinates: (x: {centroid.x}, y: {centroid.y})\n"
                user_prompt += f"Confidence: {confidence*100:.1f}%\n"
                user_prompt += f"Label: {label}\n"
                user_prompt += f"Gripper State: {'Open, which means the user is try to pick up an object' if gripper_state else 'Closed, the user is trying to place down an object'}\n"
        else:
            user_prompt = f"Coordinates: (x: {most_confident_centroid.x}, y: {most_confident_centroid.y})\nConfidence: {highest_confidence*100:.1f}\nLabel: {most_confident_label}"
            user_prompt += f"Only focus on the object at these coordinates. Do not mention any other objects in the scene."

        user_prompt +=  f"\nPlease describe the object or objects being picked up at these coordinates. Thank you."


        response = self.client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": self.describe_system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": user_prompt
                    },
                    {
                        "type": "image_url", 
                        "image_url": { 
                            "url": f"data:image/jpeg;base64,{self.current_image}"
                        }
                    },
                ],
            },
        ],
        max_tokens=800,
        )
        print(f"{Style.RED}Response from OpenAI:{Style.RESET}", response.choices[0].message.content.strip())
        return response.choices[0].message.content.strip()
