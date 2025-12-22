# Mini-Project: Implementing a VLA System for a Humanoid Robot

## Overview

In this mini-project, you'll implement a complete Vision-Language-Action (VLA) system for a humanoid robot that can understand natural language commands, perceive objects in its environment, and execute appropriate actions. You'll create a system that demonstrates the integration of vision, language, and action components.

## Prerequisites

- Completed previous VLA integration sections
- ROS 2 Humble installed
- Gazebo simulation environment
- Basic understanding of computer vision and natural language processing

## Step 1: Create the Robot Model with VLA Sensors

Create `vla_humanoid.urdf.xacro`:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="vla_humanoid">

  <!-- Base torso -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.6"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.6"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="50"/>
      <inertia ixx="2.5" ixy="0" ixz="0" iyy="3.5" iyz="0" izz="1.5"/>
    </inertial>
  </link>

  <!-- Head with cameras -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.004" ixy="0" ixz="0" iyy="0.004" iyz="0" izz="0.004"/>
    </inertial>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1"/>
  </joint>

  <!-- RGB-D Camera -->
  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="camera_joint" type="fixed">
    <parent link="head"/>
    <child link="camera_link"/>
    <origin xyz="0.05 0 0" rpy="0 0 0"/>
  </joint>

  <!-- Stereo cameras for depth perception -->
  <link name="left_camera_link">
    <visual>
      <geometry>
        <box size="0.02 0.02 0.02"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.02 0.02 0.02"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.00005" ixy="0" ixz="0" iyy="0.00005" iyz="0" izz="0.00005"/>
    </inertial>
  </link>

  <link name="right_camera_link">
    <visual>
      <geometry>
        <box size="0.02 0.02 0.02"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.02 0.02 0.02"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.00005" ixy="0" ixz="0" iyy="0.00005" iyz="0" izz="0.00005"/>
    </inertial>
  </link>

  <joint name="left_camera_joint" type="fixed">
    <parent link="head"/>
    <child link="left_camera_link"/>
    <origin xyz="0.06 0.03 0" rpy="0 0 0"/>
  </joint>

  <joint name="right_camera_joint" type="fixed">
    <parent link="head"/>
    <child link="right_camera_link"/>
    <origin xyz="0.06 -0.03 0" rpy="0 0 0"/>
  </joint>

  <!-- IMU for balance -->
  <link name="imu_link">
    <visual>
      <geometry>
        <box size="0.02 0.02 0.02"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.02 0.02 0.02"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.000001" ixy="0" ixz="0" iyy="0.000001" iyz="0" izz="0.000001"/>
    </inertial>
  </link>

  <joint name="imu_joint" type="fixed">
    <parent link="base_link"/>
    <child link="imu_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

  <!-- Left arm -->
  <link name="left_shoulder">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_shoulder"/>
    <origin xyz="0.15 0.1 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="2"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <geometry>
        <box size="0.08 0.08 0.3"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.08 0.08 0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_shoulder"/>
    <child link="left_upper_arm"/>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.35" upper="0" effort="100" velocity="2"/>
  </joint>

  <link name="left_forearm">
    <visual>
      <geometry>
        <box size="0.06 0.06 0.25"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.06 0.06 0.25"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.003"/>
    </inertial>
  </link>

  <joint name="left_wrist_joint" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_forearm"/>
    <origin xyz="0 0 -0.25" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2"/>
  </joint>

  <link name="left_hand">
    <visual>
      <geometry>
        <box size="0.1 0.08 0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.08 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_hand_joint" type="fixed">
    <parent link="left_forearm"/>
    <child link="left_hand"/>
    <origin xyz="0 0 -0.05" rpy="0 0 0"/>
  </joint>

  <!-- Right arm -->
  <link name="right_shoulder">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="right_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_shoulder"/>
    <origin xyz="0.15 -0.1 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="2"/>
  </joint>

  <link name="right_upper_arm">
    <visual>
      <geometry>
        <box size="0.08 0.08 0.3"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.08 0.08 0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="right_elbow_joint" type="revolute">
    <parent link="right_shoulder"/>
    <child link="right_upper_arm"/>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.35" upper="0" effort="100" velocity="2"/>
  </joint>

  <link name="right_forearm">
    <visual>
      <geometry>
        <box size="0.06 0.06 0.25"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.06 0.06 0.25"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.003"/>
    </inertial>
  </link>

  <joint name="right_wrist_joint" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_forearm"/>
    <origin xyz="0 0 -0.25" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2"/>
  </joint>

  <link name="right_hand">
    <visual>
      <geometry>
        <box size="0.1 0.08 0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.08 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_hand_joint" type="fixed">
    <parent link="right_forearm"/>
    <child link="right_hand"/>
    <origin xyz="0 0 -0.05" rpy="0 0 0"/>
  </joint>

  <!-- Legs for stability -->
  <link name="left_hip">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>
  </link>

  <joint name="left_hip_joint" type="fixed">
    <parent link="base_link"/>
    <child link="left_hip"/>
    <origin xyz="0 0.1 -0.3" rpy="0 0 0"/>
  </joint>

  <link name="left_knee">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.3"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="3"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="left_knee_joint" type="fixed">
    <parent link="left_hip"/>
    <child link="left_knee"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
  </joint>

  <link name="left_ankle">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="left_ankle_joint" type="fixed">
    <parent link="left_knee"/>
    <child link="left_ankle"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
  </joint>

  <link name="right_hip">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>
  </link>

  <joint name="right_hip_joint" type="fixed">
    <parent link="base_link"/>
    <child link="right_hip"/>
    <origin xyz="0 -0.1 -0.3" rpy="0 0 0"/>
  </joint>

  <link name="right_knee">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.3"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="3"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="right_knee_joint" type="fixed">
    <parent link="right_hip"/>
    <child link="right_knee"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
  </joint>

  <link name="right_ankle">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="right_ankle_joint" type="fixed">
    <parent link="right_knee"/>
    <child link="right_ankle"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
  </joint>

  <!-- Gazebo plugins -->
  <gazebo reference="camera_link">
    <sensor type="camera" name="rgb_camera">
      <update_rate>30.0</update_rate>
      <camera name="head">
        <horizontal_fov>1.3962634</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.02</near>
          <far>300</far>
        </clip>
        <noise>
          <type>gaussian</type>
          <mean>0.0</mean>
          <stddev>0.007</stddev>
        </noise>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <frame_name>camera_link</frame_name>
        <topic_name>camera/image_raw</topic_name>
      </plugin>
    </sensor>
  </gazebo>

  <gazebo reference="left_camera_link">
    <sensor type="camera" name="left_camera">
      <update_rate>30.0</update_rate>
      <camera name="left">
        <horizontal_fov>1.3962634</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.02</near>
          <far>300</far>
        </clip>
      </camera>
      <plugin name="left_camera_controller" filename="libgazebo_ros_camera.so">
        <frame_name>left_camera_link</frame_name>
        <topic_name>stereo/left/image_raw</topic_name>
      </plugin>
    </sensor>
  </gazebo>

  <gazebo reference="right_camera_link">
    <sensor type="camera" name="right_camera">
      <update_rate>30.0</update_rate>
      <camera name="right">
        <horizontal_fov>1.3962634</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.02</near>
          <far>300</far>
        </clip>
      </camera>
      <plugin name="right_camera_controller" filename="libgazebo_ros_camera.so">
        <frame_name>right_camera_link</frame_name>
        <topic_name>stereo/right/image_raw</topic_name>
      </plugin>
    </sensor>
  </gazebo>

  <gazebo reference="imu_link">
    <sensor type="imu" name="imu_sensor">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <visualize>false</visualize>
      <plugin filename="libgazebo_ros_imu_sensor.so" name="imu_plugin">
        <topicName>imu</topicName>
        <bodyName>base_link</bodyName>
        <updateRateHZ>100.0</updateRateHZ>
        <gaussianNoise>0.01</gaussianNoise>
        <xyzOffset>0 0 0</xyzOffset>
        <rpyOffset>0 0 0</rpyOffset>
        <frameName>imu_link</frameName>
      </plugin>
    </sensor>
  </gazebo>

  <!-- Joint state publisher -->
  <gazebo>
    <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
      <update_rate>30</update_rate>
      <joint_name>left_shoulder_joint</joint_name>
      <joint_name>left_elbow_joint</joint_name>
      <joint_name>left_wrist_joint</joint_name>
      <joint_name>right_shoulder_joint</joint_name>
      <joint_name>right_elbow_joint</joint_name>
      <joint_name>right_wrist_joint</joint_name>
      <joint_name>neck_joint</joint_name>
    </plugin>
  </gazebo>

</robot>
```

## Step 2: Create the VLA Integration Node

Create `vla_integration_node.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point
from cv_bridge import CvBridge
import cv2
import numpy as np
import spacy
import torch
from torchvision import transforms
from transformers import pipeline
import json

class VLAIntegrationNode(Node):
    def __init__(self):
        super().__init__('vla_integration_node')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Initialize NLP model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.get_logger().warn("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Initialize vision model (using a simple detector for this example)
        self.object_detector = cv2.dnn.readNetFromDarknet(
            "config/yolo.cfg",
            "config/yolo.weights"
        )  # In practice, you'd use a proper configuration

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )

        self.command_sub = self.create_subscription(
            String,
            'voice_command',
            self.command_callback,
            10
        )

        self.action_pub = self.create_publisher(
            String,
            'robot_action',
            10
        )

        self.feedback_pub = self.create_publisher(
            String,
            'vla_feedback',
            10
        )

        # Internal state
        self.latest_image = None
        self.detected_objects = []
        self.current_command = None

        self.get_logger().info('VLA Integration Node initialized')

    def image_callback(self, msg):
        """Process incoming camera image"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_image = cv_image

            # Process image to detect objects
            self.detected_objects = self.detect_objects(cv_image)

            # Publish processed results
            result_msg = String()
            result_msg.data = f"Detected {len(self.detected_objects)} objects"
            self.feedback_pub.publish(result_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def command_callback(self, msg):
        """Process incoming voice command"""
        self.current_command = msg.data
        self.get_logger().info(f'Received command: {msg.data}')

        # Process command with current vision data
        if self.latest_image is not None:
            self.process_vla_command(msg.data, self.latest_image)

    def detect_objects(self, image):
        """Detect objects in image using simple method"""
        # In a real implementation, you would use a proper object detection model
        # For this example, we'll use a simple color-based detection as placeholder
        detected = []

        # Convert to HSV for color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color ranges for simple detection
        colors = {
            'red': ([0, 50, 50], [10, 255, 255]),
            'blue': ([100, 50, 50], [130, 255, 255]),
            'green': ([40, 50, 50], [80, 255, 255])
        }

        for color_name, (lower, upper) in colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 1000:  # Filter small contours
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w // 2
                    center_y = y + h // 2

                    detected.append({
                        'name': color_name,
                        'bbox': [x, y, x + w, y + h],
                        'center': [center_x, center_y],
                        'confidence': 0.8
                    })

        return detected

    def parse_command(self, command):
        """Parse natural language command"""
        if self.nlp is None:
            # Simple fallback parsing
            command_lower = command.lower()
            parsed = {
                'action': 'unknown',
                'object': 'unknown',
                'location': 'unknown'
            }

            if 'pick' in command_lower or 'grasp' in command_lower or 'take' in command_lower:
                parsed['action'] = 'grasp'
            elif 'put' in command_lower or 'place' in command_lower:
                parsed['action'] = 'place'
            elif 'go' in command_lower or 'move' in command_lower:
                parsed['action'] = 'navigate'

            # Extract object (simplified)
            for word in command_lower.split():
                if word in ['cup', 'book', 'ball', 'red', 'blue', 'green']:
                    parsed['object'] = word
                    break

            return parsed

        # Use spaCy for more sophisticated parsing
        doc = self.nlp(command)

        parsed = {
            'action': None,
            'object': None,
            'location': None
        }

        # Extract action (verb)
        for token in doc:
            if token.pos_ == "VERB":
                parsed['action'] = token.lemma_
                break

        # Extract object
        for ent in doc.ents:
            if ent.label_ in ["OBJECT", "PRODUCT", "EVENT"]:
                parsed['object'] = ent.text
                break

        return parsed

    def match_object(self, command_object, detected_objects):
        """Match command object to detected objects"""
        if not detected_objects:
            return None

        # Simple matching based on color or name
        for obj in detected_objects:
            if command_object.lower() in obj['name'].lower():
                return obj

        # If no exact match, return the first detected object as fallback
        return detected_objects[0] if detected_objects else None

    def plan_action(self, parsed_command, matched_object):
        """Plan action based on command and detected objects"""
        action_plan = {
            'action_type': parsed_command['action'],
            'target_object': matched_object,
            'success': True,
            'reasoning': []
        }

        if parsed_command['action'] == 'grasp':
            if matched_object:
                action_plan['reasoning'].append(f"Found {matched_object['name']} at {matched_object['center']}")
                action_plan['target_pose'] = self.calculate_grasp_pose(matched_object)
            else:
                action_plan['success'] = False
                action_plan['reasoning'].append("Target object not found in view")

        elif parsed_command['action'] == 'navigate':
            # For navigation, we'd need a map and path planning
            action_plan['reasoning'].append("Navigation action planned")

        elif parsed_command['action'] == 'place':
            action_plan['reasoning'].append("Placement action planned")

        return action_plan

    def calculate_grasp_pose(self, object_info):
        """Calculate grasp pose for an object"""
        # Convert 2D image coordinates to 3D world coordinates (simplified)
        # In practice, you'd use depth information or stereo vision
        x_2d, y_2d = object_info['center']

        # Simplified 3D pose calculation
        pose = Pose()
        pose.position.x = x_2d / 100.0  # Scale to reasonable units
        pose.position.y = y_2d / 100.0
        pose.position.z = 0.5  # Fixed height for this example

        # Default orientation for grasping
        pose.orientation.w = 1.0  # No rotation

        return pose

    def process_vla_command(self, command, image):
        """Process complete VLA command"""
        try:
            # 1. Parse the language command
            parsed_command = self.parse_command(command)
            self.get_logger().info(f'Parsed command: {parsed_command}')

            # 2. Match to detected objects
            matched_object = self.match_object(parsed_command['object'], self.detected_objects)
            self.get_logger().info(f'Matched object: {matched_object}')

            # 3. Plan the action
            action_plan = self.plan_action(parsed_command, matched_object)
            self.get_logger().info(f'Action plan: {action_plan}')

            # 4. Execute or publish the action
            if action_plan['success']:
                action_msg = String()
                action_msg.data = json.dumps({
                    'action_type': action_plan['action_type'],
                    'target_object': action_plan['target_object'],
                    'target_pose': {
                        'x': action_plan.get('target_pose', {}).position.x if action_plan.get('target_pose') else 0,
                        'y': action_plan.get('target_pose', {}).position.y if action_plan.get('target_pose') else 0,
                        'z': action_plan.get('target_pose', {}).position.z if action_plan.get('target_pose') else 0
                    } if action_plan.get('target_pose') else None
                })
                self.action_pub.publish(action_msg)

                feedback_msg = String()
                feedback_msg.data = f"Executing: {command} - Found {len(self.detected_objects)} objects, matched to {matched_object['name'] if matched_object else 'none'}"
                self.feedback_pub.publish(feedback_msg)
            else:
                feedback_msg = String()
                feedback_msg.data = f"Could not execute: {command} - {action_plan['reasoning']}"
                self.feedback_pub.publish(feedback_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing VLA command: {e}')
            feedback_msg = String()
            feedback_msg.data = f"Error processing command: {e}"
            self.feedback_pub.publish(feedback_msg)

def main(args=None):
    rclpy.init(args=args)
    vla_node = VLAIntegrationNode()

    try:
        rclpy.spin(vla_node)
    except KeyboardInterrupt:
        pass
    finally:
        vla_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 3: Create Supporting Configuration Files

Create `config/vla_params.yaml`:

```yaml
vla_integration_node:
  ros__parameters:
    # Vision parameters
    detection_threshold: 0.5
    max_detection_objects: 10
    image_processing_rate: 10.0  # Hz

    # Language parameters
    language_confidence_threshold: 0.7
    command_timeout: 30.0  # seconds

    # Action parameters
    action_execution_timeout: 60.0  # seconds
    safety_distance: 0.1  # meters
    max_retries: 3

    # Integration parameters
    sync_window: 0.1  # seconds
    uncertainty_threshold: 0.6
```

## Step 4: Create the Launch File

Create `launch/vla_demo.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    ld = LaunchDescription()

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    params_file = LaunchConfiguration('params_file', default='')

    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'))

    ld.add_action(DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('my_robot_pkg'),
            'config',
            'vla_params.yaml'
        ]),
        description='Path to parameters file'))

    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
    )
    ld.add_action(gazebo)

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'vla_humanoid',
            '-x', '0', '-y', '0', '-z', '0.5'
        ],
        output='screen'
    )
    ld.add_action(spawn_entity)

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    ld.add_action(robot_state_publisher)

    # Joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    ld.add_action(joint_state_publisher)

    # VLA integration node
    vla_integration = Node(
        package='my_robot_pkg',
        executable='vla_integration_node',
        name='vla_integration_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            PathJoinSubstitution([
                FindPackageShare('my_robot_pkg'),
                'config',
                'vla_params.yaml'
            ])
        ],
        output='screen'
    )
    ld.add_action(vla_integration)

    # Simple voice command simulator (for testing)
    voice_simulator = Node(
        package='my_robot_pkg',
        executable='voice_command_simulator',
        name='voice_command_simulator',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    ld.add_action(voice_simulator)

    # RViz for visualization
    rviz_config = PathJoinSubstitution([
        FindPackageShare('my_robot_pkg'),
        'rviz',
        'vla_demo.rviz'
    ])
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}]
    )
    ld.add_action(rviz)

    return ld
```

## Step 5: Create a Simple Voice Command Simulator

Create `voice_command_simulator.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class VoiceCommandSimulator(Node):
    def __init__(self):
        super().__init__('voice_command_simulator')

        self.command_pub = self.create_publisher(
            String,
            'voice_command',
            10
        )

        # Timer to send commands periodically
        self.timer = self.create_timer(10.0, self.send_command)
        self.command_index = 0

        # Sample commands for demonstration
        self.commands = [
            "Pick up the red cup",
            "Find the blue ball",
            "Move to the kitchen",
            "Grasp the green object",
            "Put the object on the table"
        ]

        self.get_logger().info('Voice Command Simulator initialized')

    def send_command(self):
        """Send a voice command periodically"""
        if self.command_index < len(self.commands):
            command = String()
            command.data = self.commands[self.command_index]
            self.command_pub.publish(command)
            self.get_logger().info(f'Sent command: {command.data}')
            self.command_index += 1
        else:
            # Reset after all commands are sent
            self.command_index = 0

def main(args=None):
    rclpy.init(args=args)
    simulator = VoiceCommandSimulator()

    try:
        rclpy.spin(simulator)
    except KeyboardInterrupt:
        pass
    finally:
        simulator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 6: Build and Run the VLA System

1. Build your ROS 2 package:
```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_pkg
source install/setup.bash
```

2. Launch the VLA demonstration:
```bash
ros2 launch my_robot_pkg vla_demo.launch.py
```

3. In another terminal, you can also manually send commands:
```bash
# Send a voice command
ros2 topic pub /voice_command std_msgs/String "data: 'Pick up the red cup'"

# Monitor the robot's actions
ros2 topic echo /robot_action

# Monitor feedback
ros2 topic echo /vla_feedback
```

## Expected Results

- Robot model appears in Gazebo with appropriate sensors
- Vision system detects colored objects in the environment
- Language system parses natural language commands
- Action system executes appropriate responses based on VLA integration
- Robot successfully demonstrates simple VLA behaviors

## Troubleshooting

If the VLA system doesn't work properly:

1. Check that all required ROS 2 packages are installed
2. Verify that the robot model is properly configured with sensors
3. Ensure vision and language models are correctly loaded
4. Check that all topics are properly connected
5. Verify TF transforms are properly configured

## Next Steps

Continue to [Troubleshooting](./troubleshooting.md) to learn about common VLA integration issues and solutions.