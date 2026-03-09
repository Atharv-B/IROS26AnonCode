---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Hardware Setup](#hardware-setup)
- [Quick Start](#quick-start)
- [Package Structure](#package-structure)
- [Launch Files](#launch-files)
- [Key Components](#key-components)
- [User Study Protocol](#user-study-protocol)
- [Configuration](#configuration)
- [Data Collection](#data-collection)
- [Troubleshooting](#troubleshooting)

---

## Overview

This package implements a teleoperation system for the Kinova Gen3 robotic arm with integrated visual and auditory feedback mechanisms. It is designed for conducting human-robot interaction studies that investigate how different feedback modalities affect operator trust and task performance.

The system uses:
- **YOLOv11** for real-time object detection
- **RealSense cameras** for RGB-D sensing (end-effector, scene, and recording perspectives)
- **Xbox controller** for robot teleoperation
- **VOSA (Visual Only Shared Autonomy)** for assistive manipulation
- **Multiple feedback modalities** (visual rich, visual sparse, auditory rich, auditory sparse)

---

## Features

- 🤖 **Teleoperation Control**: Xbox controller-based teleoperation of Kinova Gen3 arm with Robotiq 2F-85 gripper
- 👁️ **Real-time Object Detection**: YOLOv11-based detection with confidence scoring
- 📹 **Multi-camera System**: Three RealSense cameras (wrist-mounted, scene, recording)
- 🎯 **Goal Alignment Tracking**: Monitors alignment between robot intentions and goal locations
- 🔄 **TF2 Transforms**: Complete coordinate frame management for camera-to-world transformations
- 📊 **Visual Feedback**: Bounding boxes and confidence overlays on detected objects
- 🔊 **Auditory Feedback**: Text-to-speech feedback for object detection and robot state
- 📝 **Logging**: Automated data collection for user study analysis
- 🧪 **Multiple Task Modes**: Support for sorting, shelving, familiarity tasks

---

## System Requirements

### Hardware
- **Robot**: Kinova Gen3 robotic arm (6 or 7 DOF)
- **Gripper**: Robotiq 2F-85 gripper
- **Cameras**: 3x Intel RealSense D435/D455 cameras
  - End-effector camera (serial: `247122071167`)
  - Environment recording camera (serial: `246422070578`)
  - Scene camera (serial: `243322074721`)
- **Controller**: Xbox controller (or compatible joystick)
- **Compute**: Ubuntu Linux machine with NVIDIA GPU

### Software
- Ubuntu 18.04 or 20.04
- ROS Melodic or Noetic
- Python 3.6+
- CUDA-compatible GPU drivers (for YOLOv11)

### Dependencies
- `kortex_driver` - Kinova ROS driver
- `realsense2_camera` - Intel RealSense ROS wrapper
- `cv_bridge` - OpenCV-ROS bridge
- `tf2_ros` - ROS transform library
- `image_geometry` - Camera projection utilities
- `joy` - Joystick driver
- Python packages: `numpy`, `opencv-python`, `ultralytics` (YOLOv11), `torch`

---

## Installation

1. **Create a catkin workspace** (if you don't have one):
   ```bash
   mkdir -p ~/catkin_ws/src
   cd ~/catkin_ws/src
   ```

2. **Clone this repository**:
   ```bash
   git clone https://github.com/yourusername/trust_and_transparency.git
   ```

3. **Install ROS dependencies**:
   ```bash
   cd ~/catkin_ws
   rosdep install --from-paths src --ignore-src -r -y
   ```

4. **Install Python dependencies**:
   ```bash
   pip3 install numpy opencv-python ultralytics torch torchvision
   ```

5. **Build the package**:
   ```bash
   cd ~/catkin_ws
   catkin_make
   source devel/setup.bash
   ```

6. **Download YOLO models**:
   - Place `yolo11n.pt` and `yolocustom.pt` in the package root directory

---

## Hardware Setup

### Camera Mounting
1. **End-effector camera**: Mount on robot wrist with clear view of gripper workspace
2. **Scene camera**: Position overhead or at angle to capture full workspace
3. **Recording camera**: External viewpoint for recording participant interactions

### Camera Calibration
The package includes automated camera calibration. Ensure the scene camera has a clear view of the robot base link for accurate transformations.

### Robot Connection
Connect the Kinova arm via Ethernet and configure the IP address (default: `192.168.1.10`).

### Controller Setup
Connect Xbox controller via USB. Verify device path (default: `/dev/input/js1`).

---

## Quick Start

### Basic Teleoperation
```bash
roslaunch trust_and_transparency direct_teleop.launch
```

### Run User Study Experiment
1. **Initialize participant session**:
   ```bash
   python src/user_study/user_study.py
   ```
   This generates a unique participant ID and randomizes treatment order.

2. **Launch experiment**:
   ```bash
   roslaunch trust_and_transparency trust_feedback.launch \
       task:=sorting \
       treatment:=C \
       feedback_type:=visual_rich
   ```

### Available Parameters
- `task`: `sorting`, `shelving`, `familiarity`, 
- `treatment`: Experiment condition identifier (e.g., `A`, `B`, `C`)
- `feedback_type`: `visual_rich`, `visual_sparse`, `auditory_rich`, `auditory_sparse`
- `ip_address`: Robot IP (default: `192.168.1.10`)

---

## Package Structure

```
trust_and_transparency/
├── launch/                          # ROS launch files
│   ├── direct_teleop.launch        # Basic teleoperation
│   ├── trust_feedback.launch       # Main experiment launch file
│   ├── familiarity.launch          # Familiarity task
│   └── trust_and_transparency_visual_feedback.launch
├── msg/                             # Custom ROS messages
│   ├── CentroidConfidence.msg      # Object detection with confidence
│   └── CentroidConfidenceArray.msg # Array of detections
├── src/                             # Python source code
│   ├── constants.py                # Task configurations and parameters
│   ├── ActuatorModel.py            # Robot kinematics model
│   ├── ForwardKinematics.py        # FK solver
│   ├── yolov11_for_scene.py        # Scene object detection
│   ├── test_yolov11.py             # Wrist camera detection
│   ├── vosa_for_trust.py           # VOSA autonomy (shelving)
│   ├── vosa_top_down.py            # VOSA autonomy (sorting)
│   ├── cam_to_world.py             # Camera transform publisher
│   ├── scene_bb_from_centroids.py  # Bounding box visualization
│   ├── viz_feedback.py             # Feedback visualizer
│   ├── goal_alignment_logger.py    # Goal alignment tracking
│   ├── record.py                   # Video recording
│   ├── auditory/                   # Auditory feedback modules
│   │   ├── tts.py                  # Text-to-speech
│   │   └── auditory_rich.py        # Rich auditory feedback
│   └── user_study/                 # User study utilities
│       ├── user_study.py           # Participant management
│       ├── joy_logger.py           # Joystick input logging
│       ├── robot_logger.py         # Robot state logging
│       ├── cam_logger.py           # Camera recording
│       ├── audio_logger.py         # Audio feedback logging
│       └── goal_logger.py          # Goal tracking
├── include/                         # C++ headers (if any)
├── CMakeLists.txt                  # Build configuration
├── package.xml                     # ROS package manifest
└── README.md                       # This file
```

---

## Launch Files

### `trust_feedback.launch`
Main launch file for user study experiments. Starts:
- Kinova robot driver
- All three RealSense cameras
- YOLO object detectors (wrist + scene)
- VOSA autonomy system
- Feedback visualization
- Data logging nodes

### `direct_teleop.launch`
Simplified teleoperation without feedback systems. Useful for:
- Testing robot connectivity
- Manual manipulation tasks
- Hardware debugging

### `familiarity.launch`
Specific configuration for familiarity training tasks.

---

## Key Components

### Object Detection
- **YOLOv11**: Custom trained model for task-specific objects
- **Dual detection**: Wrist camera (manipulation) + scene camera (awareness)
- **Confidence scoring**: Each detection includes confidence metric

### VOSA (Visual Object-centric Shared Autonomy)
- Assists operator by suggesting optimal grasp locations
- Computes goal alignment scores
- Adapts to different task types (top-down sorting vs. shelving)

### Feedback Modalities
- **Visual Rich**: All detected objects with bounding boxes and confidence
- **Visual Sparse**: Only highest confidence object highlighted
- **Auditory Rich**: Verbal descriptions of all detected objects
- **Auditory Sparse**: Alerts for highest confidence object only

### Transform Management
- Automatic calibration between camera frames and robot base
- Real-time TF2 transformations for accurate centroid projection
- Stabilization algorithms to reduce jitter

---

## User Study Protocol

### Participant Initialization
```bash
python src/user_study/user_study.py
```
Creates:
- Unique participant ID
- Randomized treatment order (for within-subjects design)
- Results directory structure

### Running Trials
```bash
roslaunch trust_and_transparency trust_feedback.launch \
    task:=sorting \
    treatment:=A \
    feedback_type:=visual_rich
```

### Data Collected Per Trial
- **`info.json`**: Participant ID, treatment order, metadata
- **`joy.csv`**: Time-series of joystick movements
- **`robot_state.csv`**: Robot joint angles and end-effector pose
- **`audio_responses.csv`**: Text log of auditory feedback (auditory conditions only)
- **`goals.json`**: Timestamps of goal location changes
- **`end_effector_perspective.mov`**: Video from wrist camera
- **`external_camera_perspective.mov`**: Video from recording camera
- **Goal alignment logs**: Real-time tracking of robot-goal alignment

---

## Configuration

### Task-Specific Settings
Edit [`src/constants.py`](src/constants.py):
- `HOME`: Robot home position (joint angles)
- `RETRACT`: Safe retract position
- `SPEED_CONTROL`: Teleoperation speed multiplier (default: 0.5)
- `PLACEMENT_THRESHOLDS`: Task-specific workspace boundaries
- `ORACLE_GOAL_SET`: Ground truth object and goal locations

### Camera Serial Numbers
Edit launch files to match your RealSense serial numbers:
```xml
<arg name="serial_no" value="YOUR_SERIAL_HERE"/>
```

### Detection Parameters
Adjust in node launch arguments:
- `confidence_threshold`: Minimum confidence for stable detection (default: 0.3)
- `movement_threshold`: Centroid stabilization distance (default: 0.03m)
- `pixel_threshold`: Pixel stabilization distance (default: 1px)

---

## Data Collection

All experimental data is saved to participant-specific directories:
```
~/user_study_data/
└── participant_XXXX/
    ├── trial_1/
    │   ├── info.json
    │   ├── joy.csv
    │   ├── robot_state.csv
    │   ├── goals.json
    │   ├── end_effector_perspective.mov
    │   └── external_camera_perspective.mov
    ├── trial_2/
    └── ...
```

### Data Analysis
CSV files contain timestamped data suitable for:
- Trajectory analysis
- Goal alignment metrics
- Input analysis (joystick patterns)
- Performance comparison across feedback conditions

---

## Troubleshooting

### Camera Issues
- **Camera not detected**: Check `rs-enumerate-devices` to verify serial numbers
- **Transform errors**: Ensure camera calibration node is running
- **Low frame rate**: Reduce `depth_fps` and `color_fps` in launch file

### Robot Connection
- **Cannot connect**: Verify IP address and Ethernet connection
- **Gripper not responding**: Check gripper parameter in `kortex_driver.launch`

### YOLO Performance
- **Slow detection**: Ensure CUDA is properly installed
- **Poor accuracy**: Consider retraining model on your specific objects
- **High CPU usage**: Reduce camera resolution or frame rate

### TF Transform Issues
```bash
# View transform tree
rosrun rqt_tf_tree rqt_tf_tree

# Check specific transform
rosrun tf tf_echo base_link static_camera_color_optical_frame
```

---
