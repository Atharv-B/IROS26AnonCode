---

## Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Package Structure](#package-structure)
- [Launch Files](#launch-files)
- [Key Components](#key-components)
- [User Study Protocol](#user-study-protocol)

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


### Camera Calibration
The package includes automated camera calibration.

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

