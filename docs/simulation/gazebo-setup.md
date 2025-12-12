# Gazebo Setup

## Overview

Gazebo is a 3D simulation environment that provides realistic physics simulation, high-quality graphics, and convenient programmatic interfaces. It's widely used in robotics for testing algorithms, training robots, and validating designs.

## Installation

### Prerequisites

- ROS 2 Humble Hawksbill installed
- Ubuntu 22.04 LTS (recommended)
- Graphics card with OpenGL 2.1 support

### Installing Gazebo Garden

ROS 2 Humble works well with Gazebo Garden:

```bash
sudo apt update
sudo apt install gazebo
# Install ROS 2 Gazebo packages
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control ros-humble-gazebo-ros2-control-demos
```

### Environment Setup

Add Gazebo paths to your environment:

```bash
echo 'export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/.gazebo/models' >> ~/.bashrc
echo 'export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:~/.gazebo/models' >> ~/.bashrc
source ~/.bashrc
```

## Basic Usage

### Launching Gazebo

Start Gazebo with an empty world:

```bash
gazebo --verbose
```

Or launch with a specific world:

```bash
gazebo worlds/willowgarage.world
```

### Connecting to ROS 2

Gazebo can interface with ROS 2 through the `gazebo_ros_pkgs`:

```bash
# Terminal 1
gazebo --verbose

# Terminal 2 (after sourcing ROS 2)
ros2 run gazebo_ros spawn_entity.py -database simple_arm -entity simple_arm -x 0 -y 0 -z 1
```

## Integration with ROS 2

Gazebo provides several ROS 2 interfaces:

- `/clock` - Simulation time publisher
- `/tf` - Transform broadcaster
- `/model_states` - Model poses in the world
- `/joint_states` - Joint positions, velocities, and efforts

## Troubleshooting

Common issues and solutions:

- **Graphics errors**: Ensure your graphics drivers are properly installed and OpenGL 2.1+ is supported
- **Model not showing**: Check that the model path is correctly set in `GAZEBO_MODEL_PATH`
- **Performance issues**: Reduce physics update rate or simplify models

## Next Steps

Continue to [Unity Setup](./unity-setup.md) to learn about alternative simulation environments.