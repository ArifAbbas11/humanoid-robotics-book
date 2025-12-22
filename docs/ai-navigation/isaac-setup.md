# Isaac Setup

## Overview

This guide covers the installation and configuration of NVIDIA Isaac tools for humanoid robotics applications. Isaac provides powerful simulation, perception, and navigation capabilities that leverage NVIDIA's GPU acceleration.

## Prerequisites

Before installing Isaac tools, ensure you have:

- Ubuntu 22.04 LTS (recommended for ROS 2 Humble)
- NVIDIA GPU with CUDA support (Compute Capability 6.0+)
- ROS 2 Humble Hawksbill installed and working
- Docker installed (for Isaac Sim)
- At least 8GB RAM (16GB+ recommended)
- 50GB+ free disk space

## System Requirements Check

First, verify your system meets the requirements:

```bash
# Check Ubuntu version
lsb_release -a

# Check NVIDIA GPU and CUDA
nvidia-smi
nvcc --version

# Check ROS 2 installation
ros2 --version
```

## Installing Isaac ROS Dependencies

### 1. Install CUDA Toolkit

If CUDA is not already installed:

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
sudo sh cuda_12.4.1_550.54.15_linux.run
```

Add CUDA to your environment:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 2. Install Isaac ROS Core Packages

```bash
# Add NVIDIA's Isaac ROS repository
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository "deb https://packages.nvidia.com/isaac/ros/humble/ubuntu/$(lsb_release -cs)/ arm64"
sudo apt update

# Install Isaac ROS packages
sudo apt install ros-humble-isaac-ros-pointcloud-utils
sudo apt install ros-humble-isaac-ros-visual- slam
sudo apt install ros-humble-isaac-ros-manipulation
sudo apt install ros-humble-isaac-ros-a100
```

### 3. Install Isaac ROS Extensions

```bash
# Install additional Isaac ROS packages
sudo apt install ros-humble-isaac-ros-apriltag
sudo apt install ros-humble-isaac-ros-ess
sudo apt install ros-humble-isaac-ros-omniverse
sudo apt install ros-humble-isaac-ros-realsense
```

## Installing Isaac Sim

### 1. Install Docker and NVIDIA Container Toolkit

```bash
# Install Docker
sudo apt update
sudo apt install docker.io
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. Pull Isaac Sim Docker Image

```bash
# Pull the latest Isaac Sim image
docker pull nvcr.io/nvidia/isaac-sim:4.2.0

# Test the installation
docker run --gpus all -it --rm --env "ACCEPT_EULA=Y" --env "NVIDIA_VISIBLE_DEVICES=all" nvcr.io/nvidia/isaac-sim:4.2.0
```

### 3. Create Isaac Sim Launch Script

Create a convenient launch script:

```bash
cat << 'EOF' > ~/launch_isaac_sim.sh
#!/bin/bash

# Isaac Sim launch script
xhost +local:docker

docker run --gpus all \
    --rm \
    -e "ACCEPT_EULA=Y" \
    -e "NVIDIA_VISIBLE_DEVICES=all" \
    -e "NVIDIA_DRIVER_CAPABILITIES=all" \
    -e "QT_X11_NO_MITSHM=1" \
    -e "PYTHON_ROOT=/isaac-sim/python.sh" \
    -e "CABANA_ROOT=/isaac-sim/cabana.sh" \
    --network=host \
    --privileged \
    --pid=host \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /tmp/.docker.xauth:/tmp/.docker.xauth:rw \
    -v ~/isaac-sim-content:/isaac-sim-content \
    -v ~/Documents/humanoid-robotics-book/assets:/assets \
    nvcr.io/nvidia/isaac-sim:4.2.0
EOF

chmod +x ~/launch_isaac_sim.sh
```

## Installing Isaac Lab

Isaac Lab provides tools for robot learning and simulation:

```bash
# Create workspace for Isaac Lab
mkdir -p ~/isaac_lab_ws/src
cd ~/isaac_lab_ws

# Clone Isaac Lab repository
git clone https://github.com/NVIDIA-Omniverse/IsaacLab.git src/IsaacLab

# Install Isaac Lab dependencies
cd src/IsaacLab
./isaaclab.sh --install

# Source the setup
source ~/isaac_lab_ws/src/IsaacLab/source.sh
```

## Configuring Isaac Tools

### 1. Environment Variables

Add Isaac-specific environment variables to your `.bashrc`:

```bash
echo '# Isaac ROS Environment' >> ~/.bashrc
echo 'export ISAAC_ROS_WS=~/isaac_ros_ws' >> ~/.bashrc
echo 'export ISAAC_SIM_WS=~/isaac_sim_ws' >> ~/.bashrc
echo 'export ISAAC_OMNIVERSE_APP_PATH=/isaac-sim' >> ~/.bashrc
```

### 2. Verify Installation

Test that Isaac tools are properly installed:

```bash
# Test Isaac ROS packages
ros2 pkg list | grep isaac

# Test Isaac Sim (run in a separate terminal)
# ~/launch_isaac_sim.sh

# Check Isaac Lab installation
cd ~/isaac_lab_ws/src/IsaacLab
python -c "import omni; print('Isaac Lab installed successfully')"
```

## Troubleshooting Common Issues

### CUDA Compatibility Issues

**Issue**: CUDA version mismatch between system and Isaac tools.

**Solutions**:
1. Verify CUDA version compatibility:
   ```bash
   nvcc --version
   nvidia-smi
   ```
2. Install compatible CUDA version if needed
3. Check Isaac ROS documentation for supported CUDA versions

### Docker Permission Issues

**Issue**: Permission denied when running Docker containers.

**Solutions**:
1. Add user to docker group:
   ```bash
   sudo usermod -aG docker $USER
   ```
2. Log out and log back in
3. Test Docker:
   ```bash
   docker run hello-world
   ```

### Isaac Sim Display Issues

**Issue**: Isaac Sim doesn't display properly or crashes.

**Solutions**:
1. Check X11 forwarding:
   ```bash
   echo $DISPLAY
   ```
2. Ensure proper GPU access:
   ```bash
   nvidia-smi
   ```
3. Try running with software rendering:
   ```bash
   export LIBGL_ALWAYS_SOFTWARE=1
   ```

## Performance Optimization

### GPU Configuration

Optimize GPU settings for Isaac tools:

```bash
# Check current GPU settings
nvidia-smi -q -d SUPPORTED_CLOCKS

# Set persistence mode for better performance
sudo nvidia-smi -pm 1
```

### Memory Management

Configure memory settings for optimal performance:

```bash
# Check available memory
free -h

# Increase shared memory for Isaac Sim
sudo mount -o remount,size=8G /dev/shm
```

## Testing the Installation

Create a simple test to verify Isaac tools work correctly:

```bash
# Create test directory
mkdir -p ~/isaac_test
cd ~/isaac_test

# Create a simple test script
cat << 'EOF' > test_isaac.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

class IsaacTestNode(Node):
    def __init__(self):
        super().__init__('isaac_test_node')
        self.publisher = self.create_publisher(PointCloud2, 'test_pointcloud', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info('Isaac test node started')

    def timer_callback(self):
        # This is just a placeholder - actual Isaac functionality will be more complex
        self.get_logger().info('Isaac test node running')

def main(args=None):
    rclpy.init(args=args)
    node = IsaacTestNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
EOF

chmod +x test_isaac.py
```

## Next Steps

Continue to [Isaac Sim](./isaac-sim.md) to learn about creating realistic simulation environments for your humanoid robot.