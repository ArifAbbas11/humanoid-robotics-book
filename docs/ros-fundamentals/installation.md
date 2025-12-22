# ROS 2 Installation Guide

This guide will walk you through installing ROS 2 Humble Hawksbill on your system.

## Prerequisites

- Ubuntu 22.04 LTS or equivalent Linux distribution
- At least 4GB of RAM recommended
- Administrative (sudo) access

## Installation Steps

1. Set up your sources list:
   ```bash
   sudo apt update && sudo apt install curl gnupg lsb-release
   sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
   ```

2. Add the repository to your sources list:
   ```bash
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
   ```

3. Install ROS 2 packages:
   ```bash
   sudo apt update
   sudo apt install ros-humble-desktop
   ```

4. Environment setup:
   ```bash
   echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
   source ~/.bashrc
   ```

## Verification

Test your installation:
```bash
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_cpp talker
```

## Next Steps

Continue to [ROS 2 Architecture Concepts](./architecture.md) to learn about the core concepts.