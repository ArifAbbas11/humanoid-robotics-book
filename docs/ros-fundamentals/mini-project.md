# Mini-Project: Creating a ROS 2 Package

## Overview

In this mini-project, you'll create a complete ROS 2 package that controls a simulated robot.

## Prerequisites

- ROS 2 Humble Hawksbill installed
- Completed previous sections

## Step 1: Create the Package

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python my_robot_pkg --dependencies rclpy std_msgs geometry_msgs
```

## Step 2: Create a Publisher Node

Create `my_robot_pkg/my_robot_pkg/publisher_member_function.py`:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'robot_status', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Robot operational: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()
```

## Step 3: Update setup.py

Update `setup.py` to include your executable:

```python
entry_points={
    'console_scripts': [
        'talker = my_robot_pkg.publisher_member_function:main',
    ],
},
```

## Step 4: Build and Run

```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_pkg
source install/setup.bash
ros2 run my_robot_pkg talker
```

## Next Steps

Continue to [Troubleshooting](./troubleshooting.md) for common issues and solutions.