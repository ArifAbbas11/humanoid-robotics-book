# Python Integration with rclpy

## Overview

rclpy is the Python client library for ROS 2. It enables Python programmers to quickly interface with ROS.

## Installation

rclpy is included when you install ROS 2. Make sure you have sourced your ROS 2 installation:

```bash
source /opt/ros/humble/setup.bash
```

## Creating a Python Node

Here's a basic Python node example:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Running the Node

To run your Python node:

```bash
python3 your_node_file.py
```

## Next Steps

Continue to [URDF Basics](./urdf-basics.md) to learn about robot modeling.