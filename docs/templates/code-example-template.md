# Code Example Template

This template provides a standard format for code examples throughout the book.

## Format

```markdown
### Example: [Brief description of what the code does]

**Purpose**: [Explain why this code is important and what problem it solves]

**Prerequisites**:
- [List any requirements before running this code]

**Code**:
```[language]
[Code snippet goes here]
```

**Explanation**:
- [Line-by-line or conceptual explanation of the code]
- [Highlight important concepts or patterns]

**Expected Output**:
```
[Example of what the output should look like]
```

**Testing**:
- [Instructions on how to verify the code works correctly]
- [Common troubleshooting tips if the code doesn't work as expected]

**Next Steps**:
- [How this code fits into the larger project]
- [What to do after running this code]
```

## Usage Guidelines

1. Always include a brief description of what the code does
2. Explain the purpose clearly for beginner-to-intermediate developers
3. Include expected output so readers know if they've run it correctly
4. Provide troubleshooting tips for common issues
5. Connect the example to the broader learning objectives

## Example

### Example: Basic ROS 2 Publisher

**Purpose**: This example demonstrates how to create a simple publisher node in ROS 2 that publishes messages to a topic.

**Prerequisites**:
- ROS 2 Humble Hawksbill installed
- Basic Python knowledge

**Code**:
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

**Explanation**:
- Creates a publisher that sends "Hello World" messages to the 'topic' topic
- Publishes every 0.5 seconds
- Increments a counter with each message

**Expected Output**:
```
[INFO] [1612345678.123456]: Publishing: "Hello World: 0"
[INFO] [1612345678.623456]: Publishing: "Hello World: 1"
[INFO] [1612345679.123456]: Publishing: "Hello World: 2"
```

**Testing**:
- Run the code and observe the output messages
- Verify that messages appear every 0.5 seconds
- If no output appears, check that ROS 2 is properly sourced

**Next Steps**:
- Learn how to create a subscriber to receive these messages
- Explore different message types in ROS 2