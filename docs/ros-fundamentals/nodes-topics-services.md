# Nodes, Topics, Services

## Nodes

A node is an executable that uses ROS 2 to communicate with other nodes. Let's look at a basic node example:

```python
import rclpy
from rclpy.node import Node

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
```

## Topics

Topics are named buses over which nodes exchange messages. Nodes can publish messages to a topic or subscribe to messages from a topic.

## Services

Services provide a request/reply communication pattern. A service client sends a request message to a service server, which processes the request and returns a response message.

## Example

Here's how to create a simple service:

```python
# Service server example
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response
```

## Next Steps

Continue to [Python Integration with rclpy](./python-integration.md) to learn about Python development with ROS 2.