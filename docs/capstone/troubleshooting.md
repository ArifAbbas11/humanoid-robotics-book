# Troubleshooting

## Overview

Troubleshooting is an essential skill for humanoid robotics development. This guide provides systematic approaches to identify, diagnose, and resolve issues that arise in complex integrated systems. Effective troubleshooting requires understanding both individual components and their interactions.

## Systematic Troubleshooting Approach

### The Troubleshooting Process

```
1. Problem Identification → 2. Information Gathering → 3. Hypothesis Formation
         ↓
4. Testing Hypotheses → 5. Solution Implementation → 6. Verification
```

### Information Gathering

#### Log Analysis

```bash
# Essential ROS 2 troubleshooting commands
ros2 topic list                    # List all active topics
ros2 topic info /topic_name        # Get information about a topic
ros2 topic echo /topic_name        # Monitor topic messages
ros2 service list                  # List all services
ros2 action list                   # List all actions
ros2 node list                     # List all nodes
ros2 run rqt_graph rqt_graph       # Visualize node connections
```

#### System Monitoring

```bash
# Monitor system resources
htop                                # CPU and memory usage
nvidia-smi                         # GPU usage (if applicable)
iotop                              # I/O usage
iftop                              # Network usage

# ROS 2 specific monitoring
ros2 run tf2_tools view_frames     # Check TF tree
ros2 param list <node_name>        # Check node parameters
ros2 lifecycle list <node_name>    # Check lifecycle nodes
```

## Common Troubleshooting Scenarios

### Communication Issues

**Problem**: Nodes cannot communicate with each other.

**Diagnosis**:
1. Check if nodes are running:
   ```bash
   ros2 node list
   ros2 lifecycle list <node_name>  # for lifecycle nodes
   ```

2. Verify topic connections:
   ```bash
   ros2 topic info /topic_name
   ros2 topic hz /topic_name        # Check message rate
   ```

3. Check network configuration:
   ```bash
   echo $ROS_DOMAIN_ID
   echo $ROS_LOCALHOST_ONLY
   # Ensure both machines have same domain ID if on same network
   ```

**Solutions**:
1. Verify correct topic names and message types
2. Check QoS profile compatibility
3. Ensure proper network configuration for multi-machine setups
4. Check firewall settings for ROS 2 ports (typically UDP/TCP 11000+)

### Sensor Data Issues

**Problem**: Sensor data is not being published or is incorrect.

**Diagnosis**:
1. Check sensor hardware:
   ```bash
   lsusb                            # List USB devices
   lspci | grep -i nvidia          # Check GPU
   ls /dev | grep camera           # Check camera devices
   ```

2. Monitor sensor topics:
   ```bash
   ros2 topic echo /sensor_topic
   ros2 run image_view image_view _image:=/camera/image_raw
   ```

3. Check sensor drivers:
   ```bash
   ros2 run rqt_reconfigure rqt_reconfigure
   ros2 param list <sensor_node_name>
   ```

**Solutions**:
1. Verify sensor calibration files
2. Check camera permissions: `sudo chmod 666 /dev/video*`
3. Ensure proper power and data connections
4. Update sensor drivers if necessary

### Navigation Issues

**Problem**: Robot fails to navigate or plans invalid paths.

**Diagnosis**:
```python
# Debug navigation with this approach
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped

class NavigationDebugger(Node):
    def __init__(self):
        super().__init__('navigation_debugger')

        # Subscribe to navigation topics
        self.path_sub = self.create_subscription(Path, '/plan', self.path_callback, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)

    def path_callback(self, msg):
        self.get_logger().info(f'Path received with {len(msg.poses)} waypoints')
        if len(msg.poses) == 0:
            self.get_logger().error('Received empty path!')

    def map_callback(self, msg):
        self.get_logger().info(f'Map: {msg.info.width}x{msg.info.height}, resolution: {msg.info.resolution}')

    def goal_callback(self, msg):
        self.get_logger().info(f'Goal received: ({msg.pose.position.x}, {msg.pose.position.y})')
```

**Solutions**:
1. Verify costmap parameters and inflation settings
2. Check TF tree for proper transforms
3. Ensure proper map resolution and inflation
4. Validate robot footprint configuration

### Perception Issues

**Problem**: Object detection or recognition fails.

**Diagnosis**:
```bash
# Check vision pipeline
ros2 run image_view image_view _image:=/camera/image_raw
ros2 topic echo /object_detections --field results
ros2 run rqt_image_view rqt_image_view
```

**Solutions**:
1. Verify camera calibration
2. Check lighting conditions
3. Adjust detection thresholds
4. Ensure proper image encoding and format

## Advanced Troubleshooting Techniques

### Debugging with GDB

For C++ nodes that crash:

```bash
# Run node with GDB
gdb --args ros2 run package_name node_name
(gdb) run
# When crash occurs:
(gdb) bt                    # Show backtrace
(gdb) info registers      # Show register state
(gdb) info locals         # Show local variables
```

### Profiling Performance Issues

```python
# Profile Python nodes
import cProfile
import pstats

def profile_node():
    pr = cProfile.Profile()
    pr.enable()

    # Run your node code here
    your_node_logic()

    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 time-consuming functions
```

### Memory Leak Detection

```bash
# Monitor memory usage
watch -n 1 'ps aux | grep ros2'
# Or use valgrind for C++ nodes
valgrind --tool=memcheck --leak-check=full ros2 run package_name node_name
```

## Component-Specific Troubleshooting

### Navigation Stack Issues

**Common Problems**:
1. **Local planner fails**: Check `local_costmap` inflation and update rates
2. **Global planner fails**: Verify `global_costmap` and map server
3. **Controller fails**: Check `controller_server` parameters

**Debug Configuration**:
```yaml
# navigation2.yaml - Debug configuration
local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom      # Ensure this matches your odometry frame
      robot_base_frame: base_link
      use_sim_time: True
      rolling_window: true
      width: 3
      height: 3
      resolution: 0.05       # Increase resolution if needed
      robot_radius: 0.2      # Match your robot's actual radius
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
```

### Vision System Issues

**Problem**: Camera images are distorted or incorrect.

**Solutions**:
1. **Check calibration**: Verify camera intrinsic and extrinsic parameters
2. **Update calibration**: Use `camera_calibration` package
3. **Check encoding**: Ensure proper image encoding (RGB8, BGR8, etc.)

```bash
# Verify camera calibration
ros2 run camera_calibration_parsers parse --file camera_info.yaml
# Check image format
ros2 topic info /camera/image_raw
```

### Control System Issues

**Problem**: Robot moves erratically or doesn't respond properly.

**Diagnosis**:
1. Check control loop frequency
2. Verify joint state feedback
3. Validate PID parameters

```python
# Control loop monitoring
class ControlMonitor(Node):
    def __init__(self):
        super().__init__('control_monitor')
        self.last_control_time = self.get_clock().now()

    def control_callback(self, msg):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_control_time).nanoseconds / 1e9
        self.last_control_time = current_time

        if dt > 0.1:  # Control loop should be faster than 10Hz
            self.get_logger().warn(f'Control loop period: {dt}s')
```

## Simulation-Specific Troubleshooting

### Gazebo Issues

**Problem**: Robot falls through ground or behaves unrealistically.

**Solutions**:
1. **Check URDF**: Verify mass, inertia, and collision properties
2. **Physics parameters**: Adjust update rate and solver settings
3. **Joint limits**: Ensure proper joint limits and dynamics

```xml
<!-- In URDF, ensure proper physical properties -->
<link name="link_name">
  <inertial>
    <mass value="1.0"/>  <!-- Proper mass value -->
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>  <!-- Proper inertia -->
  </inertial>
  <visual>
    <!-- Visual properties -->
  </visual>
  <collision>
    <!-- Collision properties -->
  </collision>
</link>
```

**Gazebo launch debugging**:
```bash
# Launch with verbose output
gazebo --verbose
# Launch with GUI
gazebo --gui
# Check model database
ls ~/.gazebo/models
```

### Isaac Sim Issues

**Problem**: Isaac Sim crashes or doesn't display properly.

**Solutions**:
1. **GPU requirements**: Ensure CUDA and GPU compatibility
2. **X11 forwarding**: Check display settings for Docker containers
3. **Memory limits**: Increase shared memory: `sudo mount -o remount,size=8G /dev/shm`

## ROS 2 Specific Troubleshooting

### Lifecycle Node Issues

**Problem**: Lifecycle nodes don't transition properly.

**Diagnosis**:
```bash
# Check lifecycle state
ros2 lifecycle list <node_name>
# Transition manually
ros2 lifecycle set <node_name> configure
ros2 lifecycle set <node_name> activate
```

### Parameter Issues

**Problem**: Parameters don't update or are incorrect.

**Solutions**:
```bash
# List parameters
ros2 param list <node_name>
# Get specific parameter
ros2 param get <node_name> param_name
# Set parameter
ros2 param set <node_name> param_name value
# Load from file
ros2 param load <node_name> params.yaml
```

### Action Server Issues

**Problem**: Action servers don't respond or goals are rejected.

**Diagnosis**:
```bash
# Check action servers
ros2 action list
# Check action types
ros2 action info /action_name
# Send test goal
ros2 action send_goal /action_name action_type "goal_data"
```

## Hardware Troubleshooting

### Sensor Calibration

**Camera Calibration**:
```bash
# For monocular camera
ros2 run camera_calibration cameracalibrator --size 8x6 --square 0.108 image:=/camera/image_raw camera:=/camera
# For stereo camera
ros2 run camera_calibration stereo_image_proc --approximate-time 0.1 left/image:=/left/image_raw right/image:=/right/image_raw left/camera_info:=/left/camera_info right/camera_info:=/right/camera_info
```

### IMU Issues

**Problem**: IMU data is noisy or biased.

**Solutions**:
1. **Calibration**: Perform IMU calibration procedures
2. **Filtering**: Apply appropriate filtering (complementary, Kalman)
3. **Mounting**: Ensure secure and proper mounting

### Joint Encoder Issues

**Problem**: Joint positions are incorrect or noisy.

**Diagnosis**:
```bash
# Monitor joint states
ros2 topic echo /joint_states
# Check for missing or noisy joints
# Verify joint limits and ranges
```

## Performance Troubleshooting

### High CPU Usage

**Diagnosis**:
1. Identify high-usage processes:
   ```bash
   htop
   # Look for ROS 2 processes using high CPU
   ```

2. Check message rates:
   ```bash
   ros2 topic hz /topic_name
   # High frequency topics can cause CPU issues
   ```

**Solutions**:
1. Reduce message rates where possible
2. Use efficient data structures
3. Implement proper throttling
4. Optimize algorithms

### Memory Issues

**Diagnosis**:
```bash
# Monitor memory usage
free -h
# Check specific ROS 2 processes
ps aux --sort=-%mem | grep ros2
```

**Solutions**:
1. Implement proper cleanup in destructors
2. Use memory pools for frequently allocated objects
3. Monitor for memory leaks using tools like Valgrind
4. Optimize data structures and message sizes

## Debugging Tools and Techniques

### Custom Debug Nodes

```python
# Create debugging utilities
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist

class SystemDebugger(Node):
    def __init__(self):
        super().__init__('system_debugger')

        # Create subscribers for all important topics
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.status_pub = self.create_publisher(String, '/debug_status', 10)

        # Timer for system health checks
        self.health_timer = self.create_timer(1.0, self.health_check)

    def joint_callback(self, msg):
        # Check for joint issues
        for i, name in enumerate(msg.name):
            position = msg.position[i] if i < len(msg.position) else 0
            velocity = msg.velocity[i] if i < len(msg.velocity) else 0
            effort = msg.effort[i] if i < len(msg.effort) else 0

            # Check for unusual values
            if abs(position) > 100:  # Likely an error
                self.get_logger().error(f'Unusual joint position for {name}: {position}')

    def cmd_vel_callback(self, msg):
        # Monitor command velocity
        if abs(msg.linear.x) > 2.0:  # Check for excessive velocity
            self.get_logger().warn(f'High linear velocity commanded: {msg.linear.x}')

    def health_check(self):
        # Overall system health check
        status_msg = String()
        status_msg.data = 'System operational'
        self.status_pub.publish(status_msg)
```

### Logging Best Practices

```python
import rclpy
from rclpy.node import Node
import logging

class WellLoggedNode(Node):
    def __init__(self):
        super().__init__('well_logged_node')

        # Set up different logging levels
        self.get_logger().set_level(logging.DEBUG)

        # Log initialization
        self.get_logger().info('Node initialized successfully')
        self.get_logger().debug('Debug information for development')

    def critical_operation(self):
        try:
            # Log important operations
            self.get_logger().info('Starting critical operation')

            # Perform operation
            result = self.perform_operation()

            self.get_logger().info('Critical operation completed successfully')
            return result

        except Exception as e:
            # Always log errors with context
            self.get_logger().error(f'Critical operation failed: {str(e)}')
            self.get_logger().error(f'Error type: {type(e).__name__}')
            return None
```

## Preventive Maintenance

### Regular System Checks

Create automated system health checks:

```bash
#!/bin/bash
# system_health_check.sh

echo "=== System Health Check ==="
echo "Date: $(date)"
echo "Uptime: $(uptime)"
echo "Load Average: $(uptime | awk -F'load average:' '{print $2}')"

# Check ROS 2
echo -e "\nROS 2 Status:"
if pgrep -f "ros2" > /dev/null; then
    echo "ROS 2 processes: $(pgrep -f ros2 | wc -l)"
else
    echo "No ROS 2 processes running"
fi

# Check disk space
echo -e "\nDisk Usage:"
df -h | grep -E '^(Filesystem|/dev)'

# Check memory
echo -e "\nMemory Usage:"
free -h

echo -e "\nCheck complete."
```

### Monitoring Scripts

```python
#!/usr/bin/env python3
# health_monitor.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import psutil
import time

class HealthMonitor(Node):
    def __init__(self):
        super().__init__('health_monitor')

        self.status_pub = self.create_publisher(String, '/system_status', 10)
        self.timer = self.create_timer(5.0, self.check_health)

    def check_health(self):
        status_msg = String()

        # Check system resources
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent

        status = f"CPU: {cpu_percent}%, Memory: {memory_percent}%, Disk: {disk_percent}%"

        # Check for issues
        issues = []
        if cpu_percent > 80:
            issues.append(f"High CPU usage: {cpu_percent}%")
        if memory_percent > 85:
            issues.append(f"High memory usage: {memory_percent}%")
        if disk_percent > 90:
            issues.append(f"Low disk space: {disk_percent}% used")

        if issues:
            status += f" [ISSUES: {', '.join(issues)}]"
            self.get_logger().warn(f"System issues detected: {', '.join(issues)}")
        else:
            self.get_logger().info("System health: All good")

        status_msg.data = status
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    monitor = HealthMonitor()

    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Troubleshooting Checklist

### Before Deployment
- [ ] All nodes start without errors
- [ ] TF tree is complete and correct
- [ ] Sensor data is being published and is valid
- [ ] Communication between nodes works properly
- [ ] Parameters are set correctly
- [ ] Safety limits are configured
- [ ] Emergency stop functions properly

### During Operation
- [ ] Monitor system resource usage
- [ ] Check for error messages in logs
- [ ] Verify sensor data quality
- [ ] Monitor robot behavior for anomalies
- [ ] Check communication health
- [ ] Verify safety systems are active

### Post-Operation
- [ ] Review logs for any issues
- [ ] Check for memory leaks or resource issues
- [ ] Update calibration if needed
- [ ] Document any issues encountered
- [ ] Plan improvements for next run

## Getting Help

### When to Seek Help

- Issues persist after systematic troubleshooting
- Problems affect safety systems
- Unknown error messages appear
- Performance is significantly below expectations
- Integration issues between components

### Resources

1. **ROS 2 Documentation**: https://docs.ros.org/
2. **ROS Answers**: https://answers.ros.org/
3. **GitHub Issues**: Check package repositories
4. **Community Forums**: Discourse, Reddit ROS communities
5. **Company/Institution Support**: Internal documentation and experts

### Providing Good Bug Reports

When seeking help, provide:
- **System information**: ROS 2 version, OS, hardware
- **Steps to reproduce**: Clear sequence of actions
- **Expected vs. actual**: What should happen vs. what happens
- **Error messages**: Complete error output
- **Configuration files**: Relevant parameter files
- **Log files**: Recent log entries
- **Minimal example**: Simplified case that reproduces the issue

## Next Steps

This completes the Capstone Project module. You now have comprehensive knowledge of integration, testing, and troubleshooting for humanoid robotics systems. Apply these concepts in your final project implementation and validation.