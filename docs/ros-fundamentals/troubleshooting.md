# Troubleshooting ROS 2 Issues

## Overview

This guide covers common issues encountered when working with ROS 2 and provides solutions to help you debug and resolve problems efficiently.

## Common ROS 2 Issues

### Network and Communication Issues

**Issue**: Nodes cannot communicate across different machines.

**Solutions**:
1. Check network connectivity:
   ```bash
   ping <other_machine_ip>
   ```

2. Verify ROS domain ID:
   ```bash
   echo $ROS_DOMAIN_ID
   # Both machines should have the same domain ID
   export ROS_DOMAIN_ID=0
   ```

3. Check firewall settings to ensure ROS ports are open (typically UDP/TCP on ports 11000+)

4. Set proper ROS environment variables:
   ```bash
   export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp  # or your chosen middleware
   ```

**Issue**: `localhost` vs network communication problems.

**Solutions**:
1. Set ROS localhost only environment variable:
   ```bash
   export ROS_LOCALHOST_ONLY=0  # Allow network communication
   ```

2. Configure network interfaces properly

### Package and Dependency Issues

**Issue**: Package not found when building or running.

**Solutions**:
1. Check if the package is in the correct directory (`~/ros2_ws/src/`)
2. Ensure proper package.xml and CMakeLists.txt files exist
3. Source the ROS 2 installation:
   ```bash
   source /opt/ros/humble/setup.bash
   ```
4. Source your workspace:
   ```bash
   source install/setup.bash
   ```

**Issue**: Build fails with dependency errors.

**Solutions**:
1. Install missing dependencies:
   ```bash
   rosdep install --from-paths src --ignore-src -r -y
   ```
2. Check package.xml for correct dependencies
3. Verify CMakeLists.txt has proper find_package calls

### Node and Topic Issues

**Issue**: Nodes fail to start or crash immediately.

**Solutions**:
1. Check console output for error messages
2. Verify all required parameters are provided
3. Ensure proper ROS 2 installation and sourcing
4. Check for conflicting node names

**Issue**: Topics not publishing or subscribing correctly.

**Solutions**:
1. List active topics:
   ```bash
   ros2 topic list
   ```
2. Check topic type:
   ```bash
   ros2 topic type <topic_name>
   ```
3. Echo topic data:
   ```bash
   ros2 topic echo <topic_name>
   ```
4. Verify publisher and subscriber QoS profiles match

### Service and Action Issues

**Issue**: Services not responding or clients timing out.

**Solutions**:
1. Verify service server is running:
   ```bash
   ros2 service list
   ```
2. Check service type:
   ```bash
   ros2 service type <service_name>
   ```
3. Test service call:
   ```bash
   ros2 service call <service_name> <service_type> <request_data>
   ```

## Python-Specific Issues

### Import Errors

**Issue**: `ModuleNotFoundError` when importing ROS 2 modules.

**Solutions**:
1. Ensure ROS 2 is properly sourced
2. Check Python version compatibility (ROS 2 Humble requires Python 3.8+)
3. Verify rclpy is installed:
   ```bash
   pip3 list | grep rclpy
   ```

### Node Lifecycle Issues

**Issue**: Node crashes or behaves unexpectedly.

**Solutions**:
1. Add proper exception handling:
   ```python
   try:
       rclpy.spin(node)
   except KeyboardInterrupt:
       pass
   finally:
       node.destroy_node()
       rclpy.shutdown()
   ```

2. Check for proper parameter declarations
3. Verify all callbacks are properly defined

## Performance Issues

### High CPU Usage

**Issue**: ROS 2 nodes consuming excessive CPU resources.

**Solutions**:
1. Check timer frequencies in your code
2. Optimize callback execution time
3. Use appropriate QoS profiles
4. Consider using multi-threading for CPU-intensive tasks

### Memory Leaks

**Issue**: Memory usage increases over time.

**Solutions**:
1. Properly destroy nodes and publishers/subscribers
2. Check for circular references in callbacks
3. Monitor memory usage with tools like `htop`
4. Implement proper cleanup in destructor methods

## Debugging Strategies

### Logging and Monitoring

Use ROS 2 logging effectively:

```python
# In Python nodes
self.get_logger().info('Information message')
self.get_logger().warn('Warning message')
self.get_logger().error('Error message')
self.get_logger().debug('Debug message')  # Requires logging level set to debug
```

Monitor system state:
```bash
# View all nodes
ros2 node list

# View node graphically
ros2 run rqt_graph rqt_graph

# Monitor parameters
ros2 param list <node_name>

# View node info
ros2 node info <node_name>
```

### Debugging with GDB

For C++ nodes, use GDB for debugging:
```bash
gdb --args ros2 run <package_name> <executable_name>
```

## Environment Setup Issues

### Sourcing Problems

**Issue**: Commands not found or environment variables not set.

**Solutions**:
1. Permanently add sourcing to your bashrc:
   ```bash
   echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
   echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
   ```

2. Verify environment variables:
   ```bash
   env | grep ROS
   ```

### Workspace Issues

**Issue**: Workspace not building properly.

**Solutions**:
1. Clean build directory:
   ```bash
   rm -rf build/ install/ log/
   ```

2. Build with verbose output:
   ```bash
   colcon build --packages-select <package_name> --event-handlers console_direct+
   ```

3. Check for mixed ROS 1/ROS 2 installations

## Common Error Messages

### "Failed to create subscription" or "Failed to create publisher"

**Cause**: Usually related to message type issues or QoS profile mismatches.

**Solutions**:
1. Verify message type is properly imported
2. Check QoS profile compatibility between publisher and subscriber
3. Ensure message packages are properly built and sourced

### "Node name already exists"

**Cause**: Attempting to create a node with a name already in use.

**Solutions**:
1. Use unique node names
2. Kill existing nodes:
   ```bash
   killall -9 <node_executable_name>
   ```
3. Use node namespaces to avoid conflicts

### "Could not find a required package"

**Cause**: Missing dependencies or incorrect package path.

**Solutions**:
1. Update package list:
   ```bash
   sudo apt update
   ```
2. Install missing packages:
   ```bash
   sudo apt install ros-humble-<package_name>
   ```
3. Check package.xml for correct dependency names

## Best Practices for Troubleshooting

### Systematic Debugging

1. **Isolate the problem**: Test components individually
2. **Check the basics**: Ensure ROS 2 is sourced, network is working
3. **Look for patterns**: Check if issues occur in specific conditions
4. **Document solutions**: Keep notes on what worked for future reference

### Testing Strategies

1. **Unit testing**: Test individual functions and methods
2. **Integration testing**: Test component interactions
3. **System testing**: Test complete system behavior
4. **Regression testing**: Ensure fixes don't break existing functionality

## Getting Help

When seeking help with ROS 2 issues:

1. **Provide system information**: ROS 2 version, OS, Python version
2. **Include error messages**: Full error output with context
3. **Show your code**: Relevant code snippets that cause the issue
4. **Describe expected vs. actual behavior**: What you expected vs. what happened
5. **List steps to reproduce**: Clear sequence of actions leading to the issue

### Resources

- [ROS 2 Documentation](https://docs.ros.org/en/humble/)
- [ROS Answers](https://answers.ros.org/questions/)
- [ROS Discourse](https://discourse.ros.org/)
- [GitHub Issues](https://github.com/ros2/ros2/issues)

## Next Steps

Continue to [Module 2: The Digital Twin (Gazebo & Unity)](../simulation/intro.md) to learn about simulation environments.