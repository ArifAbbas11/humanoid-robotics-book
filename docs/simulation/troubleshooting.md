# Troubleshooting Simulation Issues

## Overview

This guide covers common issues encountered when setting up and running robotics simulations, along with their solutions. Understanding these problems will help you debug simulation environments more efficiently.

## Common Gazebo Issues

### Gazebo Won't Start

**Symptoms**: Gazebo fails to launch or crashes immediately after starting.

**Solutions**:
1. Check graphics drivers:
   ```bash
   glxinfo | grep "OpenGL renderer"
   ```
2. Try running with software rendering:
   ```bash
   export LIBGL_ALWAYS_SOFTWARE=1
   gazebo
   ```
3. Ensure proper X11 forwarding if running remotely

### Robot Model Not Loading

**Symptoms**: Robot appears as a default shape or doesn't appear at all.

**Solutions**:
1. Verify URDF file syntax:
   ```bash
   check_urdf /path/to/robot.urdf
   ```
2. Check GAZEBO_MODEL_PATH:
   ```bash
   echo $GAZEBO_MODEL_PATH
   ```
3. Ensure all mesh files are accessible and properly referenced

### Physics Issues

**Symptoms**: Robot falls through the ground, unrealistic movements, or instability.

**Solutions**:
1. Check mass and inertia properties in URDF
2. Verify collision geometries match visual geometries
3. Adjust physics parameters in world file:
   ```xml
   <physics type="ode">
     <max_step_size>0.001</max_step_size>
     <real_time_factor>1</real_time_factor>
     <real_time_update_rate>1000</real_time_update_rate>
   </physics>
   ```

## ROS 2 Integration Issues

### TF Transform Problems

**Symptoms**: Sensor data appears in wrong coordinate frame, robot model parts misaligned.

**Solutions**:
1. Check TF tree:
   ```bash
   ros2 run tf2_tools view_frames
   ```
2. Verify joint definitions in URDF
3. Ensure robot_state_publisher is running:
   ```bash
   ros2 run robot_state_publisher robot_state_publisher --ros-args --params-file config/robot_params.yaml
   ```

### Sensor Data Issues

**Symptoms**: No sensor data published, or data doesn't match expectations.

**Solutions**:
1. Check available topics:
   ```bash
   ros2 topic list | grep sensor
   ```
2. Verify Gazebo plugins are loaded:
   ```bash
   ros2 node list
   ros2 topic echo /sensor_topic_name
   ```
3. Check plugin configuration in URDF

## Performance Issues

### Slow Simulation

**Symptoms**: Simulation runs slower than real-time, high CPU usage.

**Solutions**:
1. Reduce physics update rate in world file
2. Simplify collision meshes
3. Reduce sensor update rates
4. Use less complex physics engine settings

### High Memory Usage

**Symptoms**: System becomes unresponsive during simulation.

**Solutions**:
1. Reduce sensor resolution
2. Limit simulation world complexity
3. Close unnecessary applications during simulation

## Common URDF Issues

### Joint Limit Problems

**Symptoms**: Robot joints move beyond physical limits.

**Solutions**:
1. Verify joint limits in URDF:
   ```xml
   <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
   ```
2. Check controller configuration matches URDF

### Inertia Issues

**Symptoms**: Robot behaves unrealistically, unstable simulation.

**Solutions**:
1. Calculate proper inertia tensors:
   ```xml
   <inertial>
     <mass value="1.0"/>
     <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
   </inertial>
   ```
2. Use CAD software to calculate accurate values

## Sensor-Specific Issues

### Camera Problems

**Symptoms**: Black images, incorrect perspective, no data published.

**Solutions**:
1. Check camera parameters in URDF:
   ```xml
   <horizontal_fov>1.047</horizontal_fov>
   <image>
     <width>640</width>
     <height>480</height>
   </image>
   ```
2. Verify optical frame orientation (typically requires rotation)

### LIDAR Issues

**Symptoms**: Inconsistent ranges, missing data, incorrect angles.

**Solutions**:
1. Verify scan parameters:
   ```xml
   <ray>
     <scan>
       <horizontal>
         <samples>720</samples>
         <min_angle>-3.14</min_angle>
         <max_angle>3.14</max_angle>
       </horizontal>
     </scan>
   </ray>
   ```
2. Check for ray intersection with robot body

## Debugging Strategies

### Enable Verbose Output

Launch Gazebo with verbose output to see detailed error messages:
```bash
gazebo --verbose
```

### Use Gazebo GUI

Run with GUI to visually inspect the simulation:
```bash
gazebo --gui
```

### Check Model Files

Validate URDF files:
```bash
xmllint --noout robot.urdf
check_urdf robot.urdf
```

### Monitor System Resources

Watch CPU and memory usage:
```bash
htop
nvidia-smi  # for GPU monitoring
```

## Network and Communication Issues

### Connection Problems

**Symptoms**: ROS 2 nodes can't communicate with simulation.

**Solutions**:
1. Check ROS_DOMAIN_ID:
   ```bash
   echo $ROS_DOMAIN_ID
   ```
2. Verify network configuration
3. Ensure firewall allows ROS 2 traffic

### Plugin Loading Failures

**Symptoms**: Gazebo plugins fail to load.

**Solutions**:
1. Check plugin library paths
2. Verify plugin file permissions
3. Check for missing dependencies

## Simulation Quality Issues

### Unrealistic Behavior

**Symptoms**: Robot behaves differently than expected.

**Solutions**:
1. Validate physical parameters (mass, friction, damping)
2. Check contact properties
3. Verify controller parameters

### Inconsistent Results

**Symptoms**: Simulation behaves differently between runs.

**Solutions**:
1. Check for random seeds in physics engine
2. Verify initial conditions
3. Ensure deterministic controller behavior

## Tools for Debugging

### RViz Visualization

Use RViz to visualize sensor data and TF transforms:
```bash
ros2 run rviz2 rviz2
```

### Topic Monitoring

Monitor ROS 2 topics:
```bash
ros2 topic list
ros2 topic echo /topic_name
ros2 bag record /topic_name
```

### Gazebo Model Database

Check available models:
```bash
ls ~/.gazebo/models
```

## Prevention Best Practices

1. **Validate URDF**: Always check URDF files before simulation
2. **Start Simple**: Begin with basic models and add complexity gradually
3. **Document Configurations**: Keep track of working parameters
4. **Test Incrementally**: Test each component separately before integration
5. **Use Version Control**: Track changes to simulation configurations

## Getting Help

When seeking help with simulation issues:

1. Provide your system information (OS, ROS 2 version, Gazebo version)
2. Include relevant configuration files
3. Describe the expected vs. actual behavior
4. Share error messages or logs
5. Provide steps to reproduce the issue

## Next Steps

Continue to [AI Navigation](../ai-navigation/introduction.md) to learn about autonomous navigation for humanoid robots.