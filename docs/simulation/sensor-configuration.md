# Sensor Configuration

## Overview

Proper sensor configuration is essential for accurate robot perception in simulation. This section covers how to configure various sensors on your robot model to match real-world specifications and requirements.

## URDF Sensor Definitions

### Adding a Camera to URDF

```xml
<link name="camera_link">
  <visual>
    <geometry>
      <box size="0.05 0.05 0.05"/>
    </geometry>
  </visual>
  <collision>
    <geometry>
      <box size="0.05 0.05 0.05"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="0.1"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
  </inertial>
</link>

<joint name="camera_joint" type="fixed">
  <parent link="base_link"/>
  <child link="camera_link"/>
  <origin xyz="0.1 0 0.1" rpy="0 0 0"/>
</joint>

<gazebo reference="camera_link">
  <sensor type="camera" name="camera1">
    <update_rate>30.0</update_rate>
    <camera name="head">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>800</width>
        <height>600</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.02</near>
        <far>300</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_optical_frame</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

### Adding LIDAR to URDF

```xml
<link name="lidar_link">
  <visual>
    <geometry>
      <cylinder radius="0.05" length="0.05"/>
    </geometry>
  </visual>
  <collision>
    <geometry>
      <cylinder radius="0.05" length="0.05"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="0.2"/>
    <inertia ixx="0.0002" ixy="0" ixz="0" iyy="0.0002" iyz="0" izz="0.0001"/>
  </inertial>
</link>

<joint name="lidar_joint" type="fixed">
  <parent link="base_link"/>
  <child link="lidar_link"/>
  <origin xyz="0 0 0.15" rpy="0 0 0"/>
</joint>

<gazebo reference="lidar_link">
  <sensor type="ray" name="lidar_sensor">
    <pose>0 0 0 0 0 0</pose>
    <visualize>true</visualize>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.10</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.01</stddev>
      </noise>
    </ray>
    <plugin name="lidar_controller" filename="libgazebo_ros_laser.so">
      <topic_name>scan</topic_name>
      <frame_name>lidar_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

## Configuration Parameters

### Camera Configuration

Key parameters for camera sensors:

- **Resolution**: Image width and height in pixels
- **Field of View**: Horizontal and vertical viewing angles
- **Frame Rate**: Updates per second (Hz)
- **Image Format**: Color space (RGB8, RGBA8, etc.)

### LIDAR Configuration

Important LIDAR parameters:

- **Range**: Minimum and maximum detection distance
- **Angular Resolution**: Angle between consecutive measurements
- **Scan Frequency**: How often the scan is updated
- **Number of Rays**: Samples in horizontal plane

### IMU Configuration

IMU sensor parameters:

- **Update Rate**: Frequency of measurements
- **Noise Characteristics**: Bias, drift, and random noise
- **Measurement Range**: Maximum detectable values

## Coordinate Frames

### TF Tree Setup

Sensors require proper coordinate frame definitions:

```xml
<!-- Define optical frames for cameras -->
<joint name="camera_optical_joint" type="fixed">
  <parent link="camera_link"/>
  <child link="camera_optical_frame"/>
  <origin xyz="0 0 0" rpy="-1.570796 0 -1.570796"/>
</joint>

<link name="camera_optical_frame"/>
```

### Frame Conventions

- **Camera Frames**: Optical axis along +Z, X along image width, Y along image height
- **IMU Frames**: Typically aligned with robot body frame
- **LIDAR Frames**: Usually Z-axis up, X-forward

## ROS 2 Message Types

### Sensor Data Topics

Common ROS 2 message types for sensors:

- **Cameras**: `sensor_msgs/msg/Image` and `sensor_msgs/msg/CameraInfo`
- **LIDAR**: `sensor_msgs/msg/LaserScan` or `sensor_msgs/msg/PointCloud2`
- **IMU**: `sensor_msgs/msg/Imu`
- **Joint States**: `sensor_msgs/msg/JointState`

## Sensor Calibration

### Intrinsic Calibration

Camera intrinsic parameters:

```yaml
camera_matrix:
  rows: 3
  cols: 3
  data: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
distortion_coefficients:
  rows: 1
  cols: 5
  data: [k1, k2, p1, p2, k3]
```

### Extrinsic Calibration

Relationship between sensor and robot frames:

```yaml
# Transform between base_link and camera_link
translation: [0.1, 0.0, 0.1]
rotation: [0.0, 0.0, 0.0, 1.0]  # quaternion (x, y, z, w)
```

## Configuration Best Practices

### Performance Optimization

- Reduce sensor resolution if computational resources are limited
- Lower update rates for less time-critical sensors
- Use approximate filters for sensor fusion when precision isn't critical

### Accuracy Considerations

- Match simulation parameters to real sensor specifications
- Include realistic noise models based on datasheets
- Calibrate transforms between sensors and robot frames

### Validation

- Compare simulated sensor data with real sensor outputs
- Verify coordinate frame relationships
- Test edge cases and boundary conditions

## Troubleshooting

### Common Issues

- **Missing TF transforms**: Ensure all sensor frames are properly connected
- **Incorrect coordinate frames**: Verify orientation and position of sensors
- **Performance problems**: Reduce sensor fidelity if needed
- **Data quality**: Check noise parameters and range limits

## Next Steps

Continue to [Mini-Project](./mini-project.md) to apply sensor configuration concepts.