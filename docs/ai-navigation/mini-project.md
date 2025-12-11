# Mini-Project: Implementing Navigation for a Humanoid Robot

## Overview

In this mini-project, you'll implement a complete navigation system for a humanoid robot that includes path planning, localization, mapping, and control. You'll create a robot that can navigate to specified goals while avoiding obstacles and maintaining balance.

## Prerequisites

- Completed previous AI navigation sections
- ROS 2 Humble installed
- Gazebo simulation environment
- Basic understanding of humanoid kinematics

## Step 1: Create the Robot Model with Navigation Sensors

Create `humanoid_navigation.urdf.xacro`:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_navigator">

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.6"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.6"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="50"/>
      <inertia ixx="2.5" ixy="0" ixz="0" iyy="3.5" iyz="0" izz="1.5"/>
    </inertial>
  </link>

  <!-- Head with camera -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.004" ixy="0" ixz="0" iyy="0.004" iyz="0" izz="0.004"/>
    </inertial>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1"/>
  </joint>

  <!-- Camera in the head -->
  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 0.8"/>
      </material>
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
    <parent link="head"/>
    <child link="camera_link"/>
    <origin xyz="0.05 0 0" rpy="0 0 0"/>
  </joint>

  <!-- LIDAR on top of head -->
  <link name="lidar_link">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.05"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.0002" ixy="0" ixz="0" iyy="0.0002" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="lidar_joint" type="fixed">
    <parent link="head"/>
    <child link="lidar_link"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
  </joint>

  <!-- IMU in the torso -->
  <link name="imu_link">
    <visual>
      <geometry>
        <box size="0.02 0.02 0.02"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.02 0.02 0.02"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.000001" ixy="0" ixz="0" iyy="0.000001" iyz="0" izz="0.000001"/>
    </inertial>
  </link>

  <joint name="imu_joint" type="fixed">
    <parent link="base_link"/>
    <child link="imu_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

  <!-- Left leg -->
  <link name="left_hip">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>
  </link>

  <joint name="left_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_hip"/>
    <origin xyz="0 0.1 -0.3" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="2"/>
  </joint>

  <link name="left_knee">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.3"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="3"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="left_knee_joint" type="revolute">
    <parent link="left_hip"/>
    <child link="left_knee"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="2.35" effort="100" velocity="2"/>
  </joint>

  <link name="left_ankle">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="left_ankle_joint" type="revolute">
    <parent link="left_knee"/>
    <child link="left_ankle"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="2"/>
  </joint>

  <!-- Right leg -->
  <link name="right_hip">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>
  </link>

  <joint name="right_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_hip"/>
    <origin xyz="0 -0.1 -0.3" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="2"/>
  </joint>

  <link name="right_knee">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.3"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="3"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="right_knee_joint" type="revolute">
    <parent link="right_hip"/>
    <child link="right_knee"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="2.35" effort="100" velocity="2"/>
  </joint>

  <link name="right_ankle">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="right_ankle_joint" type="revolute">
    <parent link="right_knee"/>
    <child link="right_ankle"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="2"/>
  </joint>

  <!-- Gazebo plugins -->
  <gazebo reference="camera_link">
    <sensor type="camera" name="camera1">
      <update_rate>30.0</update_rate>
      <camera name="head">
        <horizontal_fov>1.3962634</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
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
        <frame_name>camera_link</frame_name>
        <topic_name>camera/image_raw</topic_name>
      </plugin>
    </sensor>
  </gazebo>

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

  <gazebo reference="imu_link">
    <sensor type="imu" name="imu_sensor">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <visualize>false</visualize>
      <plugin filename="libgazebo_ros_imu_sensor.so" name="imu_plugin">
        <topicName>imu</topicName>
        <bodyName>base_link</bodyName>
        <updateRateHZ>100.0</updateRateHZ>
        <gaussianNoise>0.01</gaussianNoise>
        <xyzOffset>0 0 0</xyzOffset>
        <rpyOffset>0 0 0</rpyOffset>
        <frameName>imu_link</frameName>
      </plugin>
    </sensor>
  </gazebo>

  <!-- Control plugin for joints -->
  <gazebo>
    <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
      <update_rate>30</update_rate>
      <joint_name>left_hip_joint</joint_name>
      <joint_name>left_knee_joint</joint_name>
      <joint_name>left_ankle_joint</joint_name>
      <joint_name>right_hip_joint</joint_name>
      <joint_name>right_knee_joint</joint_name>
      <joint_name>right_ankle_joint</joint_name>
      <joint_name>neck_joint</joint_name>
    </plugin>
  </gazebo>

</robot>
```

## Step 2: Create Navigation Configuration

Create `config/navigation.yaml`:

```yaml
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_link"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    likelihood_max_dist: 2.0
    set_initial_pose: true
    initial_pose:
      x: 0.0
      y: 0.0
      z: 0.0
      yaw: 0.0
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.25
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05
    scan_topic: scan

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    default_bt_xml_filename: "navigate_w_replanning_and_recovery.xml"
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_goal_reached_condition_bt_node
    - nav2_goal_updated_condition_bt_node
    - nav2_initial_pose_received_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_distance_controller_bt_node
    - nav2_speed_controller_bt_node
    - nav2_truncate_path_action_bt_node
    - nav2_goal_updater_node_bt_node
    - nav2_recovery_node_bt_node
    - nav2_pipeline_sequence_bt_node
    - nav2_round_robin_node_bt_node
    - nav2_transform_available_condition_bt_node
    - nav2_time_expired_condition_bt_node
    - nav2_path_expiring_timer_condition
    - nav2_distance_traveled_condition_bt_node
    - nav2_single_trigger_bt_node
    - nav2_is_battery_low_condition_bt_node
    - nav2_navigate_through_poses_action_bt_node
    - nav2_navigate_to_pose_action_bt_node
    - nav2_remove_passed_goals_action_bt_node
    - nav2_planner_selector_bt_node
    - nav2_controller_selector_bt_node
    - nav2_goal_checker_selector_bt_node

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Humanoid-specific controller
    FollowPath:
      plugin: "humanoid_controller/HumanoidController"
      # Balance control parameters
      balance_kp: 1.0
      balance_ki: 0.1
      balance_kd: 0.05
      # Walking parameters
      step_height: 0.05
      step_length: 0.3
      step_timing: 0.8
      max_linear_speed: 0.5
      max_angular_speed: 0.5

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: True
      rolling_window: true
      width: 10
      height: 10
      resolution: 0.05
      robot_radius: 0.3
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: False
        origin_z: 0.0
        z_resolution: 0.05
        z_voxels: 16
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: True
      robot_radius: 0.3
      resolution: 0.05
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true
```

## Step 3: Create the Navigation Controller

Create `humanoid_controller.py`:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Imu
from tf2_ros import TransformBroadcaster
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Parameters
        self.declare_parameter('balance_kp', 1.0)
        self.declare_parameter('balance_ki', 0.1)
        self.declare_parameter('balance_kd', 0.05)
        self.declare_parameter('step_height', 0.05)
        self.declare_parameter('step_length', 0.3)
        self.declare_parameter('step_timing', 0.8)
        self.declare_parameter('max_linear_speed', 0.5)
        self.declare_parameter('max_angular_speed', 0.5)

        self.balance_kp = self.get_parameter('balance_kp').value
        self.balance_ki = self.get_parameter('balance_ki').value
        self.balance_kd = self.get_parameter('balance_kd').value
        self.step_height = self.get_parameter('step_height').value
        self.step_length = self.get_parameter('step_length').value
        self.step_timing = self.get_parameter('step_timing').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value

        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            'imu',
            self.imu_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel_out', 10)

        # Controller state
        self.current_pose = None
        self.current_twist = None
        self.imu_data = None
        self.scan_data = None
        self.balance_error_sum = 0.0
        self.prev_balance_error = 0.0

        # Timer for control loop
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20Hz

        self.get_logger().info('Humanoid Navigation Controller initialized')

    def cmd_vel_callback(self, msg):
        """Handle velocity commands from navigation system"""
        self.desired_twist = msg

    def odom_callback(self, msg):
        """Handle odometry data"""
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

    def imu_callback(self, msg):
        """Handle IMU data for balance control"""
        self.imu_data = msg

    def scan_callback(self, msg):
        """Handle laser scan data for obstacle detection"""
        self.scan_data = msg

    def compute_balance_control(self):
        """Compute balance control based on IMU data"""
        if self.imu_data is None:
            return 0.0

        # Extract roll and pitch from IMU quaternion
        orientation = self.imu_data.orientation
        r = R.from_quat([
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        ])
        roll, pitch, yaw = r.as_euler('xyz')

        # Balance error (we want to maintain upright position)
        balance_error = pitch  # Using pitch as primary balance measure

        # PID control for balance
        self.balance_error_sum += balance_error * 0.05  # dt = 0.05s
        balance_error_derivative = (balance_error - self.prev_balance_error) / 0.05

        balance_control = (
            self.balance_kp * balance_error +
            self.balance_ki * self.balance_error_sum +
            self.balance_kd * balance_error_derivative
        )

        self.prev_balance_error = balance_error
        return balance_control

    def check_obstacles(self):
        """Check for obstacles in the path"""
        if self.scan_data is None:
            return False

        # Check forward direction (front 30 degrees)
        ranges = self.scan_data.ranges
        angle_min = self.scan_data.angle_min
        angle_increment = self.scan_data.angle_increment

        # Get indices for front 30 degrees
        front_start = int((0 - 15 * math.pi/180 - angle_min) / angle_increment)
        front_end = int((0 + 15 * math.pi/180 - angle_min) / angle_increment)

        if 0 <= front_start < len(ranges) and 0 <= front_end < len(ranges):
            front_ranges = ranges[front_start:front_end]
            min_range = min(front_ranges) if front_ranges else float('inf')

            # If obstacle is closer than 0.5m, consider it a problem
            return min_range < 0.5

        return False

    def control_loop(self):
        """Main control loop"""
        if self.current_pose is None or self.desired_twist is None:
            return

        cmd_vel = Twist()

        # Check for obstacles and adjust if needed
        if self.check_obstacles():
            # Stop and rotate to find clear path
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.5  # Rotate to find clear path
        else:
            # Apply desired velocity with limits
            cmd_vel.linear.x = max(-self.max_linear_speed,
                                  min(self.max_linear_speed, self.desired_twist.linear.x))
            cmd_vel.angular.z = max(-self.max_angular_speed,
                                   min(self.max_angular_speed, self.desired_twist.angular.z))

        # Apply balance control to angular velocity
        balance_correction = self.compute_balance_control()
        cmd_vel.angular.z += balance_correction

        # Publish the command
        self.cmd_vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    controller = HumanoidController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 4: Create the Launch File

Create `launch/humanoid_navigation.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from nav2_common.launch import RewrittenYaml

def generate_launch_description():
    ld = LaunchDescription()

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    params_file = LaunchConfiguration('params_file')
    autostart = LaunchConfiguration('autostart', default='true')
    use_composition = LaunchConfiguration('use_composition', default='True')
    container_name = LaunchConfiguration('container_name', default='nav2_container')

    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'))

    ld.add_action(DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('my_robot_pkg'),
            'config',
            'navigation.yaml'
        ]),
        description='Full path to the ROS2 parameters file to use for all launched nodes'))

    ld.add_action(DeclareLaunchArgument(
        'autostart',
        default_value='true',
        description='Automatically startup the nav2 stack'))

    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
    )
    ld.add_action(gazebo)

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'humanoid_navigator',
            '-x', '0', '-y', '0', '-z', '0.5'
        ],
        output='screen'
    )
    ld.add_action(spawn_entity)

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    ld.add_action(robot_state_publisher)

    # Joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    ld.add_action(joint_state_publisher)

    # Navigation2
    navigation2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'),
                'launch',
                'navigation_launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': params_file,
            'autostart': autostart,
            'use_composition': use_composition,
            'container_name': container_name
        }.items()
    )
    ld.add_action(navigation2)

    # SLAM Toolbox (for mapping)
    slam_toolbox = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('my_robot_pkg'),
                'config',
                'slam.yaml'
            ])
        ],
        output='screen'
    )
    ld.add_action(slam_toolbox)

    # Humanoid controller
    humanoid_controller = Node(
        package='my_robot_pkg',
        executable='humanoid_controller',
        name='humanoid_controller',
        parameters=[{
            'use_sim_time': use_sim_time,
            'balance_kp': 1.0,
            'balance_ki': 0.1,
            'balance_kd': 0.05,
            'step_height': 0.05,
            'step_length': 0.3,
            'step_timing': 0.8
        }],
        output='screen'
    )
    ld.add_action(humanoid_controller)

    # RViz
    rviz_config = PathJoinSubstitution([
        FindPackageShare('my_robot_pkg'),
        'rviz',
        'navigation.rviz'
    ])
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}]
    )
    ld.add_action(rviz)

    return ld
```

## Step 5: Build and Run the Navigation System

1. Build your ROS 2 package:
```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_pkg
source install/setup.bash
```

2. Launch the navigation system:
```bash
ros2 launch my_robot_pkg humanoid_navigation.launch.py
```

3. In another terminal, send a navigation goal:
```bash
# Use RViz2 to set a 2D Nav Goal, or send a goal programmatically
ros2 run nav2_msgs action_client
```

## Expected Results

- Robot model appears in Gazebo with all sensors
- Navigation system initializes and waits for goals
- Robot plans paths and executes navigation while maintaining balance
- Obstacle avoidance works when obstacles are detected
- Robot successfully reaches specified goals

## Troubleshooting

If the navigation system doesn't work properly:

1. Check that all required ROS 2 packages are installed
2. Verify sensor data is being published correctly
3. Check that TF transforms are properly configured
4. Ensure the robot model has appropriate physical properties
5. Verify navigation parameters are tuned for humanoid locomotion

## Next Steps

Continue to [Troubleshooting](./troubleshooting.md) to learn about common navigation issues and solutions.