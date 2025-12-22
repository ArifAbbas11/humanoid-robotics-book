# URDF Basics

## Overview

URDF (Unified Robot Description Format) is an XML format for representing a robot model. URDF is used in ROS to represent many different aspects of a robot including the physical structure, visual appearance, and kinematics.

## Basic Structure

A basic URDF file looks like this:

```xml
<?xml version="1.0"?>
<robot name="my_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.5"/>
      </geometry>
    </visual>
  </link>
</robot>
```

## Links and Joints

- **Links**: Represent rigid parts of the robot
- **Joints**: Connect links and define their motion

### Example with Joint:

```xml
<?xml version="1.0"?>
<robot name="two_links">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.5"/>
      </geometry>
    </visual>
  </link>

  <joint name="joint1" type="continuous">
    <parent link="base_link"/>
    <child link="child_link"/>
  </joint>

  <link name="child_link">
    <visual>
      <geometry>
        <box size="0.2 0.2 0.2"/>
      </geometry>
    </visual>
  </link>
</robot>
```

## Properties

URDF can define:
- Visual properties (shape, color, texture)
- Collision properties
- Inertial properties
- Kinematic properties

## Next Steps

Continue to [Mini-Project](./mini-project.md) to apply what you've learned.