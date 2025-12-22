# Mapping for Humanoid Robot Navigation

## Overview

Mapping is the process of creating a representation of the environment that can be used for navigation. For humanoid robots, mapping must account for the robot's unique characteristics such as height, balance requirements, and the need to navigate complex 3D environments with stairs, ramps, and other obstacles that wheeled robots don't encounter.

## Types of Maps

### 2D Occupancy Grid Maps

Occupancy grid maps divide the environment into discrete cells:

- **Binary Representation**: Each cell is either occupied or free
- **Probabilistic**: Each cell has a probability of occupancy
- **Resolution**: Grid cell size affects map detail and computational requirements

### 3D Volumetric Maps

For humanoid robots, 3D maps are often necessary:

- **Voxel Grids**: 3D grid-based representation
- **Point Clouds**: Dense 3D point representation
- **Mesh Maps**: Surface-based 3D representation

### Semantic Maps

Adding semantic information to geometric maps:

- **Object Labels**: Identifying doors, furniture, stairs
- **Functional Areas**: Rooms, corridors, obstacles
- **Traversability**: Distinguishing walkable from non-walkable areas

## Humanoid-Specific Mapping Considerations

### Multi-Level Mapping

Humanoid robots need maps that account for vertical navigation:

- **Stair Detection**: Identifying and mapping stairs for navigation
- **Ramp Mapping**: Recognizing gradual elevation changes
- **Multi-story Maps**: Connecting maps across different floors

### Height-Aware Mapping

Accounting for the robot's height and reach:

- **Clearance Mapping**: Ensuring sufficient headroom
- **Step Height Detection**: Identifying obstacles that affect walking
- **Reachable Areas**: Mapping areas accessible to the robot

### Traversability Analysis

Determining which areas are safe for humanoid navigation:

- **Surface Stability**: Identifying uneven or slippery surfaces
- **Obstacle Height**: Distinguishing obstacles that can be stepped over
- **Narrow Passages**: Ensuring sufficient width for bipedal locomotion

## Mapping Algorithms

### Occupancy Grid Mapping

Basic algorithm for creating 2D occupancy maps:

```python
import numpy as np
from scipy.ndimage import binary_dilation

class OccupancyGridMap:
    def __init__(self, width, height, resolution):
        self.resolution = resolution
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))  # -1: unknown, 0: free, 1: occupied
        self.log_odds = np.zeros((height, width))  # Log odds representation
        self.log_odds_free = np.log(0.3 / 0.7)  # Log odds for free space
        self.log_odds_occupied = np.log(0.9 / 0.1)  # Log odds for occupied space

    def update_cell(self, x, y, occupied):
        """Update a single cell based on sensor reading"""
        if 0 <= x < self.width and 0 <= y < self.height:
            if occupied:
                self.log_odds[y, x] += self.log_odds_occupied
            else:
                self.log_odds[y, x] += self.log_odds_free

            # Convert back to probability
            prob = 1 - 1 / (1 + np.exp(self.log_odds[y, x]))
            if prob > 0.7:
                self.grid[y, x] = 1  # occupied
            elif prob < 0.3:
                self.grid[y, x] = 0  # free
            else:
                self.grid[y, x] = -1  # unknown

    def ray_trace(self, start, end, occupied_end):
        """Perform ray tracing from sensor reading"""
        # Bresenham's line algorithm for ray tracing
        x0, y0 = int(start[0] / self.resolution), int(start[1] / self.resolution)
        x1, y1 = int(end[0] / self.resolution), int(end[1] / self.resolution)

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        # Mark endpoint as occupied or free
        if occupied_end:
            self.update_cell(x1, y1, True)
        else:
            self.update_cell(x1, y1, False)

        # Mark free space along the ray
        x, y = x0, y0
        while x != x1 or y != y1:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.update_cell(x, y, False)
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
```

### 3D Mapping

Extending mapping to three dimensions:

- **OctoMap**: Hierarchical 3D occupancy mapping
- **TSDF (Truncated Signed Distance Function)**: Surface reconstruction
- **Point Cloud Integration**: Combining multiple point cloud observations

### SLAM (Simultaneous Localization and Mapping)

SLAM algorithms perform mapping and localization simultaneously:

- **Graph-Based SLAM**: Optimization-based approach
- **Filter-Based SLAM**: Recursive Bayesian estimation
- **Keyframe-Based SLAM**: Using key poses for efficiency

## Sensor Integration for Mapping

### LIDAR Mapping

LIDAR provides accurate range measurements:

- **Scan Registration**: Aligning scans using odometry or IMU
- **Loop Closure**: Detecting revisited locations
- **Multi-Beam LIDAR**: Handling 3D LIDAR data

### Visual Mapping

Camera-based mapping techniques:

- **Visual SLAM**: Feature-based mapping using cameras
- **Direct Methods**: Using pixel intensities instead of features
- **Structure from Motion**: Reconstructing 3D structure from multiple views

### Multi-Sensor Mapping

Combining multiple sensor types:

- **Sensor Fusion**: Integrating data from different sensors
- **Complementary Sensors**: Using different sensors for different aspects
- **Redundancy**: Multiple sensors for robust mapping

## ROS 2 Mapping Tools

### Navigation2 Map Server

The Navigation2 stack includes mapping capabilities:

```yaml
# Example map server configuration
map_saver:
  ros__parameters:
    save_map_timeout: 5.0
    free_thresh_default: 0.25
    occupied_thresh_default: 0.65
    map_subscribe_transient_local: True
```

### Cartographer

Google's SLAM library integrated with ROS 2:

```lua
-- Example Cartographer configuration for humanoid robot
options = {
  tracking_frame = "base_link",
  published_frame = "odom",
  map_frame = "map",
  odom_frame = "odom",
  provide_odom_frame = true,
  use_odometry = true,
  use_nav_sat = false,
  use_landmarks = false,
  num_laser_scans = 1,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 0,
  lookup_transform_timeout_sec = 0.2,
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 5e-03,
  trajectory_publish_period_sec = 30e-03,
  rangefinder_sampling_ratio = 1.,
  odometry_sampling_ratio = 1.,
  fixed_frame_pose_sampling_ratio = 1.,
  imu_sampling_ratio = 1.,
  landmarks_sampling_ratio = 1.,
}
```

### Custom Mapping Nodes

Develop custom nodes for humanoid-specific mapping:

- **Height-Aware Mapping**: Creating maps that consider robot height
- **Traversability Mapping**: Distinguishing walkable from non-walkable areas
- **Stair Mapping**: Specialized algorithms for stair detection and mapping

## Mapping Challenges

### Dynamic Environments

Handling moving objects in the environment:

- **Dynamic Object Filtering**: Removing moving objects from static maps
- **Temporary Obstacles**: Distinguishing between permanent and temporary obstacles
- **Moving Object Tracking**: Tracking dynamic obstacles separately

### Large-Scale Mapping

Managing maps of large environments:

- **Map Tiling**: Dividing large maps into smaller tiles
- **Memory Management**: Efficient storage and retrieval of map data
- **Map Streaming**: Loading only necessary map portions

### Computational Efficiency

Balancing map quality and computational requirements:

- **Multi-Resolution Maps**: Different detail levels for different needs
- **Incremental Updates**: Updating only changed portions of the map
- **Parallel Processing**: Using multiple cores for mapping operations

## Semantic Mapping

### Object Recognition Integration

Adding semantic information to geometric maps:

- **Deep Learning**: Using CNNs for object recognition
- **Instance Segmentation**: Distinguishing individual objects
- **Semantic Labels**: Associating labels with map regions

### Functional Area Mapping

Identifying areas with specific functions:

- **Room Detection**: Identifying and labeling rooms
- **Path Planning Areas**: Marking corridors and open spaces
- **Safety Zones**: Identifying safe areas for robot operation

## Quality Assessment

### Map Accuracy

Evaluating map quality:

- **Geometric Accuracy**: How well the map matches the real environment
- **Completeness**: Coverage of the environment
- **Consistency**: Internal consistency of the map

### Map Validation

Verifying map correctness:

- **Ground Truth Comparison**: Comparing with known maps when available
- **Navigation Performance**: Testing navigation using the map
- **Loop Closure Detection**: Verifying that loops are properly closed

## Best Practices

### Map Management

Effective map handling strategies:

- **Map Updates**: Regularly updating maps as the environment changes
- **Map Merging**: Combining maps from different sessions
- **Map Compression**: Efficiently storing large maps

### Sensor Configuration

Optimal sensor setup for mapping:

- **Sensor Placement**: Positioning sensors for optimal coverage
- **Sensor Fusion**: Combining multiple sensors for robust mapping
- **Calibration**: Properly calibrating all sensors

### Incremental Mapping

Building maps incrementally:

- **Local Mapping**: Creating local maps and combining them
- **Global Optimization**: Periodically optimizing the entire map
- **Consistency Maintenance**: Ensuring map consistency over time

## Next Steps

Continue to [Control Systems](./control-systems.md) to learn about controlling humanoid robot navigation.