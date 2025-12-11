# Navigation Planning

## Overview

Navigation planning is the process of computing safe and efficient paths for humanoid robots to move from their current location to a desired goal. This involves considering the robot's unique kinematic constraints, balance requirements, and environmental obstacles to generate feasible trajectories.

## Types of Navigation Planning

### Global Path Planning

Global planners compute high-level routes from start to goal:

- **A* Algorithm**: Weighted graph search that balances path optimality and computation time
- **Dijkstra's Algorithm**: Guarantees optimal paths but can be computationally expensive
- **RRT (Rapidly-exploring Random Trees)**: Effective for high-dimensional spaces
- **PRM (Probabilistic Roadmaps)**: Pre-computed roadmap for multiple queries

### Local Path Planning

Local planners adjust paths in real-time to avoid dynamic obstacles:

- **Dynamic Window Approach (DWA)**: Considers robot dynamics and constraints
- **Trajectory Rollout**: Evaluates multiple potential trajectories
- **Potential Fields**: Uses attractive and repulsive forces
- **Model Predictive Control (MPC)**: Optimizes over a finite horizon

### Humanoid-Specific Planning

Unique to bipedal robots, footstep planning determines where to place feet:

- **Footstep Graphs**: Precomputed sets of feasible foot placements
- **Ankle-Height Functions**: Maintaining stable center of pressure
- **Stability Criteria**: Ensuring each step maintains balance

## Humanoid Navigation Challenges

### Balance and Stability

Humanoid robots must maintain balance during navigation:

- **Zero Moment Point (ZMP)**: Ensuring forces act within support polygon
- **Capture Point**: Predicting where to place feet to stop safely
- **Centroidal Dynamics**: Managing center of mass motion
- **Stability Margins**: Maintaining sufficient stability during movement

### Kinematic Constraints

Complex kinematic chains affect path planning:

- **Joint Limits**: Ensuring planned paths respect actuator constraints
- **Workspace Boundaries**: Avoiding configurations outside reachable space
- **Self-Collision Avoidance**: Preventing limbs from colliding during movement
- **Singularity Avoidance**: Preventing configurations with reduced mobility

### Gait Patterns

Different walking patterns for various scenarios:

- **Static Walking**: Stable at each step (slow but stable)
- **Dynamic Walking**: Continuous motion (faster but requires balance control)
- **Omnidirectional Walking**: Moving in any direction while maintaining balance
- **Terrain Adaptation**: Modifying gait for different surfaces

## Path Planning Algorithms

### Sampling-Based Methods

Effective for high-dimensional spaces:

```python
import numpy as np
from scipy.spatial.distance import euclidean

class HumanoidRRT:
    def __init__(self, start, goal, map_resolution=0.1):
        self.start = start
        self.goal = goal
        self.map_resolution = map_resolution
        self.nodes = [start]
        self.parent = {start: None}
        self.step_size = 0.2  # Adjust for humanoid step constraints

    def plan_path(self, max_iterations=1000):
        """Plan path using RRT algorithm with humanoid constraints"""
        for i in range(max_iterations):
            # Sample random configuration
            random_config = self.sample_free_space()

            # Find nearest node
            nearest_node = self.nearest_node(random_config)

            # Extend towards random configuration
            new_node = self.extend_towards(nearest_node, random_config)

            if new_node is not None:
                # Check if goal is reached
                if self.is_near_goal(new_node):
                    return self.extract_path(new_node)

                # Add to tree
                self.nodes.append(new_node)
                self.parent[new_node] = nearest_node

        return None  # No path found

    def sample_free_space(self):
        """Sample configuration space considering humanoid constraints"""
        while True:
            # Sample random position
            sample = np.random.random(2) * 10  # Simplified 2D case
            if self.is_valid_configuration(sample):
                return sample

    def is_valid_configuration(self, config):
        """Check for collisions and kinematic constraints"""
        return (not self.in_collision(config) and
                self.kinematically_feasible(config))

    def extend_towards(self, from_node, to_config):
        """Extend tree towards target configuration with humanoid constraints"""
        direction = to_config - from_node
        distance = np.linalg.norm(direction)

        if distance <= self.step_size:
            new_config = to_config
        else:
            # Normalize direction and scale by step size
            direction = direction / distance
            new_config = from_node + direction * self.step_size

        # Check if path between nodes is valid
        if self.is_valid_path(from_node, new_config):
            return new_config

        return None

    def is_valid_path(self, start, end):
        """Check if path between two configurations is valid"""
        # Check for collisions along the path
        steps = int(np.linalg.norm(end - start) / (self.map_resolution / 2))
        for i in range(1, steps + 1):
            intermediate = start + (end - start) * i / steps
            if not self.is_valid_configuration(intermediate):
                return False
        return True

    def nearest_node(self, config):
        """Find nearest node in tree"""
        distances = [euclidean(config, node) for node in self.nodes]
        return self.nodes[np.argmin(distances)]

    def is_near_goal(self, config):
        """Check if configuration is near goal"""
        return euclidean(config, self.goal) < self.step_size

    def extract_path(self, goal_node):
        """Extract path from goal back to start"""
        path = []
        current = goal_node
        while current is not None:
            path.append(current)
            current = self.parent[current]
        return path[::-1]  # Reverse to get start-to-goal path
```

### Optimization-Based Methods

Formulate path planning as an optimization problem:

```python
import numpy as np
from scipy.optimize import minimize

class OptimizationBasedPlanner:
    def __init__(self, start, goal, obstacles):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles

    def plan_path(self, num_waypoints=10):
        """Plan path using optimization"""
        # Initialize waypoints along straight line
        waypoints = np.linspace(self.start, self.goal, num_waypoints)
        waypoints = waypoints.reshape(-1)

        # Optimize path
        result = minimize(
            self.objective_function,
            waypoints,
            method='SLSQP',
            constraints=self.get_constraints(),
            options={'disp': True}
        )

        if result.success:
            optimized_path = result.x.reshape(-1, 2)
            return optimized_path
        return None

    def objective_function(self, waypoints):
        """Minimize path length and deviation from straight line"""
        waypoints = waypoints.reshape(-1, 2)

        # Path length cost
        length_cost = 0
        for i in range(1, len(waypoints)):
            length_cost += np.linalg.norm(waypoints[i] - waypoints[i-1])

        # Deviation from straight line cost
        straight_line_cost = 0
        for i, waypoint in enumerate(waypoints):
            expected_pos = self.start + (self.goal - self.start) * i / (len(waypoints) - 1)
            straight_line_cost += np.linalg.norm(waypoint - expected_pos)

        return length_cost + 0.1 * straight_line_cost

    def get_constraints(self):
        """Define constraints for optimization"""
        constraints = []

        # Start constraint
        def start_constraint(waypoints):
            waypoints = waypoints.reshape(-1, 2)
            return np.linalg.norm(waypoints[0] - self.start)

        # Goal constraint
        def goal_constraint(waypoints):
            waypoints = waypoints.reshape(-1, 2)
            return np.linalg.norm(waypoints[-1] - self.goal)

        constraints.append({'type': 'eq', 'fun': start_constraint})
        constraints.append({'type': 'eq', 'fun': goal_constraint})

        # Obstacle avoidance constraints
        for obs in self.obstacles:
            def obstacle_constraint(waypoints, obs=obs):
                waypoints = waypoints.reshape(-1, 2)
                for waypoint in waypoints:
                    if np.linalg.norm(waypoint - obs[:2]) < obs[2]:  # obs[2] is radius
                        return -1  # Violation
                return 1  # Satisfied

            constraints.append({'type': 'ineq', 'fun': obstacle_constraint})

        return constraints
```

## Footstep Planning

### Basic Footstep Planning

```python
class FootstepPlanner:
    def __init__(self, robot_params):
        self.step_length = robot_params['step_length']
        self.step_width = robot_params['step_width']
        self.max_step_height = robot_params['max_step_height']
        self.support_polygon = robot_params['support_polygon']

    def plan_footsteps(self, path, start_pose):
        """Plan footstep sequence for given path"""
        footsteps = []
        current_pose = start_pose.copy()

        for i in range(len(path) - 1):
            # Calculate required step
            next_pose = path[i + 1]
            step_vector = next_pose - current_pose[:2]

            # Plan footstep based on direction and distance
            if np.linalg.norm(step_vector) > 0.1:  # Minimum step threshold
                footstep = self.calculate_footstep(
                    current_pose, step_vector, len(footsteps)
                )
                footsteps.append(footstep)
                current_pose[:2] = next_pose

        return footsteps

    def calculate_footstep(self, current_pose, step_vector, step_count):
        """Calculate next footstep based on current pose and step vector"""
        # Determine foot placement based on walking pattern
        step_direction = step_vector / np.linalg.norm(step_vector)

        # Alternate between left and right foot
        if step_count % 2 == 0:
            # Left foot step
            foot_offset = np.array([-self.step_width/2, 0])
        else:
            # Right foot step
            foot_offset = np.array([self.step_width/2, 0])

        # Rotate offset based on step direction
        rotation_matrix = np.array([
            [step_direction[0], -step_direction[1]],
            [step_direction[1], step_direction[0]]
        ])
        foot_offset = rotation_matrix @ foot_offset

        # Calculate foot position
        foot_position = current_pose[:2] + step_direction * self.step_length + foot_offset

        # Add timing and other parameters
        footstep = {
            'position': foot_position,
            'orientation': np.arctan2(step_vector[1], step_vector[0]),
            'step_count': step_count,
            'timing': 0.8  # seconds
        }

        return footstep

    def validate_footstep(self, footstep, terrain_map):
        """Validate footstep for stability and terrain"""
        # Check if footstep is on stable terrain
        if not self.is_stable_terrain(footstep['position'], terrain_map):
            return False

        # Check if footstep maintains balance
        if not self.maintains_balance(footstep):
            return False

        return True

    def is_stable_terrain(self, position, terrain_map):
        """Check if terrain at position is stable for stepping"""
        # Implementation would check terrain properties
        return True

    def maintains_balance(self, footstep):
        """Check if footstep maintains robot balance"""
        # Implementation would check support polygon
        return True
```

## AI-Based Navigation Planning

### Deep Reinforcement Learning for Navigation

```python
import torch
import torch.nn as nn
import numpy as np

class NavigationPolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(NavigationPolicyNetwork, self).__init__()

        # Input: sensor data, goal direction, robot state
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)

        # Output: navigation action (velocity, angular velocity)
        self.action_head = nn.Linear(128, action_size)
        self.value_head = nn.Linear(128, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))

        action = torch.tanh(self.action_head(x))  # Actions in [-1, 1]
        value = self.value_head(x)

        return action, value

class DRLNavigationPlanner:
    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = NavigationPolicyNetwork(state_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)

    def get_action(self, state):
        """Get navigation action from policy network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, _ = self.policy_net(state_tensor)
        return action.cpu().data.numpy()[0]

    def train_step(self, states, actions, rewards, next_states, dones):
        """Perform one training step"""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_actions, current_values = self.policy_net(states)
        next_actions, next_values = self.policy_net(next_states)

        # Calculate loss (simplified for example)
        action_loss = nn.MSELoss()(current_actions, actions)
        value_loss = nn.MSELoss()(current_values, rewards)

        total_loss = action_loss + value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()
```

### Navigation with Neural Networks

```python
class NeuralPathPlanner:
    def __init__(self):
        self.collision_predictor = self.build_collision_predictor()
        self.path_generator = self.build_path_generator()

    def build_collision_predictor(self):
        """Build neural network to predict collision probability"""
        # This would be a CNN or other appropriate architecture
        # Input: sensor data, proposed path
        # Output: collision probability
        pass

    def build_path_generator(self):
        """Build neural network to generate navigation paths"""
        # Input: start, goal, environment representation
        # Output: sequence of waypoints
        pass

    def predict_collision(self, path, environment):
        """Predict collision probability for a given path"""
        # Use neural network to predict collision likelihood
        pass

    def generate_path(self, start, goal, environment):
        """Generate path using neural network"""
        # Use neural network to generate initial path
        # Then refine with traditional methods if needed
        pass
```

## Integration with ROS 2 Navigation

### Custom Navigation Plugin

```python
import rclpy
from rclpy.node import Node
from nav2_core.global_planner import GlobalPlanner
from nav2_core.local_planner import LocalPlanner
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from builtin_interfaces.msg import Duration
import numpy as np

class HumanoidNavigationPlanner(GlobalPlanner):
    def __init__(self):
        super().__init__()
        self.logger = None
        self.costmap_ros = None
        self.global_frame = None
        self.robot_base_frame = None

    def configure(self, tf_buffer, costmap_ros, global_frame, robot_base_frame, plugin_name):
        """Configure the planner with ROS 2 components"""
        self.logger = rclpy.logging.get_logger(plugin_name)
        self.costmap_ros = costmap_ros
        self.global_frame = global_frame
        self.robot_base_frame = robot_base_frame
        self.plugin_name = plugin_name

        self.logger.info(f'{self.plugin_name} plugin configured')

    def cleanup(self):
        """Clean up the planner"""
        self.logger.info(f'{self.plugin_name} plugin cleaned up')

    def set_costmap_topic(self, topic_name):
        """Set the costmap topic"""
        pass

    def create_plan(self, start, goal):
        """Create a navigation plan from start to goal"""
        self.logger.info(f'Creating plan from ({start.pose.position.x}, {start.pose.position.y}) to ({goal.pose.position.x}, {goal.pose.position.y})')

        # Convert ROS poses to numpy arrays
        start_pos = np.array([start.pose.position.x, start.pose.position.y])
        goal_pos = np.array([goal.pose.position.x, goal.pose.position.y])

        # Plan path considering humanoid constraints
        path = self.plan_humanoid_path(start_pos, goal_pos)

        if path is not None:
            # Convert path to ROS Path message
            ros_path = Path()
            ros_path.header.frame_id = self.global_frame
            ros_path.header.stamp = rclpy.time.Time().to_msg()

            for point in path:
                pose = PoseStamped()
                pose.header.frame_id = self.global_frame
                pose.header.stamp = rclpy.time.Time().to_msg()
                pose.pose.position.x = point[0]
                pose.pose.position.y = point[1]
                pose.pose.position.z = 0.0
                pose.pose.orientation.w = 1.0  # No rotation
                ros_path.poses.append(pose)

            return ros_path

        # Return empty path if planning failed
        empty_path = Path()
        empty_path.header.frame_id = self.global_frame
        empty_path.header.stamp = rclpy.time.Time().to_msg()
        return empty_path

    def plan_humanoid_path(self, start, goal):
        """Plan path considering humanoid-specific constraints"""
        # Use RRT or other appropriate algorithm
        planner = HumanoidRRT(start, goal)
        path = planner.plan_path()

        if path is not None:
            # Plan footstep sequence for the path
            robot_params = {
                'step_length': 0.3,
                'step_width': 0.2,
                'max_step_height': 0.1,
                'support_polygon': 'rectangle'
            }

            footstep_planner = FootstepPlanner(robot_params)
            footsteps = footstep_planner.plan_footsteps(path, np.concatenate([start, [0, 0]]))

            # Validate footsteps
            for footstep in footsteps:
                if not footstep_planner.validate_footstep(footstep, None):
                    self.logger.warn('Invalid footstep in path')
                    return None

        return path
```

## Performance Optimization

### Multi-Resolution Planning

```python
class MultiResolutionPlanner:
    def __init__(self):
        self.global_planner = None  # Coarse resolution
        self.local_planner = None   # Fine resolution
        self.footstep_planner = None  # Very fine resolution

    def plan_navigation(self, start, goal):
        """Plan navigation using multi-resolution approach"""
        # 1. Global planning (coarse map)
        global_path = self.global_planner.plan_path(start, goal)

        if global_path is None:
            return None

        # 2. Local refinement (fine map) around global path
        refined_path = self.refine_path_locally(global_path)

        # 3. Footstep planning (very fine resolution)
        footstep_sequence = self.plan_footsteps(refined_path)

        return {
            'global_path': global_path,
            'refined_path': refined_path,
            'footsteps': footstep_sequence
        }

    def refine_path_locally(self, global_path):
        """Refine global path using local information"""
        refined_path = []

        for i in range(len(global_path) - 1):
            segment_start = global_path[i]
            segment_end = global_path[i + 1]

            # Plan fine-grained path between waypoints
            local_path = self.local_planner.plan_path(segment_start, segment_end)
            refined_path.extend(local_path[:-1])  # Exclude last point to avoid duplication

        # Add final point
        refined_path.append(global_path[-1])

        return refined_path
```

## Safety and Reliability

### Emergency Planning

```python
class SafeNavigationPlanner:
    def __init__(self):
        self.emergency_stop_positions = []
        self.safe_zones = []
        self.backup_plans = {}

    def plan_with_safety(self, start, goal, environment):
        """Plan navigation with safety considerations"""
        # Identify safe zones and emergency stops
        safe_zones = self.identify_safe_zones(environment)
        emergency_stops = self.identify_emergency_stops(environment)

        # Plan primary path
        primary_path = self.plan_path(start, goal)

        # Generate backup plans to safe zones
        backup_paths = {}
        for safe_zone in safe_zones:
            backup_path = self.plan_path(start, safe_zone)
            if backup_path:
                backup_paths[safe_zone] = backup_path

        return {
            'primary_path': primary_path,
            'backup_paths': backup_paths,
            'safe_zones': safe_zones,
            'emergency_stops': emergency_stops
        }

    def identify_safe_zones(self, environment):
        """Identify safe zones in the environment"""
        # Safe zones are areas with low obstacle density and good visibility
        safe_zones = []
        # Implementation would analyze environment map
        return safe_zones

    def identify_emergency_stops(self, environment):
        """Identify potential emergency stop positions"""
        # Emergency stops are locations where robot can safely stop
        emergency_stops = []
        # Implementation would find open areas
        return emergency_stops
```

## Quality Assessment

### Planning Metrics

Evaluate navigation planning performance:

- **Path Optimality**: How close to optimal the path is
- **Computation Time**: Time required to generate the plan
- **Success Rate**: Percentage of successful planning attempts
- **Safety Margin**: How well the path maintains safety distances
- **Smoothness**: Continuity and smoothness of the planned path

### Validation Techniques

- **Simulation Testing**: Extensive testing in simulated environments
- **Real-world Validation**: Testing in controlled real environments
- **Benchmarking**: Comparison with standard datasets and algorithms
- **Stress Testing**: Testing in challenging scenarios

## Best Practices

### Algorithm Selection

- **Environment Type**: Choose algorithms based on environment characteristics
- **Real-time Requirements**: Balance optimality with computation time
- **Robot Constraints**: Consider specific humanoid kinematic constraints
- **Sensor Availability**: Use appropriate algorithms for available sensors

### Implementation Guidelines

- **Modular Design**: Keep planning components modular and testable
- **Parameter Tuning**: Provide adjustable parameters for different scenarios
- **Error Handling**: Implement robust error handling and recovery
- **Performance Monitoring**: Include metrics and monitoring capabilities

## Next Steps

Continue to [Mini-Project](./mini-project.md) to apply navigation planning concepts in a practical humanoid robot navigation project.