# Testing Guide

## Overview

Testing is critical for ensuring the reliability, safety, and performance of humanoid robot systems. This guide covers comprehensive testing strategies for all aspects of your integrated robot system, from unit-level components to end-to-end system validation.

## Testing Philosophy

### Testing Pyramid for Robotics

```
        System Testing (10%)
           ↓
    Integration Testing (20%)
           ↓
     Component Testing (70%)
```

In robotics, the traditional testing pyramid is adapted to account for the unique challenges of physical systems:

- **Unit Testing**: Test individual functions and algorithms
- **Component Testing**: Test individual robot subsystems
- **Integration Testing**: Test interactions between components
- **System Testing**: Test the complete robot system
- **Field Testing**: Test in real-world environments

## Unit Testing

### Testing Individual Functions

```python
import unittest
import numpy as np
from geometry_msgs.msg import Point, Pose
from tf2_ros import TransformException

class TestGeometryFunctions(unittest.TestCase):
    def test_pose_to_matrix_conversion(self):
        """Test conversion between Pose message and transformation matrix"""
        from geometry_utils import pose_to_matrix, matrix_to_pose

        # Create test pose
        test_pose = Pose()
        test_pose.position.x = 1.0
        test_pose.position.y = 2.0
        test_pose.position.z = 3.0
        test_pose.orientation.w = 1.0
        test_pose.orientation.x = 0.0
        test_pose.orientation.y = 0.0
        test_pose.orientation.z = 0.0

        # Convert to matrix and back
        matrix = pose_to_matrix(test_pose)
        recovered_pose = matrix_to_pose(matrix)

        # Verify conversion accuracy
        self.assertAlmostEqual(test_pose.position.x, recovered_pose.position.x, places=5)
        self.assertAlmostEqual(test_pose.position.y, recovered_pose.position.y, places=5)
        self.assertAlmostEqual(test_pose.position.z, recovered_pose.position.z, places=5)

    def test_distance_calculation(self):
        """Test distance calculation between points"""
        from navigation_utils import calculate_distance

        point1 = Point(x=0.0, y=0.0, z=0.0)
        point2 = Point(x=3.0, y=4.0, z=0.0)

        distance = calculate_distance(point1, point2)
        expected_distance = 5.0  # 3-4-5 triangle

        self.assertAlmostEqual(distance, expected_distance, places=5)

class TestControlAlgorithms(unittest.TestCase):
    def test_pid_controller(self):
        """Test PID controller behavior"""
        from control_algorithms import PIDController

        pid = PIDController(kp=1.0, ki=0.1, kd=0.01)

        # Test with zero error
        output = pid.compute(0.0, 0.0, 0.1)  # error=0, dt=0.1
        self.assertEqual(output, 0.0)

        # Test with positive error
        output = pid.compute(1.0, 0.0, 0.1)  # error=1.0
        self.assertGreater(output, 0.0)

        # Test with negative error
        output = pid.compute(-1.0, 0.0, 0.1)  # error=-1.0
        self.assertLess(output, 0.0)
```

### Testing ROS 2 Components

```python
import unittest
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile
from rclpy.executors import SingleThreadedExecutor
import threading

class TestROSNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_simple_publisher_subscriber(self):
        """Test basic publisher-subscriber communication"""
        class TestPublisher(Node):
            def __init__(self):
                super().__init__('test_publisher')
                self.publisher = self.create_publisher(String, 'test_topic', 10)

            def publish_message(self, msg):
                string_msg = String()
                string_msg.data = msg
                self.publisher.publish(string_msg)

        class TestSubscriber(Node):
            def __init__(self):
                super().__init__('test_subscriber')
                self.received_messages = []
                self.subscription = self.create_subscription(
                    String, 'test_topic', self.listener_callback, 10)

            def listener_callback(self, msg):
                self.received_messages.append(msg.data)

        # Create nodes
        publisher_node = TestPublisher()
        subscriber_node = TestSubscriber()

        # Create executor and add nodes
        executor = SingleThreadedExecutor()
        executor.add_node(publisher_node)
        executor.add_node(subscriber_node)

        # Start executor in a separate thread
        executor_thread = threading.Thread(target=executor.spin)
        executor_thread.start()

        # Publish and verify message
        test_message = "Hello, ROS!"
        publisher_node.publish_message(test_message)

        # Wait for message to be received
        import time
        time.sleep(0.5)

        # Stop executor
        executor.shutdown()
        executor_thread.join()

        # Verify message was received
        self.assertEqual(len(subscriber_node.received_messages), 1)
        self.assertEqual(subscriber_node.received_messages[0], test_message)
```

## Component Testing

### Testing Individual Robot Subsystems

```python
class TestNavigationComponent(unittest.TestCase):
    def setUp(self):
        self.navigation_system = NavigationSystem()
        self.test_map = self.create_test_map()

    def create_test_map(self):
        """Create a simple test map for navigation testing"""
        # Create a 10x10 grid map with some obstacles
        test_map = np.zeros((10, 10), dtype=np.uint8)
        # Add some obstacles
        test_map[5, 2:8] = 255  # Horizontal obstacle
        test_map[2:8, 5] = 255  # Vertical obstacle
        return test_map

    def test_path_planning(self):
        """Test path planning in simple environment"""
        start = (1, 1)
        goal = (8, 8)

        path = self.navigation_system.plan_path(start, goal, self.test_map)

        # Verify path exists and is valid
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 0)

        # Verify path starts and ends at correct locations
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)

        # Verify path doesn't go through obstacles
        for point in path:
            self.assertEqual(self.test_map[point], 0)  # 0 = free space

    def test_obstacle_avoidance(self):
        """Test obstacle avoidance behavior"""
        # Create a scenario where robot must avoid obstacles
        start = (1, 1)
        goal = (8, 8)

        path = self.navigation_system.plan_path(start, goal, self.test_map)

        # Path should go around obstacles, not through them
        # This is a basic check - more sophisticated validation needed
        self.assertTrue(self.path_avoids_obstacles(path, self.test_map))

    def path_avoids_obstacles(self, path, map_data):
        """Check if path avoids obstacles"""
        for point in path:
            if map_data[point] > 0:  # Obstacle
                return False
        return True

class TestManipulationComponent(unittest.TestCase):
    def setUp(self):
        self.manipulator = ManipulationSystem()
        self.test_objects = self.create_test_objects()

    def create_test_objects(self):
        """Create test objects for manipulation"""
        return [
            {'name': 'cup', 'pose': (1.0, 0.5, 0.8), 'size': (0.1, 0.1, 0.1)},
            {'name': 'box', 'pose': (1.2, 0.7, 0.8), 'size': (0.2, 0.2, 0.2)}
        ]

    def test_grasp_planning(self):
        """Test grasp planning for objects"""
        for obj in self.test_objects:
            grasp_poses = self.manipulator.plan_grasps(obj)

            # Verify grasps are generated
            self.assertGreater(len(grasp_poses), 0)

            # Verify grasp poses are valid (not None)
            for grasp in grasp_poses:
                self.assertIsNotNone(grasp)

    def test_reachability_check(self):
        """Test reachability of objects"""
        for obj in self.test_objects:
            is_reachable = self.manipulator.is_reachable(obj['pose'])

            # Basic reachability should return boolean
            self.assertIsInstance(is_reachable, bool)
```

## Integration Testing

### Testing Component Interactions

```python
class TestPerceptionNavigationIntegration(unittest.TestCase):
    def setUp(self):
        self.perception_system = PerceptionSystem()
        self.navigation_system = NavigationSystem()
        self.integration_layer = PerceptionNavigationIntegration(
            self.perception_system,
            self.navigation_system
        )

    def test_object_guided_navigation(self):
        """Test navigation to detected objects"""
        # Simulate object detection
        detected_objects = [
            {'name': 'target', 'pose': (5.0, 5.0, 0.0), 'confidence': 0.9}
        ]

        # Process detection and generate navigation goal
        navigation_goal = self.integration_layer.process_object_detection(
            detected_objects
        )

        # Verify navigation goal is generated correctly
        self.assertIsNotNone(navigation_goal)
        self.assertEqual(navigation_goal.target_pose.position.x, 5.0)
        self.assertEqual(navigation_goal.target_pose.position.y, 5.0)

    def test_dynamic_obstacle_avoidance(self):
        """Test navigation with dynamic obstacle detection"""
        # Set up initial conditions
        start_pose = (0.0, 0.0, 0.0)
        goal_pose = (10.0, 10.0, 0.0)

        # Plan initial path
        initial_path = self.navigation_system.plan_path(start_pose, goal_pose)

        # Simulate dynamic obstacle appearing
        dynamic_obstacle = {'pose': (5.0, 5.0, 0.0), 'radius': 1.0}

        # Update navigation with obstacle information
        updated_path = self.integration_layer.update_path_with_obstacle(
            initial_path, dynamic_obstacle
        )

        # Verify path was updated to avoid obstacle
        self.assertNotEqual(len(initial_path), len(updated_path))
        # Additional validation would check that path avoids obstacle

class TestVLAIntegration(unittest.TestCase):
    def setUp(self):
        self.voice_recognition = VoiceRecognitionSystem()
        self.language_understanding = LanguageUnderstandingSystem()
        self.action_execution = ActionExecutionSystem()
        self.vla_integration = VLAIntegration(
            self.voice_recognition,
            self.language_understanding,
            self.action_execution
        )

    def test_voice_command_to_action(self):
        """Test complete VLA pipeline"""
        # Simulate voice command
        voice_input = "Go to the kitchen and pick up the red cup"

        # Process through VLA pipeline
        result = self.vla_integration.process_command(voice_input)

        # Verify pipeline processed correctly
        self.assertIsNotNone(result)
        self.assertTrue(result['success'])
        self.assertIn('navigation', result['actions'])
        self.assertIn('manipulation', result['actions'])
```

## System Testing

### End-to-End System Tests

```python
class TestCompleteRobotSystem(unittest.TestCase):
    def setUp(self):
        # Set up complete robot system
        self.robot_system = CompleteRobotSystem()
        self.test_scenarios = self.load_test_scenarios()

    def load_test_scenarios(self):
        """Load various test scenarios"""
        return [
            {
                'name': 'simple_navigation',
                'commands': ['go to kitchen'],
                'expected_outcomes': ['navigation_success']
            },
            {
                'name': 'object_interaction',
                'commands': ['find red cup', 'pick up red cup', 'place cup on table'],
                'expected_outcomes': ['detection_success', 'grasp_success', 'placement_success']
            },
            {
                'name': 'complex_task',
                'commands': ['go to kitchen', 'find ingredients', 'bring to counter'],
                'expected_outcomes': ['multiple_successes']
            }
        ]

    def test_scenario_execution(self):
        """Test execution of various scenarios"""
        for scenario in self.test_scenarios:
            with self.subTest(scenario=scenario['name']):
                # Execute scenario
                results = self.execute_scenario(scenario)

                # Verify expected outcomes
                for expected_outcome in scenario['expected_outcomes']:
                    self.assertIn(expected_outcome, results)

    def execute_scenario(self, scenario):
        """Execute a test scenario and return results"""
        results = []

        for command in scenario['commands']:
            try:
                result = self.robot_system.execute_command(command)
                results.append(result['status'])
            except Exception as e:
                results.append(f'error: {str(e)}')

        return results

    def test_system_stress(self):
        """Test system under stress conditions"""
        # Test with rapid command succession
        commands = ['go to kitchen'] * 10

        start_time = time.time()
        for command in commands:
            self.robot_system.execute_command(command)
        end_time = time.time()

        # Verify system can handle rapid commands within time constraints
        execution_time = end_time - start_time
        self.assertLess(execution_time, 30.0)  # Should complete in under 30 seconds

    def test_failure_recovery(self):
        """Test system recovery from failures"""
        # Simulate a component failure
        self.robot_system.navigation_system.force_failure()

        # Try to execute navigation command
        result = self.robot_system.execute_command('go to kitchen')

        # Verify system handles failure gracefully
        self.assertIn('recovery', result['actions'])
        self.assertEqual(result['status'], 'recovered')
```

## Simulation Testing

### Gazebo Integration Tests

```python
import unittest
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from std_srvs.srv import Empty
import time

class TestSimulationIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = Node('simulation_tester')

        # Create publishers for robot control
        self.cmd_vel_pub = self.node.create_publisher(
            Twist, '/cmd_vel', 10
        )

        # Create subscribers for sensor data
        self.laser_sub = self.node.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10
        )
        self.camera_sub = self.node.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, 10
        )

        # Store sensor data
        self.latest_laser = None
        self.latest_image = None

    def laser_callback(self, msg):
        self.latest_laser = msg

    def camera_callback(self, msg):
        self.latest_image = msg

    def test_sensor_data_availability(self):
        """Test that sensor data is available in simulation"""
        # Wait for sensor data
        timeout = 5.0  # seconds
        start_time = time.time()

        while (self.latest_laser is None or self.latest_image is None) and \
              (time.time() - start_time < timeout):
            rclpy.spin_once(self.node, timeout_sec=0.1)

        # Verify data was received
        self.assertIsNotNone(self.latest_laser, "Laser data not received")
        self.assertIsNotNone(self.latest_image, "Camera data not received")

    def test_robot_movement(self):
        """Test robot movement in simulation"""
        # Send movement command
        cmd = Twist()
        cmd.linear.x = 0.5  # Move forward at 0.5 m/s
        cmd.angular.z = 0.0  # No rotation

        # Publish command
        self.cmd_vel_pub.publish(cmd)

        # Wait and verify movement affects sensor readings
        time.sleep(2.0)  # Allow time for movement

        # In a real test, you would verify that position changed
        # or that laser readings changed appropriately
        self.assertIsNotNone(self.latest_laser)

    def test_obstacle_detection(self):
        """Test obstacle detection in simulation"""
        # Wait for sensor data
        timeout = 5.0
        start_time = time.time()

        while self.latest_laser is None and (time.time() - start_time < timeout):
            rclpy.spin_once(self.node, timeout_sec=0.1)

        if self.latest_laser:
            # Check that there are valid range readings
            valid_ranges = [r for r in self.latest_laser.ranges if r > 0 and r < float('inf')]
            self.assertGreater(len(valid_ranges), 0, "No valid laser readings")
```

## Performance Testing

### Load and Stress Testing

```python
import time
import threading
import psutil
from concurrent.futures import ThreadPoolExecutor
import statistics

class PerformanceTester:
    def __init__(self, robot_system):
        self.robot_system = robot_system
        self.metrics = {
            'response_times': [],
            'throughput': [],
            'cpu_usage': [],
            'memory_usage': [],
            'success_rates': []
        }

    def test_concurrent_commands(self, num_commands=10):
        """Test system performance with concurrent commands"""
        start_time = time.time()

        def send_command(i):
            command = f"command_{i}"
            start_cmd = time.time()
            try:
                result = self.robot_system.execute_command(command)
                response_time = time.time() - start_cmd
                return {'success': True, 'response_time': response_time, 'result': result}
            except Exception as e:
                response_time = time.time() - start_cmd
                return {'success': False, 'response_time': response_time, 'error': str(e)}

        # Execute commands concurrently
        with ThreadPoolExecutor(max_workers=num_commands) as executor:
            futures = [executor.submit(send_command, i) for i in range(num_commands)]
            results = [future.result() for future in futures]

        end_time = time.time()
        total_time = end_time - start_time

        # Calculate metrics
        successful_commands = [r for r in results if r['success']]
        response_times = [r['response_time'] for r in results]

        metrics = {
            'total_commands': num_commands,
            'successful_commands': len(successful_commands),
            'success_rate': len(successful_commands) / num_commands,
            'total_time': total_time,
            'throughput': num_commands / total_time,
            'avg_response_time': statistics.mean(response_times) if response_times else 0,
            'max_response_time': max(response_times) if response_times else 0,
            'min_response_time': min(response_times) if response_times else 0
        }

        return metrics

    def test_long_running_stability(self, duration_minutes=10):
        """Test system stability over extended period"""
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)  # Convert to seconds

        command_counter = 0
        successful_commands = 0

        while time.time() < end_time:
            try:
                # Send periodic commands to keep system active
                result = self.robot_system.execute_command(f"status_check_{command_counter}")
                if result.get('success', False):
                    successful_commands += 1
            except:
                pass  # Command failed, continue testing

            command_counter += 1

            # Record system metrics periodically
            if command_counter % 10 == 0:
                self.record_system_metrics()

            time.sleep(1)  # Send command every second

        # Calculate stability metrics
        total_commands = command_counter
        success_rate = successful_commands / total_commands if total_commands > 0 else 0

        return {
            'total_duration': duration_minutes,
            'total_commands': total_commands,
            'successful_commands': successful_commands,
            'success_rate': success_rate,
            'avg_commands_per_minute': total_commands / duration_minutes
        }

    def record_system_metrics(self):
        """Record system performance metrics"""
        self.metrics['cpu_usage'].append(psutil.cpu_percent())
        self.metrics['memory_usage'].append(psutil.virtual_memory().percent)

    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        report = {
            'system_performance': {
                'avg_cpu_usage': statistics.mean(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0,
                'avg_memory_usage': statistics.mean(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0,
                'max_cpu_usage': max(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0,
                'max_memory_usage': max(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0
            }
        }

        return report
```

## Safety Testing

### Safety Validation Tests

```python
class SafetyTester:
    def __init__(self, robot_system):
        self.robot_system = robot_system
        self.safety_limits = {
            'max_velocity': 1.0,  # m/s
            'max_angular_velocity': 1.0,  # rad/s
            'max_force': 50.0,  # N
            'min_distance_to_human': 0.5,  # m
            'max_operation_time': 3600  # seconds
        }

    def test_velocity_limits(self):
        """Test that robot respects velocity limits"""
        # Command robot to move at high speed
        cmd = Twist()
        cmd.linear.x = 5.0  # Much higher than limit
        cmd.angular.z = 5.0  # Much higher than limit

        # Verify robot doesn't exceed limits
        actual_velocity = self.robot_system.execute_command_with_monitoring(cmd)

        self.assertLessEqual(actual_velocity.linear.x, self.safety_limits['max_velocity'])
        self.assertLessEqual(actual_velocity.angular.z, self.safety_limits['max_angular_velocity'])

    def test_collision_prevention(self):
        """Test collision prevention system"""
        # Simulate obstacles in path
        obstacles = [
            {'position': (1.0, 0.0, 0.0), 'radius': 0.5},
            {'position': (2.0, 0.0, 0.0), 'radius': 0.3}
        ]

        # Attempt navigation with obstacles
        result = self.robot_system.navigate_with_obstacles(obstacles)

        # Verify robot stopped before collision
        self.assertTrue(result['stopped_safely'])
        self.assertFalse(result['collision_occurred'])

    def test_emergency_stop(self):
        """Test emergency stop functionality"""
        # Start robot in motion
        self.robot_system.start_motion()

        # Trigger emergency stop
        self.robot_system.trigger_emergency_stop()

        # Verify robot stops within safety limits
        time.sleep(0.5)  # Allow time for stop
        current_state = self.robot_system.get_state()

        self.assertEqual(current_state['velocity'], 0.0)
        self.assertEqual(current_state['status'], 'emergency_stopped')

    def test_human_detection_safety(self):
        """Test safety when humans are detected"""
        # Simulate human detection
        human_positions = [(2.0, 1.0, 0.0), (2.5, 1.2, 0.0)]

        # Attempt navigation near humans
        for pos in human_positions:
            safety_check = self.robot_system.check_human_safety(pos)
            self.assertGreater(safety_check['distance'], self.safety_limits['min_distance_to_human'])

    def test_operation_time_limits(self):
        """Test automatic shutdown after time limits"""
        start_time = time.time()

        # Run robot for extended period
        while time.time() - start_time < self.safety_limits['max_operation_time'] + 10:
            self.robot_system.execute_command("continue_operation")
            time.sleep(1)

        # Verify robot automatically stops
        final_state = self.robot_system.get_state()
        self.assertEqual(final_state['status'], 'auto_shutdown')
```

## Regression Testing

### Automated Test Suite

```python
import unittest
import xmlrunner
import subprocess
import os

class RegressionTestSuite(unittest.TestSuite):
    def __init__(self):
        super().__init__()

        # Add all test cases
        self.addTest(unittest.makeSuite(TestGeometryFunctions))
        self.addTest(unittest.makeSuite(TestControlAlgorithms))
        self.addTest(unittest.makeSuite(TestNavigationComponent))
        self.addTest(unittest.makeSuite(TestManipulationComponent))
        self.addTest(unittest.makeSuite(TestPerceptionNavigationIntegration))
        self.addTest(unittest.makeSuite(TestVLAIntegration))
        self.addTest(unittest.makeSuite(TestCompleteRobotSystem))

def run_regression_tests(output_dir='test_results'):
    """Run complete regression test suite"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create test suite
    suite = RegressionTestSuite()

    # Run tests with XML output for CI/CD
    with open(f'{output_dir}/test_results.xml', 'wb') as output:
        runner = xmlrunner.XMLTestRunner(
            output=output,
            outsuffix='',
            verbosity=2
        )
        result = runner.run(suite)

    # Generate coverage report if available
    try:
        subprocess.run(['coverage', 'run', '-m', 'unittest', 'discover'], check=True)
        subprocess.run(['coverage', 'report'], check=True)
        subprocess.run(['coverage', 'html'], check=True)
    except subprocess.CalledProcessError:
        print("Coverage tools not available, skipping coverage report")

    return result

def test_continuous_integration():
    """Function to run in CI/CD pipeline"""
    print("Starting CI/CD test pipeline...")

    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)

    # Run integration tests
    result = run_regression_tests()

    # Check if tests passed
    if result.wasSuccessful():
        print("All tests passed! ✅")
        return True
    else:
        print("Some tests failed! ❌")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        return False
```

## Testing Best Practices

### Test Organization and Management

```python
# test_config.py
TEST_CONFIG = {
    'default_timeout': 30.0,
    'simulation_timeout_multiplier': 2.0,
    'retries_on_failure': 3,
    'parallel_test_workers': 4,
    'test_data_dir': 'test_data',
    'results_dir': 'test_results',
    'coverage_threshold': 80.0,  # Percent
    'performance_thresholds': {
        'response_time_ms': 100,
        'throughput_hz': 10,
        'cpu_usage_percent': 80,
        'memory_usage_mb': 1024
    }
}

# test_decorators.py
import functools
import time
import unittest

def retry_on_failure(max_attempts=3, delay=1.0):
    """Decorator to retry test on failure"""
    def decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return test_func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
                    else:
                        raise e

        return wrapper
    return decorator

def performance_test(expected_duration):
    """Decorator to test performance requirements"""
    def decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = test_func(*args, **kwargs)
            end_time = time.time()

            actual_duration = end_time - start_time

            if actual_duration > expected_duration:
                raise AssertionError(
                    f"Test took {actual_duration:.2f}s, "
                    f"expected < {expected_duration}s"
                )

            return result
        return wrapper
    return decorator

# Example usage
class TestWithDecorators(unittest.TestCase):
    @retry_on_failure(max_attempts=3, delay=0.5)
    def test_unreliable_connection(self):
        """Test that might fail due to network issues"""
        # Implementation that might be flaky
        pass

    @performance_test(expected_duration=0.1)  # 100ms
    def test_fast_algorithm(self):
        """Test that must complete quickly"""
        # Implementation that should be fast
        pass
```

## Continuous Testing

### Automated Testing Pipeline

```python
class ContinuousTestingPipeline:
    def __init__(self):
        self.test_suites = {
            'unit': unittest.TestSuite(),
            'integration': unittest.TestSuite(),
            'system': unittest.TestSuite(),
            'performance': unittest.TestSuite()
        }
        self.results_history = []

    def add_test_suite(self, level, suite):
        """Add test suite for specific level"""
        if level in self.test_suites:
            self.test_suites[level].addTest(suite)

    def run_all_tests(self):
        """Run all test suites in order"""
        results = {}

        for level, suite in self.test_suites.items():
            if suite.countTestCases() > 0:
                print(f"Running {level} tests...")
                result = unittest.TextTestRunner(verbosity=1).run(suite)
                results[level] = result

                # Store result for history
                self.results_history.append({
                    'timestamp': time.time(),
                    'level': level,
                    'result': result
                })

        return results

    def check_regression(self):
        """Check for test regressions"""
        if len(self.results_history) < 2:
            return False  # Need at least 2 runs to compare

        # Compare latest run with previous run
        latest = self.results_history[-1]
        previous = self.results_history[-2]

        # Check if success rate decreased
        latest_success_rate = self.calculate_success_rate(latest['result'])
        previous_success_rate = self.calculate_success_rate(previous['result'])

        return latest_success_rate < previous_success_rate

    def calculate_success_rate(self, result):
        """Calculate test success rate"""
        total = result.testsRun
        if total == 0:
            return 1.0  # If no tests, consider 100% success

        failed = len(result.failures)
        errors = len(result.errors)
        unsuccessful = failed + errors

        return (total - unsuccessful) / total

    def generate_test_report(self):
        """Generate comprehensive test report"""
        report = {
            'timestamp': time.time(),
            'total_tests_run': 0,
            'total_successes': 0,
            'total_failures': 0,
            'total_errors': 0,
            'success_rate': 0.0,
            'regression_detected': self.check_regression(),
            'suite_results': {}
        }

        for level, result in self.run_all_tests().items():
            report['suite_results'][level] = {
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'success_rate': self.calculate_success_rate(result)
            }
            report['total_tests_run'] += result.testsRun
            report['total_failures'] += len(result.failures)
            report['total_errors'] += len(result.errors)

        successful_tests = report['total_tests_run'] - report['total_failures'] - report['total_errors']
        report['total_successes'] = successful_tests
        report['success_rate'] = successful_tests / report['total_tests_run'] if report['total_tests_run'] > 0 else 0

        return report
```

## Next Steps

Continue to [Troubleshooting](./troubleshooting.md) to learn about identifying and resolving issues in your integrated humanoid robot system.