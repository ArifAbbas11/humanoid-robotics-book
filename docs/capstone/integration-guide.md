# Integration Guide

## Overview

System integration is one of the most challenging aspects of humanoid robotics development. This guide provides best practices, strategies, and techniques for successfully integrating the various components of your humanoid robot system, from perception to action.

## Integration Architecture

### Layered Integration Approach

The recommended approach for integrating humanoid robot systems follows a layered architecture:

```
┌─────────────────────────────────────┐
│            User Interface           │
├─────────────────────────────────────┤
│         Task Planning & AI          │
├─────────────────────────────────────┤
│        Motion Planning & Control    │
├─────────────────────────────────────┤
│          Perception & Sensing       │
├─────────────────────────────────────┤
│           Hardware Interface        │
└─────────────────────────────────────┘
```

### Communication Patterns

#### ROS 2 Communication Architecture

Use appropriate ROS 2 communication patterns for different integration needs:

```python
# Publisher-Subscriber Pattern for sensor data
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, JointState
from geometry_msgs.msg import Twist

class SensorIntegrationNode(Node):
    def __init__(self):
        super().__init__('sensor_integration_node')

        # Publishers for processed sensor data
        self.vision_pub = self.create_publisher(Image, 'processed_vision', 10)
        self.imu_pub = self.create_publisher(Imu, 'filtered_imu', 10)
        self.joint_pub = self.create_publisher(JointState, 'filtered_joints', 10)

        # Subscribers for raw sensor data
        self.raw_vision_sub = self.create_subscription(
            Image, 'camera/image_raw', self.vision_callback, 10)
        self.raw_imu_sub = self.create_subscription(
            Imu, 'imu/data_raw', self.imu_callback, 10)
        self.raw_joint_sub = self.create_subscription(
            JointState, 'joint_states_raw', self.joint_callback, 10)

    def vision_callback(self, msg):
        """Process raw vision data"""
        # Apply filtering, calibration, etc.
        processed_msg = self.process_vision(msg)
        self.vision_pub.publish(processed_msg)

    def process_vision(self, raw_msg):
        """Apply processing to raw vision data"""
        # Implementation would include:
        # - Camera calibration
        # - Noise reduction
        # - Feature extraction
        # - Object detection
        return raw_msg  # Placeholder
```

#### Service-Based Integration

For request-response interactions:

```python
# Service-based integration for planning
from vla_msgs.srv import PlanPath, ExecuteAction

class PlanningIntegrationNode(Node):
    def __init__(self):
        super().__init__('planning_integration_node')

        # Services for planning requests
        self.plan_path_service = self.create_service(
            PlanPath, 'plan_path', self.plan_path_callback)
        self.execute_action_service = self.create_service(
            ExecuteAction, 'execute_action', self.execute_action_callback)

    def plan_path_callback(self, request, response):
        """Handle path planning requests"""
        try:
            # Plan path using integrated system
            path = self.integrated_planner.plan_path(
                request.start, request.goal, request.constraints)

            response.success = True
            response.path = path
            response.message = "Path planned successfully"

        except Exception as e:
            response.success = False
            response.message = f"Planning failed: {str(e)}"

        return response
```

#### Action-Based Integration

For long-running operations with feedback:

```python
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from vla_msgs.action import NavigateToPose

class NavigationIntegrationServer:
    def __init__(self, node):
        self.node = node
        self.action_server = ActionServer(
            node,
            NavigateToPose,
            'navigate_to_pose',
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

    def goal_callback(self, goal_request):
        """Accept or reject navigation goals"""
        # Validate goal before accepting
        if self.is_valid_goal(goal_request.pose):
            return GoalResponse.ACCEPT
        else:
            return GoalResponse.REJECT

    async def execute_callback(self, goal_handle):
        """Execute navigation goal with feedback"""
        feedback_msg = NavigateToPose.Feedback()

        # Execute navigation with continuous feedback
        for step in range(100):  # Simplified navigation loop
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return NavigateToPose.Result()

            # Update feedback
            feedback_msg.current_pose = self.get_current_pose()
            feedback_msg.distance_remaining = self.calculate_distance_remaining()
            goal_handle.publish_feedback(feedback_msg)

            # Sleep to allow other callbacks to run
            await asyncio.sleep(0.1)

        goal_handle.succeed()
        result = NavigateToPose.Result()
        result.success = True
        return result
```

## Component Integration Strategies

### Modular Integration Pattern

Build integration using modular, reusable components:

```python
class ComponentManager:
    def __init__(self):
        self.components = {}
        self.connections = []

    def register_component(self, name, component):
        """Register a component for integration"""
        self.components[name] = component

    def connect_components(self, source, target, connection_type='data'):
        """Establish connection between components"""
        connection = {
            'source': source,
            'target': target,
            'type': connection_type,
            'active': False
        }
        self.connections.append(connection)

    def initialize_connections(self):
        """Initialize all registered connections"""
        for conn in self.connections:
            if conn['source'] in self.components and conn['target'] in self.components:
                self.establish_connection(conn)
                conn['active'] = True

    def establish_connection(self, connection):
        """Establish specific connection between components"""
        source_comp = self.components[connection['source']]
        target_comp = self.components[connection['target']]

        # Example: Connect publisher to subscriber
        if connection['type'] == 'data':
            # Implementation depends on component types
            pass
```

### Configuration-Driven Integration

Use configuration files to manage integration parameters:

```yaml
# integration_config.yaml
integration:
  components:
    perception:
      enabled: true
      frequency: 30.0
      timeout: 5.0
    planning:
      enabled: true
      frequency: 10.0
      timeout: 10.0
    control:
      enabled: true
      frequency: 100.0
      timeout: 1.0

  connections:
    - source: "camera/image_raw"
      target: "vision_processor/image_in"
      type: "sensor_msgs/Image"
    - source: "vision_processor/detections"
      target: "planning/object_detections"
      type: "vision_msgs/Detection2DArray"

  safety_limits:
    max_velocity: 1.0
    max_acceleration: 2.0
    collision_threshold: 0.5
```

## Real-Time Integration Considerations

### Timing and Synchronization

```python
import threading
import time
from collections import deque

class RealTimeIntegrator:
    def __init__(self):
        self.sensors = {}
        self.processors = {}
        self.sync_window = 0.05  # 50ms sync window
        self.main_loop_rate = 50.0  # 50Hz main loop

    def add_sensor(self, name, topic, callback):
        """Add sensor with synchronization requirements"""
        self.sensors[name] = {
            'topic': topic,
            'callback': callback,
            'buffer': deque(maxlen=10),
            'last_update': 0
        }

    def add_processor(self, name, inputs, output_callback):
        """Add processor that requires synchronized inputs"""
        self.processors[name] = {
            'inputs': inputs,  # List of required input names
            'output_callback': output_callback,
            'last_execution': 0
        }

    def sensor_callback(self, sensor_name, data):
        """Handle incoming sensor data"""
        sensor = self.sensors[sensor_name]
        sensor['buffer'].append((time.time(), data))
        sensor['last_update'] = time.time()

        # Check if we can execute any processors
        self.check_processor_execution()

    def check_processor_execution(self):
        """Check if any processors have all required inputs"""
        current_time = time.time()

        for proc_name, processor in self.processors.items():
            # Check if all required inputs are available within sync window
            all_inputs_available = True
            synced_data = {}

            for input_name in processor['inputs']:
                if input_name not in self.sensors:
                    all_inputs_available = False
                    break

                sensor = self.sensors[input_name]
                latest_data = None

                # Find most recent data within sync window
                for timestamp, data in reversed(sensor['buffer']):
                    if current_time - timestamp <= self.sync_window:
                        latest_data = data
                        break

                if latest_data is None:
                    all_inputs_available = False
                    break

                synced_data[input_name] = latest_data

            # Execute processor if all inputs are available
            if all_inputs_available and \
               current_time - processor['last_execution'] >= 1.0/self.main_loop_rate:
                try:
                    result = processor['output_callback'](synced_data)
                    processor['last_execution'] = current_time
                except Exception as e:
                    print(f"Processor {proc_name} execution failed: {e}")
```

### Resource Management

```python
class ResourceManager:
    def __init__(self):
        self.resources = {
            'cpu': {'total': 100, 'used': 0, 'reserved': {}},
            'gpu': {'total': 100, 'used': 0, 'reserved': {}},
            'memory': {'total': 8192, 'used': 0, 'reserved': {}},  # MB
            'bandwidth': {'total': 1000, 'used': 0, 'reserved': {}}  # Mbps
        }

    def reserve_resources(self, component_id, requirements):
        """Reserve resources for a component"""
        for resource_type, amount in requirements.items():
            if resource_type in self.resources:
                available = (self.resources[resource_type]['total'] -
                           self.resources[resource_type]['used'])

                if available >= amount:
                    self.resources[resource_type]['reserved'][component_id] = amount
                    self.resources[resource_type]['used'] += amount
                else:
                    raise ResourceNotAvailableError(
                        f"Not enough {resource_type} available")

    def release_resources(self, component_id):
        """Release resources when component is done"""
        for resource_type in self.resources:
            if component_id in self.resources[resource_type]['reserved']:
                amount = self.resources[resource_type]['reserved'][component_id]
                self.resources[resource_type]['used'] -= amount
                del self.resources[resource_type]['reserved'][component_id]

class ResourceNotAvailableError(Exception):
    pass
```

## Error Handling and Fault Tolerance

### Graceful Degradation

```python
class FaultTolerantIntegrator:
    def __init__(self):
        self.components = {}
        self.fallback_strategies = {}
        self.health_monitor = HealthMonitor()

    def register_component(self, name, component, fallback_strategy=None):
        """Register component with optional fallback"""
        self.components[name] = {
            'instance': component,
            'healthy': True,
            'fallback': fallback_strategy
        }

        if fallback_strategy:
            self.fallback_strategies[name] = fallback_strategy

    def execute_with_fallback(self, component_name, method_name, *args, **kwargs):
        """Execute component method with fallback handling"""
        component_info = self.components[component_name]

        if not component_info['healthy']:
            # Use fallback strategy
            fallback = self.fallback_strategies.get(component_name)
            if fallback:
                return fallback(*args, **kwargs)
            else:
                raise ComponentUnhealthyError(f"Component {component_name} is unhealthy and no fallback available")

        try:
            component = component_info['instance']
            method = getattr(component, method_name)
            result = method(*args, **kwargs)

            # Update health status
            self.health_monitor.update_component_health(component_name, True)
            return result

        except Exception as e:
            # Mark component as unhealthy
            component_info['healthy'] = False
            self.health_monitor.update_component_health(component_name, False)

            # Try fallback
            fallback = self.fallback_strategies.get(component_name)
            if fallback:
                return fallback(*args, **kwargs)
            else:
                raise e

class HealthMonitor:
    def __init__(self):
        self.component_health = {}
        self.health_history = {}

    def update_component_health(self, component_name, healthy):
        """Update health status of a component"""
        self.component_health[component_name] = {
            'healthy': healthy,
            'timestamp': time.time(),
            'consecutive_failures': 0
        }

        if component_name not in self.health_history:
            self.health_history[component_name] = []

        self.health_history[component_name].append({
            'timestamp': time.time(),
            'healthy': healthy
        })

        # Keep only recent history
        if len(self.health_history[component_name]) > 100:
            self.health_history[component_name] = \
                self.health_history[component_name][-100:]

class ComponentUnhealthyError(Exception):
    pass
```

## Data Integration Patterns

### Sensor Fusion Integration

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class SensorFusionIntegrator:
    def __init__(self):
        self.sensors = {}
        self.fusion_engine = KalmanFusionEngine()
        self.calibration_data = {}

    def add_sensor(self, name, sensor_type, topic, transform):
        """Add sensor to fusion system"""
        self.sensors[name] = {
            'type': sensor_type,
            'topic': topic,
            'transform': transform,  # TF transform from sensor to base frame
            'reliability': 0.9,  # Initial reliability estimate
            'bias': np.zeros(3),  # Sensor bias correction
            'last_update': 0
        }

    def integrate_sensor_data(self, sensor_name, raw_data, timestamp):
        """Integrate sensor data into fused estimate"""
        if sensor_name not in self.sensors:
            return None

        sensor_info = self.sensors[sensor_name]

        # Apply calibration and bias correction
        calibrated_data = self.calibrate_sensor_data(
            raw_data, sensor_info['bias'], sensor_info['transform'])

        # Update fusion engine
        fused_estimate = self.fusion_engine.update(
            sensor_name, calibrated_data, timestamp)

        return fused_estimate

    def calibrate_sensor_data(self, raw_data, bias, transform):
        """Apply calibration to raw sensor data"""
        # Apply bias correction
        corrected_data = raw_data - bias

        # Apply coordinate frame transformation
        if transform is not None:
            corrected_data = self.apply_transform(corrected_data, transform)

        return corrected_data

    def apply_transform(self, data, transform):
        """Apply coordinate frame transformation"""
        # Implementation depends on data type
        # For position data: apply translation and rotation
        # For orientation data: apply rotation only
        return data  # Placeholder

class KalmanFusionEngine:
    def __init__(self):
        # Initialize Kalman filter parameters
        self.state_vector = np.zeros(13)  # [position, orientation, velocity, angular_velocity]
        self.covariance_matrix = np.eye(13) * 1000  # High initial uncertainty

    def update(self, sensor_name, sensor_data, timestamp):
        """Update state estimate with new sensor measurement"""
        # Prediction step (if time has passed)
        # Update step with sensor measurement
        # Return updated state estimate
        pass
```

### Multi-Modal Data Integration

```python
class MultiModalIntegrator:
    def __init__(self):
        self.modalities = {
            'vision': VisionModalityProcessor(),
            'audio': AudioModalityProcessor(),
            'tactile': TactileModalityProcessor(),
            'proprioceptive': ProprioceptiveModalityProcessor()
        }
        self.cross_modal_fusion = CrossModalFusionEngine()

    def process_multimodal_input(self, inputs):
        """Process inputs from multiple modalities"""
        processed_outputs = {}

        # Process each modality independently
        for modality_name, data in inputs.items():
            if modality_name in self.modalities:
                processed_outputs[modality_name] = \
                    self.modalities[modality_name].process(data)

        # Perform cross-modal fusion
        fused_output = self.cross_modal_fusion.fuse(processed_outputs)

        return fused_output

class CrossModalFusionEngine:
    def __init__(self):
        self.attention_mechanisms = {}
        self.confidence_estimators = {}

    def fuse(self, modality_outputs):
        """Fuse outputs from multiple modalities"""
        # Estimate confidence for each modality
        confidences = {}
        for modality, output in modality_outputs.items():
            confidences[modality] = self.estimate_confidence(modality, output)

        # Apply attention-weighted fusion
        fused_result = self.attention_fusion(modality_outputs, confidences)

        return fused_result

    def estimate_confidence(self, modality, output):
        """Estimate confidence in modality output"""
        # Implementation depends on modality type
        # Consider factors like signal quality, noise level, consistency
        return 0.8  # Placeholder

    def attention_fusion(self, outputs, confidences):
        """Apply attention-weighted fusion"""
        # Weight each modality by its confidence
        weighted_sum = {}
        total_weight = sum(confidences.values())

        for modality, output in outputs.items():
            weight = confidences[modality] / total_weight
            # Apply weighting to output (implementation depends on data type)
            pass

        return {}  # Placeholder
```

## Testing Integration

### Integration Testing Framework

```python
import unittest
from unittest.mock import Mock, patch

class IntegrationTestSuite(unittest.TestCase):
    def setUp(self):
        """Set up integration test environment"""
        self.integration_manager = ComponentManager()
        self.test_data = self.generate_test_data()

    def test_sensor_to_planner_integration(self):
        """Test integration between perception and planning"""
        # Mock sensor data
        mock_vision_data = self.create_mock_vision_data()

        # Simulate data flow through system
        processed_objects = self.process_vision_data(mock_vision_data)
        planning_request = self.create_planning_request(processed_objects)
        plan = self.generate_plan(planning_request)

        # Verify integration worked correctly
        self.assertIsNotNone(plan)
        self.assertGreater(len(plan), 0)
        self.assertTrue(self.validate_plan(plan))

    def test_multi_component_integration(self):
        """Test integration of multiple components"""
        # Set up component connections
        self.integration_manager.connect_components('vision', 'planning', 'data')
        self.integration_manager.connect_components('planning', 'control', 'commands')
        self.integration_manager.initialize_connections()

        # Simulate end-to-end flow
        input_data = self.test_data['sample_input']
        output = self.execute_end_to_end(input_data)

        # Verify complete integration
        self.assertIsNotNone(output)
        self.assertTrue(self.validate_output(output))

    def generate_test_data(self):
        """Generate test data for integration tests"""
        return {
            'sample_input': {
                'objects': [{'type': 'cup', 'position': [1.0, 2.0, 0.5]}],
                'goal': {'position': [3.0, 4.0, 0.0]}
            },
            'expected_output': {
                'actions': ['navigate', 'grasp', 'place']
            }
        }

def run_integration_tests():
    """Run the complete integration test suite"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(IntegrationTestSuite)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result
```

## Performance Optimization

### Integration Performance Monitoring

```python
import time
import psutil
from collections import defaultdict, deque

class IntegrationPerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'latency': defaultdict(deque),
            'throughput': defaultdict(deque),
            'cpu_usage': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'bandwidth_usage': defaultdict(deque)
        }
        self.start_times = {}

    def start_operation(self, operation_name):
        """Start timing an operation"""
        self.start_times[operation_name] = time.time()

    def end_operation(self, operation_name):
        """End timing an operation and record metrics"""
        if operation_name in self.start_times:
            duration = time.time() - self.start_times[operation_name]
            self.metrics['latency'][operation_name].append(duration)
            del self.start_times[operation_name]

    def record_throughput(self, component_name, count):
        """Record throughput for a component"""
        self.metrics['throughput'][component_name].append(count)

    def update_system_metrics(self):
        """Update system-level metrics"""
        self.metrics['cpu_usage'].append(psutil.cpu_percent())
        self.metrics['memory_usage'].append(psutil.virtual_memory().percent)

    def get_performance_report(self):
        """Generate performance report"""
        report = {}

        for component, latencies in self.metrics['latency'].items():
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                report[f'{component}_avg_latency'] = avg_latency
                report[f'{component}_max_latency'] = max(latencies)

        for component, throughput in self.metrics['throughput'].items():
            if throughput:
                avg_throughput = sum(throughput) / len(throughput)
                report[f'{component}_avg_throughput'] = avg_throughput

        if self.metrics['cpu_usage']:
            report['avg_cpu_usage'] = sum(self.metrics['cpu_usage']) / len(self.metrics['cpu_usage'])

        if self.metrics['memory_usage']:
            report['avg_memory_usage'] = sum(self.metrics['memory_usage']) / len(self.metrics['memory_usage'])

        return report
```

## Best Practices

### Integration Best Practices

1. **Start Simple**: Begin with basic component connections before adding complexity
2. **Test Incrementally**: Validate each integration step before proceeding
3. **Use Standard Interfaces**: Maintain consistent message types and APIs
4. **Implement Fallbacks**: Always have backup strategies for component failures
5. **Monitor Performance**: Continuously track integration performance metrics
6. **Document Dependencies**: Clearly document component interdependencies
7. **Plan for Scalability**: Design integration to handle increased complexity

### Troubleshooting Integration Issues

**Common Integration Problems**:
- Timing mismatches between components
- Data format incompatibilities
- Resource contention
- Communication failures
- State synchronization issues

**Diagnostic Approaches**:
- Use ROS 2 tools (rqt_graph, ros2 topic echo, etc.)
- Implement comprehensive logging
- Create integration test suites
- Monitor system performance metrics
- Use debugging tools and profilers

## Next Steps

Continue to [Testing Guide](./testing-guide.md) to learn about comprehensive testing strategies for your integrated humanoid robot system.