# Cognitive Planning

## Overview

Cognitive planning is the high-level reasoning component of Vision-Language-Action (VLA) systems that bridges natural language understanding with executable robot actions. It involves breaking down complex tasks into manageable subtasks, reasoning about the environment, and generating executable plans that achieve user goals while respecting robot capabilities and safety constraints.

## Cognitive Planning Fundamentals

### What is Cognitive Planning?

Cognitive planning in robotics involves:

- **Task Decomposition**: Breaking complex goals into simpler, executable subtasks
- **Reasoning**: Using knowledge about the world and robot capabilities to make decisions
- **Planning**: Generating sequences of actions to achieve goals
- **Adaptation**: Adjusting plans based on environmental changes and feedback

### Key Components

- **Knowledge Representation**: How the robot represents its world knowledge
- **Task Planning**: High-level planning of task sequences
- **Action Selection**: Choosing appropriate actions based on context
- **Plan Execution**: Monitoring and executing the planned sequence
- **Replanning**: Adjusting plans when unexpected situations arise

## Planning Architecture

### Hierarchical Task Network (HTN) Planning

HTN planning decomposes high-level tasks into primitive actions:

```python
class HTNPlanner:
    def __init__(self):
        self.task_networks = {}
        self.primitive_actions = {}
        self.knowledge_base = {}

    def plan_task(self, task, state):
        """Plan a task using hierarchical decomposition"""
        if self.is_primitive(task):
            return [task]  # Base case: primitive action

        # Decompose complex task into subtasks
        subtasks = self.decompose_task(task, state)

        plan = []
        for subtask in subtasks:
            subplan = self.plan_task(subtask, state)
            plan.extend(subplan)
            # Update state after each subtask
            state = self.update_state(state, subplan[-1] if subplan else None)

        return plan

    def decompose_task(self, task, state):
        """Decompose a task into subtasks"""
        if task.name == "fetch_object":
            return [
                Task("navigate_to", target=task.params["location"]),
                Task("detect_object", target=task.params["object"]),
                Task("grasp_object", target=task.params["object"]),
                Task("navigate_to", target=task.params["delivery_location"]),
                Task("place_object", target=task.params["object"])
            ]
        # Add more task decompositions as needed
        return []

    def is_primitive(self, task):
        """Check if task is primitive (cannot be decomposed further)"""
        return task.name in self.primitive_actions
```

### Knowledge Representation

Representing knowledge about the world and robot capabilities:

```python
class KnowledgeBase:
    def __init__(self):
        self.objects = {}  # Object properties and locations
        self.locations = {}  # Spatial relationships
        self.capabilities = {}  # Robot capabilities
        self.affordances = {}  # Object affordances
        self.procedures = {}  # Known procedures

    def update_object_location(self, obj_name, location):
        """Update object location in knowledge base"""
        if obj_name not in self.objects:
            self.objects[obj_name] = {}
        self.objects[obj_name]['location'] = location
        self.objects[obj_name]['last_seen'] = time.time()

    def get_reachable_objects(self, robot_location):
        """Get objects reachable from robot location"""
        reachable = []
        for obj_name, obj_info in self.objects.items():
            if self.is_reachable(robot_location, obj_info.get('location')):
                reachable.append(obj_name)
        return reachable

    def is_reachable(self, location1, location2):
        """Check if location2 is reachable from location1"""
        # Implementation would use navigation map
        return True  # Simplified
```

## Cognitive Planning with LLMs

### LLM-Enhanced Planning

Using large language models for cognitive reasoning:

```python
class LLMCognitivePlanner:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.knowledge_base = KnowledgeBase()
        self.action_library = self.initialize_action_library()

    def plan_with_llm(self, goal, context):
        """Generate plan using LLM reasoning"""
        # Create structured prompt for planning
        prompt = self.create_planning_prompt(goal, context)

        response = self.llm_client.generate(prompt)

        # Parse LLM response into structured plan
        plan = self.parse_llm_plan(response)

        return plan

    def create_planning_prompt(self, goal, context):
        """Create prompt for LLM-based planning"""
        prompt = f"""
        You are a cognitive planner for a humanoid robot. Given the following information:

        Robot Capabilities: {self.action_library}
        Current Context: {context}
        Goal: {goal}

        Break down this goal into a sequence of executable actions. Each action should be:
        1. Specific and actionable
        2. Within the robot's capabilities
        3. Logically ordered
        4. Include necessary parameters

        Respond with a JSON list of actions in the format:
        [
            {{"action": "action_name", "parameters": {{"param1": "value1", ...}}}},
            ...
        ]
        """
        return prompt

    def parse_llm_plan(self, llm_response):
        """Parse LLM response into executable plan"""
        try:
            # Try to parse as JSON
            plan = json.loads(llm_response)
            return plan
        except json.JSONDecodeError:
            # If not JSON, try to extract using other methods
            return self.extract_plan_regex(llm_response)

    def validate_plan(self, plan):
        """Validate plan against robot capabilities"""
        for action in plan:
            action_name = action.get('action')
            if action_name not in self.action_library:
                raise ValueError(f"Unknown action: {action_name}")

            # Check parameters
            required_params = self.action_library[action_name].get('required_params', [])
            provided_params = action.get('parameters', {})

            for param in required_params:
                if param not in provided_params:
                    raise ValueError(f"Missing required parameter {param} for action {action_name}")

        return plan
```

## Task and Motion Planning Integration

### Combining High-Level and Low-Level Planning

```python
class IntegratedPlanner:
    def __init__(self):
        self.cognitive_planner = LLMCognitivePlanner()
        self.motion_planner = MotionPlanner()  # Navigation and manipulation planner
        self.executor = ActionExecutor()

    def execute_goal(self, goal, context):
        """Execute goal with integrated cognitive and motion planning"""
        # 1. Generate high-level plan
        high_level_plan = self.cognitive_planner.plan_with_llm(goal, context)

        # 2. Validate and refine plan
        validated_plan = self.cognitive_planner.validate_plan(high_level_plan)

        # 3. Execute plan with motion planning integration
        execution_result = self.execute_plan(validated_plan)

        return execution_result

    def execute_plan(self, plan):
        """Execute plan with real-time adaptation"""
        for i, action in enumerate(plan):
            try:
                # Get current state
                current_state = self.get_current_state()

                # Execute action
                result = self.executor.execute_action(action, current_state)

                if not result.success:
                    # Handle failure - replan or skip
                    if self.can_recover(action, result.error):
                        recovery_plan = self.generate_recovery_plan(action, result.error)
                        self.execute_plan(recovery_plan)
                    else:
                        # Skip to next action or abort
                        continue

            except Exception as e:
                self.get_logger().error(f"Error executing action {i}: {e}")
                # Implement error handling strategy
                return False

        return True

    def generate_recovery_plan(self, failed_action, error):
        """Generate recovery plan for failed action"""
        # Based on error type, generate appropriate recovery
        if "navigation" in str(error).lower():
            return self.generate_navigation_recovery(failed_action)
        elif "manipulation" in str(error).lower():
            return self.generate_manipulation_recovery(failed_action)
        else:
            return self.generate_general_recovery(failed_action)
```

## Context-Aware Planning

### Environmental Context Integration

```python
class ContextAwarePlanner:
    def __init__(self):
        self.spatial_reasoner = SpatialReasoner()
        self.temporal_reasoner = TemporalReasoner()
        self.social_reasoner = SocialReasoner()

    def plan_with_context(self, goal, environment_context):
        """Plan considering multiple contextual factors"""
        # Analyze spatial context
        spatial_constraints = self.analyze_spatial_context(
            environment_context['spatial']
        )

        # Analyze temporal context
        temporal_constraints = self.analyze_temporal_context(
            environment_context['temporal']
        )

        # Analyze social context (if humans present)
        social_constraints = self.analyze_social_context(
            environment_context.get('social', {})
        )

        # Generate plan considering all constraints
        plan = self.generate_contextual_plan(
            goal,
            spatial_constraints,
            temporal_constraints,
            social_constraints
        )

        return plan

    def analyze_spatial_context(self, spatial_data):
        """Analyze spatial context for planning"""
        constraints = {
            'obstacles': spatial_data.get('obstacles', []),
            'navigable_areas': spatial_data.get('navigable_areas', []),
            'object_locations': spatial_data.get('objects', {}),
            'safety_zones': spatial_data.get('safety_zones', [])
        }
        return constraints

    def analyze_temporal_context(self, temporal_data):
        """Analyze temporal context for planning"""
        constraints = {
            'time_limits': temporal_data.get('time_limits', {}),
            'recurring_events': temporal_data.get('recurring_events', []),
            'urgency_level': temporal_data.get('urgency', 'normal')
        }
        return constraints

    def analyze_social_context(self, social_data):
        """Analyze social context for planning"""
        constraints = {
            'human_locations': social_data.get('humans', []),
            'social_norms': social_data.get('norms', []),
            'interaction_preferences': social_data.get('preferences', {})
        }
        return constraints
```

## Planning for Humanoid-Specific Capabilities

### Bipedal Navigation Planning

```python
class HumanoidPlanner:
    def __init__(self):
        self.footstep_planner = FootstepPlanner()
        self.balance_controller = BalanceController()
        self.gait_generator = GaitGenerator()

    def plan_bipedal_navigation(self, start, goal, environment_map):
        """Plan navigation considering bipedal constraints"""
        # 1. Plan high-level path
        global_path = self.plan_global_path(start, goal, environment_map)

        # 2. Generate footstep plan
        footsteps = self.footstep_planner.plan_footsteps(
            global_path,
            start_pose=start
        )

        # 3. Validate balance throughout path
        if not self.validate_balance_path(footsteps):
            # Regenerate with balance constraints
            footsteps = self.generate_balance_aware_footsteps(
                global_path,
                start_pose=start
            )

        # 4. Generate gait pattern
        gait_pattern = self.gait_generator.generate_gait(footsteps)

        return {
            'path': global_path,
            'footsteps': footsteps,
            'gait': gait_pattern,
            'balance_checks': self.generate_balance_checks(footsteps)
        }

    def validate_balance_path(self, footsteps):
        """Validate that path maintains balance"""
        for i, footstep in enumerate(footsteps):
            # Check if footstep maintains balance
            if not self.balance_controller.is_stable_footstep(footstep):
                return False
        return True

    def generate_balance_aware_footsteps(self, path, start_pose):
        """Generate footsteps that maintain balance"""
        footsteps = []
        current_pose = start_pose.copy()

        for waypoint in path:
            # Calculate stable footstep to reach waypoint
            footstep = self.calculate_stable_footstep(current_pose, waypoint)
            footsteps.append(footstep)
            current_pose = self.update_pose_with_footstep(current_pose, footstep)

        return footsteps
```

## Multi-Modal Planning

### Integrating Vision and Language for Planning

```python
class MultiModalPlanner:
    def __init__(self):
        self.vision_system = VisionSystem()
        self.language_system = LanguageSystem()
        self.planning_system = HTNPlanner()

    def plan_from_multimodal_input(self, language_input, visual_input):
        """Plan using both language and visual inputs"""
        # 1. Process language input to extract goal
        language_context = self.language_system.process_input(language_input)
        goal = language_context.get('goal')
        constraints = language_context.get('constraints', {})

        # 2. Process visual input to understand environment
        visual_context = self.vision_system.process_input(visual_input)
        environment_state = visual_context.get('environment_state')
        detected_objects = visual_context.get('objects', [])

        # 3. Integrate multimodal information
        integrated_context = self.integrate_contexts(
            language_context,
            visual_context
        )

        # 4. Generate plan considering both modalities
        plan = self.planning_system.plan_task(
            goal,
            integrated_context
        )

        return plan

    def integrate_contexts(self, language_context, visual_context):
        """Integrate language and visual contexts"""
        integrated = {
            'spatial_info': visual_context.get('spatial_info', {}),
            'object_info': visual_context.get('objects', {}),
            'task_info': language_context.get('task_info', {}),
            'constraint_info': language_context.get('constraints', {}),
            'temporal_info': language_context.get('temporal_info', {})
        }

        # Resolve conflicts between modalities
        integrated = self.resolve_conflicts(integrated)

        return integrated

    def resolve_conflicts(self, integrated_context):
        """Resolve conflicts between different modalities"""
        # Example: If language says "red cup" but vision sees multiple cups
        # Use additional reasoning to identify the correct object
        return integrated_context
```

## Reactive Planning and Execution

### Real-Time Plan Adaptation

```python
class ReactivePlanner:
    def __init__(self):
        self.high_level_planner = HTNPlanner()
        self.monitoring_system = PlanMonitor()
        self.recovery_system = RecoverySystem()

    def execute_with_monitoring(self, plan, environment_callback):
        """Execute plan with real-time monitoring and adaptation"""
        execution_context = {
            'plan': plan,
            'current_step': 0,
            'execution_history': [],
            'environment_state': {}
        }

        while execution_context['current_step'] < len(plan):
            current_action = plan[execution_context['current_step']]

            # Monitor environment
            execution_context['environment_state'] = environment_callback()

            # Check if plan is still valid
            if not self.is_plan_valid(current_action, execution_context):
                # Replan or recover
                new_plan = self.adapt_plan(plan, execution_context)
                plan = new_plan

            # Execute action
            result = self.execute_action(current_action, execution_context)

            # Update execution context
            execution_context['execution_history'].append({
                'action': current_action,
                'result': result,
                'timestamp': time.time()
            })

            if result.success:
                execution_context['current_step'] += 1
            else:
                # Handle failure
                recovery_result = self.handle_failure(
                    current_action,
                    result,
                    execution_context
                )

                if recovery_result.success:
                    execution_context['current_step'] += 1
                else:
                    # Plan failed completely
                    return False

        return True

    def is_plan_valid(self, current_action, context):
        """Check if current plan is still valid"""
        # Check if environment has changed significantly
        # Check if robot state is as expected
        # Check if goal is still achievable
        return True

    def adapt_plan(self, original_plan, context):
        """Adapt plan based on new information"""
        # Strategies: skip action, replan from current step, global replan
        return original_plan  # Simplified
```

## Performance Optimization

### Planning Efficiency Techniques

```python
class EfficientPlanner:
    def __init__(self):
        self.plan_cache = {}
        self.heuristic_functions = {}
        self.parallel_planners = []

    def plan_with_optimization(self, goal, context):
        """Plan with performance optimizations"""
        # 1. Check plan cache
        cache_key = self.generate_cache_key(goal, context)
        if cache_key in self.plan_cache:
            cached_plan, timestamp = self.plan_cache[cache_key]
            if time.time() - timestamp < 300:  # 5 minutes
                return cached_plan

        # 2. Use hierarchical planning for complex tasks
        if self.is_complex_task(goal):
            plan = self.hierarchical_plan(goal, context)
        else:
            plan = self.direct_plan(goal, context)

        # 3. Cache the result
        self.cache_plan(cache_key, plan)

        return plan

    def hierarchical_plan(self, goal, context):
        """Use hierarchical planning for complex tasks"""
        # Decompose into subgoals
        subgoals = self.decompose_goal(goal)

        plan = []
        for subgoal in subgoals:
            subplan = self.direct_plan(subgoal, context)
            plan.extend(subplan)

        return plan

    def generate_cache_key(self, goal, context):
        """Generate cache key for plan caching"""
        return f"{hash(goal)}_{hash(str(context))}"
```

## Safety and Validation

### Plan Safety Checking

```python
class SafePlanner:
    def __init__(self):
        self.safety_rules = self.load_safety_rules()
        self.collision_checker = CollisionChecker()
        self.balance_checker = BalanceChecker()

    def generate_safe_plan(self, goal, context):
        """Generate plan with safety validation"""
        # Generate initial plan
        plan = self.planning_system.plan_task(goal, context)

        # Validate safety for each action
        safe_plan = self.validate_plan_safety(plan)

        return safe_plan

    def validate_plan_safety(self, plan):
        """Validate that plan is safe to execute"""
        safe_plan = []

        for action in plan:
            if self.is_action_safe(action):
                safe_plan.append(action)
            else:
                # Try to modify action to make it safe
                safe_action = self.make_action_safe(action)
                if safe_action:
                    safe_plan.append(safe_action)
                else:
                    raise ValueError(f"Cannot make action safe: {action}")

        return safe_plan

    def is_action_safe(self, action):
        """Check if action is safe to execute"""
        # Check collision safety
        if not self.check_collision_safety(action):
            return False

        # Check balance safety
        if not self.check_balance_safety(action):
            return False

        # Check other safety constraints
        if not self.check_general_safety(action):
            return False

        return True

    def check_collision_safety(self, action):
        """Check if action is collision-safe"""
        # Implementation would check planned trajectory
        return True

    def check_balance_safety(self, action):
        """Check if action maintains robot balance"""
        # Implementation would check balance during action
        return True
```

## Integration with ROS 2

### Planning Service Implementation

```python
import rclpy
from rclpy.node import Node
from vla_msgs.srv import PlanTask
from vla_msgs.msg import Plan, PlanStep
from geometry_msgs.msg import Pose
from std_msgs.msg import String

class CognitivePlanningNode(Node):
    def __init__(self):
        super().__init__('cognitive_planning_node')

        # Service for planning requests
        self.planning_service = self.create_service(
            PlanTask,
            'plan_task',
            self.handle_plan_request
        )

        # Publisher for plan visualization
        self.plan_pub = self.create_publisher(Plan, 'generated_plan', 10)

        # Initialize planners
        self.cognitive_planner = LLMCognitivePlanner()
        self.motion_planner = MotionPlanner()
        self.knowledge_base = KnowledgeBase()

        self.get_logger().info('Cognitive Planning Node initialized')

    def handle_plan_request(self, request, response):
        """Handle planning service request"""
        try:
            # Extract goal and context from request
            goal = request.goal
            context = self.extract_context_from_request(request)

            # Generate plan
            plan = self.cognitive_planner.plan_with_llm(goal, context)

            # Validate plan
            validated_plan = self.cognitive_planner.validate_plan(plan)

            # Convert to ROS message
            ros_plan = self.convert_to_ros_plan(validated_plan)

            # Publish plan for visualization
            self.plan_pub.publish(ros_plan)

            # Set response
            response.success = True
            response.plan = ros_plan
            response.message = "Plan generated successfully"

        except Exception as e:
            self.get_logger().error(f'Planning failed: {e}')
            response.success = False
            response.message = f"Planning failed: {e}"

        return response

    def extract_context_from_request(self, request):
        """Extract context from service request"""
        context = {
            'robot_state': request.robot_state,
            'environment_map': request.environment_map,
            'object_locations': request.object_locations,
            'constraints': request.constraints
        }
        return context

    def convert_to_ros_plan(self, plan):
        """Convert internal plan representation to ROS message"""
        ros_plan = Plan()
        ros_plan.header.stamp = self.get_clock().now().to_msg()
        ros_plan.header.frame_id = "map"

        for action in plan:
            step = PlanStep()
            step.action_name = action.get('action', '')
            step.parameters = json.dumps(action.get('parameters', {}))
            step.expected_duration = 0.0  # Would be calculated
            ros_plan.steps.append(step)

        return ros_plan
```

## Troubleshooting Common Issues

### Planning Failures

**Issue**: Plans fail to achieve goals consistently.

**Solutions**:
1. Improve state estimation accuracy
2. Add more detailed environment modeling
3. Implement better failure detection and recovery
4. Use more robust action primitives

**Issue**: Planning takes too long for real-time applications.

**Solutions**:
1. Use hierarchical planning to break down complex tasks
2. Implement plan caching for common scenarios
3. Use approximate planning methods when exact solutions are too slow
4. Optimize planning algorithms and data structures

### Integration Challenges

**Issue**: Cognitive plans don't align with low-level execution capabilities.

**Solutions**:
1. Maintain consistent action representations across planning levels
2. Implement proper plan refinement between levels
3. Use shared knowledge representations
4. Test plans in simulation before real-world execution

## Best Practices

### System Design

- **Modular Architecture**: Keep planning components separate for maintainability
- **Fallback Mechanisms**: Implement graceful degradation when planning fails
- **Performance Monitoring**: Track planning time, success rates, and quality
- **Validation**: Always validate plans before execution

### Knowledge Management

- **Consistent Representations**: Use consistent data formats across components
- **Regular Updates**: Keep knowledge base updated with current information
- **Uncertainty Handling**: Account for uncertainty in planning processes
- **Learning**: Incorporate learning from execution outcomes

## Next Steps

Continue to [Multi-Modal Processing](./multi-modal.md) to learn about integrating multiple sensory inputs for comprehensive scene understanding in VLA systems.