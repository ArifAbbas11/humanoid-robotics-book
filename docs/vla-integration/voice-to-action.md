# Voice-to-Action Mapping

## Overview

Voice-to-Action mapping is the process of converting natural language voice commands into executable robot actions. This critical component of Vision-Language-Action (VLA) systems bridges human communication with robot behavior, enabling intuitive human-robot interaction through speech.

## Voice-to-Action Architecture

### The Mapping Pipeline

The voice-to-action pipeline typically follows this sequence:

1. **Voice Input**: User speaks a command to the robot
2. **Speech Recognition**: Converts speech to text
3. **Natural Language Understanding**: Interprets the meaning of the command
4. **Action Mapping**: Maps the understood command to specific robot actions
5. **Action Validation**: Ensures actions are safe and feasible
6. **Action Execution**: Executes the mapped actions on the robot
7. **Feedback**: Provides feedback to the user about the execution

### System Components

- **Voice Recognition Module**: Converts speech to text
- **Language Understanding Module**: Parses and interprets commands
- **Action Mapping Engine**: Maps commands to robot actions
- **Action Executor**: Executes the actions on the robot
- **Feedback System**: Communicates execution status to the user

## Natural Language Understanding for Action Mapping

### Command Parsing

```python
import re
from typing import Dict, List, Tuple

class CommandParser:
    def __init__(self):
        # Define action patterns
        self.action_patterns = {
            'navigation': [
                r'go to (.+)',
                r'move to (.+)',
                r'walk to (.+)',
                r'navigate to (.+)',
                r'go (.+)'
            ],
            'manipulation': [
                r'pick up (.+)',
                r'grasp (.+)',
                r'take (.+)',
                r'get (.+)',
                r'put (.+) (?:on|in) (.+)',
                r'place (.+) (?:on|in) (.+)'
            ],
            'interaction': [
                r'greet (.+)',
                r'say hello to (.+)',
                r'wave to (.+)',
                r'nod to (.+)'
            ],
            'information': [
                r'what is (.+)',
                r'where is (.+)',
                r'find (.+)',
                r'show me (.+)'
            ]
        }

    def parse_command(self, text: str) -> Dict:
        """Parse natural language command into structured format"""
        text_lower = text.lower().strip()

        for action_type, patterns in self.action_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    params = match.groups()
                    return {
                        'action_type': action_type,
                        'action_pattern': pattern,
                        'parameters': params,
                        'original_text': text,
                        'confidence': 0.9  # High confidence for regex match
                    }

        # If no pattern matches, return unknown
        return {
            'action_type': 'unknown',
            'action_pattern': None,
            'parameters': [],
            'original_text': text,
            'confidence': 0.0
        }

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities like objects, locations, people from text"""
        entities = {
            'objects': [],
            'locations': [],
            'people': [],
            'actions': []
        }

        # Simple keyword-based entity extraction
        object_keywords = ['cup', 'book', 'ball', 'bottle', 'box', 'chair', 'table']
        location_keywords = ['kitchen', 'living room', 'bedroom', 'office', 'hall', 'door']
        people_keywords = ['person', 'man', 'woman', 'child', 'me', 'you']

        text_lower = text.lower()
        words = text_lower.split()

        for keyword in object_keywords:
            if keyword in text_lower:
                entities['objects'].append(keyword)

        for keyword in location_keywords:
            if keyword in text_lower:
                entities['locations'].append(keyword)

        for keyword in people_keywords:
            if keyword in text_lower:
                entities['people'].append(keyword)

        return entities
```

### Intent Classification

```python
class IntentClassifier:
    def __init__(self):
        self.intent_keywords = {
            'navigation': ['go', 'move', 'walk', 'navigate', 'to', 'toward', 'towards'],
            'manipulation': ['pick', 'grasp', 'take', 'get', 'put', 'place', 'hold', 'drop'],
            'interaction': ['greet', 'hello', 'wave', 'talk', 'speak', 'chat'],
            'information': ['what', 'where', 'find', 'show', 'tell', 'describe']
        }

    def classify_intent(self, text: str) -> str:
        """Classify the intent of a voice command"""
        text_lower = text.lower()
        scores = {}

        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[intent] = score

        # Return intent with highest score
        if scores:
            return max(scores, key=scores.get)
        return 'unknown'
```

## Action Mapping Strategies

### Rule-Based Mapping

Simple rule-based approach for mapping commands to actions:

```python
class RuleBasedActionMapper:
    def __init__(self):
        self.action_mapping_rules = {
            # Navigation commands
            ('navigation', 'go to'): self.map_navigation,
            ('navigation', 'move to'): self.map_navigation,
            ('navigation', 'walk to'): self.map_navigation,

            # Manipulation commands
            ('manipulation', 'pick up'): self.map_manipulation_pick,
            ('manipulation', 'grasp'): self.map_manipulation_grasp,
            ('manipulation', 'put'): self.map_manipulation_place,

            # Interaction commands
            ('interaction', 'greet'): self.map_interaction_greet,
            ('interaction', 'wave'): self.map_interaction_wave,
        }

    def map_action(self, parsed_command: Dict) -> List[Dict]:
        """Map parsed command to robot actions"""
        action_type = parsed_command['action_type']
        pattern = parsed_command['action_pattern']

        key = (action_type, self.extract_action_from_pattern(pattern))

        if key in self.action_mapping_rules:
            return self.action_mapping_rules[key](parsed_command)
        else:
            return self.default_mapping(parsed_command)

    def extract_action_from_pattern(self, pattern: str) -> str:
        """Extract action from regex pattern"""
        # Simple extraction - in practice, this would be more sophisticated
        parts = pattern.split()
        if parts:
            return parts[0]
        return pattern

    def map_navigation(self, parsed_command: Dict) -> List[Dict]:
        """Map navigation commands to robot actions"""
        destination = parsed_command['parameters'][0] if parsed_command['parameters'] else None

        actions = [{
            'action_type': 'navigation',
            'action_name': 'navigate_to',
            'parameters': {
                'destination': destination,
                'speed': 'normal'
            },
            'description': f'Navigating to {destination}'
        }]

        return actions

    def map_manipulation_pick(self, parsed_command: Dict) -> List[Dict]:
        """Map pick-up commands to robot actions"""
        object_to_pick = parsed_command['parameters'][0] if parsed_command['parameters'] else None

        actions = [
            {
                'action_type': 'navigation',
                'action_name': 'navigate_to_object',
                'parameters': {
                    'object': object_to_pick,
                    'approach_distance': 0.5
                },
                'description': f'Navigating to {object_to_pick}'
            },
            {
                'action_type': 'manipulation',
                'action_name': 'grasp_object',
                'parameters': {
                    'object': object_to_pick
                },
                'description': f'Grasping {object_to_pick}'
            }
        ]

        return actions

    def default_mapping(self, parsed_command: Dict) -> List[Dict]:
        """Default mapping for unknown commands"""
        return [{
            'action_type': 'unknown',
            'action_name': 'unknown_command',
            'parameters': {},
            'description': f"Unknown command: {parsed_command['original_text']}"
        }]
```

### Machine Learning-Based Mapping

Using neural networks for more sophisticated mapping:

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class MLActionMapper(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_actions):
        super().__init__()

        # Text encoding
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')

        # Action prediction head
        self.action_classifier = nn.Sequential(
            nn.Linear(768, hidden_dim),  # BERT hidden size is 768
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_actions)
        )

        # Action parameter prediction
        self.param_predictor = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128)  # Predict 128 parameters (flexible)
        )

    def forward(self, text_inputs):
        """Forward pass through the action mapping network"""
        # Encode text
        encoded = self.text_encoder(**text_inputs)
        pooled_output = encoded.pooler_output  # [batch_size, 768]

        # Predict action
        action_logits = self.action_classifier(pooled_output)
        action_probs = torch.softmax(action_logits, dim=-1)

        # Predict parameters
        param_logits = self.param_predictor(pooled_output)

        return action_probs, param_logits

    def map_command(self, text: str):
        """Map text command to action using the neural network"""
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )

        # Get predictions
        action_probs, param_logits = self.forward(inputs)

        # Decode predictions to action
        predicted_action = torch.argmax(action_probs, dim=-1).item()
        predicted_params = param_logits.detach().cpu().numpy()

        return {
            'action_id': predicted_action,
            'action_params': predicted_params,
            'confidence': action_probs.max().item()
        }
```

## Context-Aware Action Mapping

### Environment Context Integration

```python
class ContextAwareActionMapper:
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.vision_processor = VisionProcessor()
        self.action_validator = ActionValidator()

    def map_with_context(self, command: str, environment_context: Dict):
        """Map command considering environmental context"""
        # Parse the command
        parsed_command = self.parse_command(command)

        # Get current environment state
        current_state = self.get_environment_state(environment_context)

        # Map command with context awareness
        raw_actions = self.map_command_to_actions(parsed_command)

        # Validate actions against current state
        validated_actions = self.validate_actions(raw_actions, current_state)

        # Adapt actions based on context
        adapted_actions = self.adapt_actions_to_context(
            validated_actions,
            current_state,
            parsed_command
        )

        return adapted_actions

    def get_environment_state(self, context: Dict) -> Dict:
        """Extract current environment state from context"""
        state = {
            'robot_location': context.get('robot_location'),
            'detected_objects': context.get('detected_objects', []),
            'reachable_objects': context.get('reachable_objects', []),
            'navigable_areas': context.get('navigable_areas', []),
            'human_positions': context.get('humans', []),
            'robot_state': context.get('robot_state', {})
        }
        return state

    def validate_actions(self, actions: List[Dict], state: Dict) -> List[Dict]:
        """Validate actions against current environment state"""
        valid_actions = []

        for action in actions:
            if self.action_validator.is_valid(action, state):
                valid_actions.append(action)
            else:
                # Try to adapt the action
                adapted = self.adapt_invalid_action(action, state)
                if adapted:
                    valid_actions.append(adapted)

        return valid_actions

    def adapt_actions_to_context(self, actions: List[Dict], state: Dict, command: Dict) -> List[Dict]:
        """Adapt actions based on environmental context"""
        adapted_actions = []

        for action in actions:
            if action['action_type'] == 'navigation':
                # Check if destination is navigable
                destination = action['parameters'].get('destination')
                if destination:
                    # Resolve destination to specific location
                    resolved_location = self.resolve_location(destination, state)
                    action['parameters']['destination'] = resolved_location

            elif action['action_type'] == 'manipulation':
                # Check if object is reachable
                target_object = action['parameters'].get('object')
                if target_object:
                    # Find the specific object instance
                    object_instance = self.find_object_instance(target_object, state)
                    if object_instance:
                        action['parameters']['object_pose'] = object_instance['pose']

            adapted_actions.append(action)

        return adapted_actions

    def resolve_location(self, location_desc: str, state: Dict) -> str:
        """Resolve location description to specific coordinates"""
        # Look up in knowledge base
        known_locations = self.knowledge_base.get_locations()

        for loc_name, loc_info in known_locations.items():
            if location_desc.lower() in loc_name.lower():
                return loc_info['coordinates']

        # If not found, try to find in current context
        for nav_area in state['navigable_areas']:
            if location_desc.lower() in nav_area.get('name', '').lower():
                return nav_area['coordinates']

        # Default to current location if can't resolve
        return state['robot_location']
```

## Action Execution Planning

### Sequential Action Planning

```python
class ActionExecutionPlanner:
    def __init__(self):
        self.action_library = self.load_action_library()
        self.precondition_checker = PreconditionChecker()
        self.effect_predictor = EffectPredictor()

    def plan_execution_sequence(self, actions: List[Dict]) -> List[Dict]:
        """Plan sequence of actions for execution"""
        execution_plan = []

        for action in actions:
            # Check preconditions
            if not self.precondition_checker.check(action):
                # Try to satisfy preconditions
                precondition_actions = self.satisfy_preconditions(action)
                execution_plan.extend(precondition_actions)

            # Add the main action
            execution_plan.append(action)

            # Update expected state effects
            self.effect_predictor.update_state(action)

        return execution_plan

    def satisfy_preconditions(self, action: Dict) -> List[Dict]:
        """Generate actions to satisfy preconditions for a given action"""
        preconditions = self.action_library[action['action_name']].get('preconditions', [])
        precondition_actions = []

        for precondition in preconditions:
            if not self.precondition_checker.is_satisfied(precondition):
                # Generate action to satisfy precondition
                satisfying_action = self.generate_satisfying_action(precondition)
                if satisfying_action:
                    precondition_actions.append(satisfying_action)

        return precondition_actions

    def generate_satisfying_action(self, precondition: Dict) -> Dict:
        """Generate action to satisfy a specific precondition"""
        # Example: if precondition is "robot_at_location", generate navigation action
        if precondition.get('type') == 'robot_at_location':
            return {
                'action_type': 'navigation',
                'action_name': 'navigate_to',
                'parameters': {
                    'destination': precondition['location']
                },
                'description': f'Navigating to satisfy precondition: at {precondition["location"]}'
            }

        # Add more precondition types as needed
        return None
```

## ROS 2 Integration

### Voice-to-Action Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from vla_msgs.msg import ActionCommand, ActionResult
from vla_msgs.srv import ExecuteAction

class VoiceToActionNode(Node):
    def __init__(self):
        super().__init__('voice_to_action_node')

        # Subscribers
        self.voice_sub = self.create_subscription(
            String,
            'recognized_text',
            self.voice_callback,
            10
        )

        # Publishers
        self.action_pub = self.create_publisher(
            ActionCommand,
            'robot_action_commands',
            10
        )

        self.feedback_pub = self.create_publisher(
            String,
            'action_feedback',
            10
        )

        # Services
        self.execute_service = self.create_service(
            ExecuteAction,
            'execute_mapped_action',
            self.execute_action_callback
        )

        # Initialize action mapping components
        self.command_parser = CommandParser()
        self.intent_classifier = IntentClassifier()
        self.action_mapper = RuleBasedActionMapper()
        self.context_aware_mapper = ContextAwareActionMapper()

        self.get_logger().info('Voice-to-Action Node initialized')

    def voice_callback(self, msg: String):
        """Process voice command from speech recognition"""
        command_text = msg.data
        self.get_logger().info(f'Received voice command: {command_text}')

        try:
            # Parse the command
            parsed_command = self.command_parser.parse_command(command_text)

            if parsed_command['confidence'] > 0.5:  # Threshold
                # Map to actions
                actions = self.action_mapper.map_action(parsed_command)

                # Publish actions for execution
                for action in actions:
                    action_msg = self.create_action_message(action)
                    self.action_pub.publish(action_msg)

                # Provide feedback
                feedback_msg = String()
                feedback_msg.data = f"Understood command: {command_text}, executing {len(actions)} actions"
                self.feedback_pub.publish(feedback_msg)

            else:
                # Low confidence - ask for clarification
                feedback_msg = String()
                feedback_msg.data = f"Sorry, I didn't understand: {command_text}. Could you repeat?"
                self.feedback_pub.publish(feedback_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing voice command: {e}')
            feedback_msg = String()
            feedback_msg.data = f"Error processing command: {e}"
            self.feedback_pub.publish(feedback_msg)

    def execute_action_callback(self, request, response):
        """Service callback for executing mapped actions"""
        try:
            # Get current context (would typically come from other nodes)
            context = self.get_current_context()

            # Map the command with context
            actions = self.context_aware_mapper.map_with_context(
                request.command,
                context
            )

            # Execute actions
            execution_results = []
            for action in actions:
                result = self.execute_single_action(action)
                execution_results.append(result)

            # Set response
            response.success = all(result['success'] for result in execution_results)
            response.results = execution_results
            response.message = "Actions executed successfully" if response.success else "Some actions failed"

        except Exception as e:
            self.get_logger().error(f'Action execution failed: {e}')
            response.success = False
            response.message = f"Execution failed: {e}"

        return response

    def get_current_context(self) -> Dict:
        """Get current environment context"""
        # This would typically subscribe to various sensor topics
        # and aggregate the information
        return {
            'robot_location': 'kitchen',
            'detected_objects': ['red cup', 'blue book'],
            'reachable_objects': ['red cup'],
            'navigable_areas': ['kitchen', 'living room'],
            'robot_state': {'battery': 85, 'gripper': 'open'}
        }

    def execute_single_action(self, action: Dict) -> Dict:
        """Execute a single action"""
        # This would typically call other ROS services/nodes
        # to execute the specific action
        return {
            'action_name': action['action_name'],
            'success': True,
            'message': f"Executed {action['action_name']}",
            'execution_time': 0.0
        }

    def create_action_message(self, action: Dict) -> ActionCommand:
        """Create ROS message from action dictionary"""
        action_msg = ActionCommand()
        action_msg.action_type = action['action_type']
        action_msg.action_name = action['action_name']
        action_msg.parameters = str(action['parameters'])  # Convert to string for simplicity
        action_msg.description = action['description']
        action_msg.header.stamp = self.get_clock().now().to_msg()
        return action_msg
```

## Advanced Mapping Techniques

### Semantic Action Mapping

```python
class SemanticActionMapper:
    def __init__(self):
        self.semantic_parser = SemanticParser()
        self.action_space = ActionSpace()
        self.reasoning_engine = ReasoningEngine()

    def map_with_semantics(self, command: str) -> List[Dict]:
        """Map command using semantic understanding"""
        # Parse command semantically
        semantic_structure = self.semantic_parser.parse(command)

        # Ground semantics in robot capabilities
        grounded_actions = self.ground_semantics(semantic_structure)

        # Reason about the best action sequence
        reasoned_actions = self.reasoning_engine.reason(
            grounded_actions,
            semantic_structure
        )

        return reasoned_actions

    def ground_semantics(self, semantic_structure: Dict) -> List[Dict]:
        """Ground semantic structure in robot action space"""
        actions = []

        # Example semantic structure processing
        if semantic_structure.get('action') == 'transport':
            source = semantic_structure.get('source')
            target = semantic_structure.get('target')
            object = semantic_structure.get('object')

            actions = [
                {
                    'action_type': 'navigation',
                    'action_name': 'navigate_to',
                    'parameters': {'destination': source}
                },
                {
                    'action_type': 'manipulation',
                    'action_name': 'grasp',
                    'parameters': {'object': object}
                },
                {
                    'action_type': 'navigation',
                    'action_name': 'navigate_to',
                    'parameters': {'destination': target}
                },
                {
                    'action_type': 'manipulation',
                    'action_name': 'place',
                    'parameters': {'object': object}
                }
            ]

        return actions
```

## Error Handling and Recovery

### Robust Action Mapping

```python
class RobustActionMapper:
    def __init__(self):
        self.fallback_strategies = [
            self.fallback_to_simple_navigation,
            self.fallback_to_manual_control_request,
            self.fallback_to_context_query
        ]

    def map_with_fallback(self, command: str, context: Dict) -> List[Dict]:
        """Map command with fallback strategies"""
        try:
            # Primary mapping
            actions = self.primary_mapping(command, context)

            # Validate actions
            if self.validate_actions(actions, context):
                return actions
        except Exception as e:
            self.get_logger().warning(f'Primary mapping failed: {e}')

        # Try fallback strategies
        for fallback_strategy in self.fallback_strategies:
            try:
                fallback_actions = fallback_strategy(command, context)
                if fallback_actions and self.validate_actions(fallback_actions, context):
                    return fallback_actions
            except Exception as e:
                self.get_logger().warning(f'Fallback strategy failed: {e}')
                continue

        # If all strategies fail, return error action
        return [{
            'action_type': 'error',
            'action_name': 'unknown_command',
            'parameters': {'original_command': command},
            'description': f'Unable to map command: {command}'
        }]

    def validate_actions(self, actions: List[Dict], context: Dict) -> bool:
        """Validate that actions are feasible in current context"""
        for action in actions:
            if not self.is_action_feasible(action, context):
                return False
        return True

    def is_action_feasible(self, action: Dict, context: Dict) -> bool:
        """Check if action is feasible in current context"""
        # Check robot capabilities
        if not self.has_capability(action['action_name']):
            return False

        # Check safety constraints
        if not self.is_safe(action, context):
            return False

        # Check resource constraints
        if not self.has_resources(action, context):
            return False

        return True

    def fallback_to_simple_navigation(self, command: str, context: Dict) -> List[Dict]:
        """Fallback to simple navigation if command is unclear"""
        # Extract potential location from command
        location = self.extract_location(command)
        if location:
            return [{
                'action_type': 'navigation',
                'action_name': 'navigate_to',
                'parameters': {'destination': location},
                'description': f'Navigating to {location} (fallback)'
            }]
        return []

    def extract_location(self, command: str) -> str:
        """Extract location from command"""
        # Simple keyword-based extraction
        locations = ['kitchen', 'living room', 'bedroom', 'office', 'hall']
        command_lower = command.lower()

        for location in locations:
            if location in command_lower:
                return location

        return None
```

## Performance Optimization

### Efficient Command Processing

```python
class EfficientCommandProcessor:
    def __init__(self):
        self.command_cache = {}
        self.pattern_matcher = OptimizedPatternMatcher()
        self.command_templates = self.load_command_templates()

    def process_command_efficiently(self, command: str) -> List[Dict]:
        """Process command efficiently with caching and optimization"""
        # Check cache first
        if command in self.command_cache:
            cached_result, timestamp = self.command_cache[command]
            if time.time() - timestamp < 300:  # 5 minute cache
                return cached_result

        # Use optimized pattern matching
        matched_template = self.pattern_matcher.match(command)

        if matched_template:
            actions = self.generate_actions_from_template(matched_template)
        else:
            # Fall back to full processing
            actions = self.full_command_processing(command)

        # Cache result
        self.cache_command_result(command, actions)

        return actions

    def generate_actions_from_template(self, template: Dict) -> List[Dict]:
        """Generate actions from matched template"""
        # Template-based action generation is faster than parsing
        return template['actions']

    def cache_command_result(self, command: str, result: List[Dict]):
        """Cache command processing result"""
        if len(self.command_cache) > 100:  # Limit cache size
            # Remove oldest entries
            oldest_key = next(iter(self.command_cache))
            del self.command_cache[oldest_key]

        self.command_cache[command] = (result, time.time())
```

## Quality Assessment

### Action Mapping Quality Metrics

```python
class ActionMappingQualityAssessment:
    def __init__(self):
        self.metrics = {
            'accuracy': 0,
            'completeness': 0,
            'safety': 0,
            'efficiency': 0
        }

    def assess_mapping_quality(self, command: str, mapped_actions: List[Dict], expected_actions: List[Dict] = None) -> Dict:
        """Assess quality of action mapping"""
        quality_metrics = {}

        # Accuracy: How well do mapped actions match expected actions?
        if expected_actions:
            quality_metrics['accuracy'] = self.compute_accuracy(
                mapped_actions, expected_actions
            )

        # Completeness: Do mapped actions cover the full intent?
        quality_metrics['completeness'] = self.compute_completeness(
            command, mapped_actions
        )

        # Safety: Are mapped actions safe to execute?
        quality_metrics['safety'] = self.compute_safety_score(mapped_actions)

        # Efficiency: How many actions are needed?
        quality_metrics['efficiency'] = self.compute_efficiency(mapped_actions)

        return quality_metrics

    def compute_accuracy(self, mapped: List[Dict], expected: List[Dict]) -> float:
        """Compute accuracy of action mapping"""
        if not expected:
            return 0.0

        correct_actions = 0
        for exp_action in expected:
            for map_action in mapped:
                if (exp_action['action_name'] == map_action['action_name'] and
                    self.parameters_match(exp_action['parameters'], map_action['parameters'])):
                    correct_actions += 1
                    break

        return correct_actions / len(expected)

    def compute_completeness(self, command: str, actions: List[Dict]) -> float:
        """Compute how completely the command intent is addressed"""
        # This would involve analyzing whether the actions address all aspects of the command
        return 1.0 if actions else 0.0

    def compute_safety_score(self, actions: List[Dict]) -> float:
        """Compute safety score for the action sequence"""
        # Check each action for safety
        safe_actions = sum(1 for action in actions if self.is_action_safe(action))
        return safe_actions / len(actions) if actions else 0.0

    def parameters_match(self, params1: Dict, params2: Dict, threshold: float = 0.8) -> bool:
        """Check if parameters approximately match"""
        # Implementation would compare parameter values
        return True
```

## Troubleshooting Common Issues

### Command Understanding Problems

**Issue**: Commands are not being understood correctly.

**Solutions**:
1. Expand command pattern database
2. Improve natural language preprocessing
3. Add context-aware disambiguation
4. Implement user feedback learning

**Issue**: Ambiguous commands lead to incorrect actions.

**Solutions**:
1. Implement clarification requests
2. Use confidence thresholds
3. Add disambiguation strategies
4. Maintain command history for context

### Action Execution Problems

**Issue**: Mapped actions fail during execution.

**Solutions**:
1. Improve action validation before execution
2. Add simulation-based verification
3. Implement robust error handling
4. Use gradual action refinement

## Best Practices

### System Design

- **Modular Architecture**: Keep command parsing, mapping, and execution separate
- **Fallback Mechanisms**: Always have fallback strategies for failed mappings
- **User Feedback**: Provide clear feedback about command understanding
- **Safety First**: Validate all actions before execution

### Performance Considerations

- **Caching**: Cache frequently used command mappings
- **Optimization**: Use efficient pattern matching algorithms
- **Parallel Processing**: Process multiple aspects of commands in parallel
- **Real-time Constraints**: Ensure mapping completes within real-time requirements

## Next Steps

Continue to [Capstone Project](./capstone-project.md) to apply all VLA integration concepts in a comprehensive humanoid robotics project.