# LLM Integration

## Overview

Large Language Model (LLM) integration is a crucial component of Vision-Language-Action (VLA) systems, providing the natural language understanding and reasoning capabilities that allow humanoid robots to interpret complex commands and engage in meaningful conversations. This integration bridges human language with robotic action execution.

## LLM Fundamentals

### What are Large Language Models?

Large Language Models are deep learning models trained on vast amounts of text data to understand and generate human-like language. They can perform various natural language tasks including:

- **Text Generation**: Creating coherent, contextually relevant text
- **Question Answering**: Providing answers to user queries
- **Instruction Following**: Executing tasks based on natural language instructions
- **Reasoning**: Performing logical inference and problem solving
- **Context Understanding**: Maintaining context across conversations

### Popular LLM Architectures

- **Transformer Architecture**: Foundation for most modern LLMs
- **GPT Series**: Generative Pre-trained Transformers (OpenAI)
- **LLaMA**: Open-source models from Meta
- **PaLM**: Pathways Language Model from Google
- **Claude**: Anthropic's conversational AI models

## LLM Integration Approaches

### Cloud-Based APIs

Cloud-based LLM services provide easy integration with minimal setup:

- **OpenAI API**: GPT models with comprehensive documentation
- **Anthropic API**: Claude models focused on helpful, harmless responses
- **Google AI API**: PaLM and Gemini models
- **AWS Bedrock**: Managed LLM services
- **Azure OpenAI**: Microsoft's managed OpenAI service

### Local Deployment

Local deployment provides privacy and reduced latency:

- **Ollama**: Simple local LLM serving
- **vLLM**: Fast LLM inference engine
- **Hugging Face Transformers**: Open-source model library
- **TensorRT-LLM**: NVIDIA's optimized inference engine
- **llama.cpp**: Lightweight LLM inference in C++

## ROS 2 Integration Patterns

### LLM Service Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vla_msgs.srv import LLMQuery  # Custom service
import openai
import asyncio
import threading

class LLMIntegrationNode(Node):
    def __init__(self):
        super().__init__('llm_integration_node')

        # Service for LLM queries
        self.llm_service = self.create_service(
            LLMQuery,
            'llm_query',
            self.handle_llm_request
        )

        # Publishers for LLM responses
        self.response_pub = self.create_publisher(String, 'llm_response', 10)

        # Initialize LLM client
        self.llm_client = self.initialize_llm_client()

        # Conversation history
        self.conversation_history = []

        self.get_logger().info('LLM Integration Node initialized')

    def initialize_llm_client(self):
        """Initialize the LLM client based on chosen provider"""
        # Example for OpenAI
        try:
            openai.api_key = self.get_parameter_or_set_default(
                'openai_api_key',
                'your-api-key-here'
            ).value

            return openai
        except Exception as e:
            self.get_logger().error(f'Failed to initialize LLM client: {e}')
            return None

    def handle_llm_request(self, request, response):
        """Handle LLM query requests"""
        try:
            # Add to conversation history
            self.conversation_history.append({
                'role': 'user',
                'content': request.query
            })

            # Generate response from LLM
            llm_response = self.query_llm(
                self.conversation_history,
                request.context
            )

            # Process the response
            processed_response = self.process_llm_response(llm_response)

            # Update conversation history
            self.conversation_history.append({
                'role': 'assistant',
                'content': processed_response
            })

            # Publish response
            response_msg = String()
            response_msg.data = processed_response
            self.response_pub.publish(response_msg)

            # Set service response
            response.success = True
            response.response = processed_response

        except Exception as e:
            self.get_logger().error(f'LLM request failed: {e}')
            response.success = False
            response.response = f'Error processing request: {e}'

        return response

    def query_llm(self, messages, context=None):
        """Query the LLM with conversation history"""
        try:
            # Prepare messages with context if provided
            full_messages = messages.copy()

            if context:
                full_messages.insert(0, {
                    'role': 'system',
                    'content': f'Context: {context}\n\n'
                              f'You are a helpful assistant for controlling a humanoid robot. '
                              f'Interpret the user\'s requests and provide appropriate responses. '
                              f'If the user wants to control the robot, respond with structured commands.'
                })

            # Call the LLM API
            result = self.llm_client.ChatCompletion.create(
                model="gpt-3.5-turbo",  # or gpt-4, Claude, etc.
                messages=full_messages,
                max_tokens=500,
                temperature=0.7
            )

            return result.choices[0].message.content

        except Exception as e:
            self.get_logger().error(f'LLM API call failed: {e}')
            return f"Sorry, I couldn't process your request: {e}"

    def process_llm_response(self, response):
        """Process and format LLM response for the application"""
        # Remove any potentially harmful content
        # Format response appropriately
        # Extract structured commands if present
        return response.strip()
```

## Using Open-Source LLMs Locally

### Ollama Integration

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vla_msgs.srv import LLMQuery
import requests
import json

class OllamaLLMNode(Node):
    def __init__(self):
        super().__init__('ollama_llm_node')

        # Service for LLM queries
        self.llm_service = self.create_service(
            LLMQuery,
            'ollama_query',
            self.handle_ollama_request
        )

        # Publisher for responses
        self.response_pub = self.create_publisher(String, 'ollama_response', 10)

        # Ollama configuration
        self.ollama_url = 'http://localhost:11434/api/generate'
        self.model_name = 'llama2'  # or 'mistral', 'phi', etc.

        # Test connection
        if self.test_connection():
            self.get_logger().info('Ollama connection established')
        else:
            self.get_logger().error('Failed to connect to Ollama')

    def test_connection(self):
        """Test connection to Ollama server"""
        try:
            response = requests.get('http://localhost:11434/api/tags')
            return response.status_code == 200
        except Exception:
            return False

    def handle_ollama_request(self, request, response):
        """Handle LLM request using Ollama"""
        try:
            # Prepare the request payload
            payload = {
                'model': self.model_name,
                'prompt': request.query,
                'stream': False,
                'options': {
                    'temperature': 0.7,
                    'num_ctx': 2048
                }
            }

            # Send request to Ollama
            result = requests.post(self.ollama_url, json=payload)

            if result.status_code == 200:
                result_data = result.json()
                llm_response = result_data.get('response', '')

                # Publish and return response
                response_msg = String()
                response_msg.data = llm_response
                self.response_pub.publish(response_msg)

                response.success = True
                response.response = llm_response
            else:
                error_msg = f'Ollama request failed: {result.status_code}'
                self.get_logger().error(error_msg)
                response.success = False
                response.response = error_msg

        except Exception as e:
            error_msg = f'Ollama processing error: {e}'
            self.get_logger().error(error_msg)
            response.success = False
            response.response = error_msg

        return response
```

## Context-Aware LLM Integration

### Environment Context Integration

```python
class ContextAwareLLMNode(Node):
    def __init__(self):
        super().__init__('context_aware_llm_node')

        # Service for contextual queries
        self.contextual_service = self.create_service(
            LLMQuery,
            'contextual_llm_query',
            self.handle_contextual_request
        )

        # Subscribers for environmental context
        self.vision_sub = self.create_subscription(
            String,  # Simplified - would typically be sensor_msgs
            'vision_context',
            self.vision_callback,
            10
        )

        self.location_sub = self.create_subscription(
            String,
            'robot_location',
            self.location_callback,
            10
        )

        # Store environmental context
        self.current_vision_context = ""
        self.current_location = ""
        self.robot_capabilities = self.get_robot_capabilities()

    def get_robot_capabilities(self):
        """Get robot's current capabilities and limitations"""
        return {
            'manipulation': True,
            'navigation': True,
            'sensors': ['camera', 'lidar', 'imu'],
            'max_speed': 0.5,
            'weight_limit': 2.0,  # kg
            'reachable_area': 'humanoid_workspace'
        }

    def vision_callback(self, msg):
        """Update vision context"""
        self.current_vision_context = msg.data

    def location_callback(self, msg):
        """Update location context"""
        self.current_location = msg.data

    def handle_contextual_request(self, request, response):
        """Handle LLM request with environmental context"""
        try:
            # Build comprehensive context
            context = self.build_context(request.query)

            # Query LLM with context
            llm_response = self.query_llm_with_context(
                request.query,
                context
            )

            # Process and return response
            processed_response = self.process_contextual_response(
                llm_response,
                request.query
            )

            response.success = True
            response.response = processed_response

        except Exception as e:
            self.get_logger().error(f'Contextual LLM request failed: {e}')
            response.success = False
            response.response = f'Error: {e}'

        return response

    def build_context(self, query):
        """Build comprehensive context for the LLM"""
        context_parts = []

        # Add location context
        if self.current_location:
            context_parts.append(f"Robot Location: {self.current_location}")

        # Add vision context
        if self.current_vision_context:
            context_parts.append(f"Current View: {self.current_vision_context}")

        # Add robot capabilities
        capabilities = f"Robot Capabilities: {self.robot_capabilities}"
        context_parts.append(capabilities)

        # Add time context
        current_time = self.get_clock().now().to_msg()
        context_parts.append(f"Current Time: {current_time}")

        return "\n".join(context_parts)

    def query_llm_with_context(self, query, context):
        """Query LLM with comprehensive context"""
        # Prepare messages with context
        messages = [
            {
                'role': 'system',
                'content': f'You are a helpful assistant for a humanoid robot. '
                          f'Context: {context}\n\n'
                          f'Robot capabilities: {self.robot_capabilities}\n\n'
                          f'Use this information to provide helpful and feasible responses. '
                          f'For action requests, suggest appropriate robot actions when possible.'
            },
            {
                'role': 'user',
                'content': query
            }
        ]

        # Call LLM with context (implementation depends on chosen LLM)
        return self.query_llm(messages)
```

## Structured Output for Robot Commands

### Command Extraction and Validation

```python
import json
import re

class CommandExtractionNode(Node):
    def __init__(self):
        super().__init__('command_extraction_node')

        # Service for command extraction
        self.command_service = self.create_service(
            LLMQuery,
            'extract_commands',
            self.handle_command_extraction
        )

        # Publisher for extracted commands
        self.command_pub = self.create_publisher(
            String,  # Would typically be a custom command message
            'robot_commands',
            10
        )

        # Define supported command types
        self.supported_commands = {
            'navigation': ['go to', 'move to', 'navigate to', 'walk to'],
            'manipulation': ['pick up', 'grasp', 'take', 'put down', 'place'],
            'interaction': ['greet', 'wave', 'nod', 'shake hands'],
            'information': ['what is', 'where is', 'find', 'show me']
        }

    def handle_command_extraction(self, request, response):
        """Extract structured commands from LLM response"""
        try:
            # Get LLM response
            llm_response = self.query_llm([{
                'role': 'user',
                'content': f'Extract structured commands from this request: {request.query}. '
                          f'Respond in JSON format with command type and parameters.'
            }])

            # Parse structured response
            structured_commands = self.parse_structured_response(llm_response)

            # Validate commands
            valid_commands = self.validate_commands(structured_commands)

            # Publish valid commands
            for command in valid_commands:
                cmd_msg = String()
                cmd_msg.data = json.dumps(command)
                self.command_pub.publish(cmd_msg)

            response.success = True
            response.response = json.dumps(valid_commands)

        except Exception as e:
            self.get_logger().error(f'Command extraction failed: {e}')
            response.success = False
            response.response = f'Error extracting commands: {e}'

        return response

    def parse_structured_response(self, llm_response):
        """Parse LLM response to extract structured commands"""
        try:
            # Try to parse as JSON first
            return json.loads(llm_response)
        except json.JSONDecodeError:
            # If not JSON, try to extract using regex or other methods
            return self.extract_commands_regex(llm_response)

    def extract_commands_regex(self, text):
        """Extract commands using regex patterns"""
        commands = []

        # Navigation commands
        nav_pattern = r'(?:go to|move to|navigate to|walk to) (.+)'
        nav_matches = re.findall(nav_pattern, text, re.IGNORECASE)
        for match in nav_matches:
            commands.append({
                'type': 'navigation',
                'target': match.strip(),
                'parameters': {}
            })

        # Manipulation commands
        manip_pattern = r'(?:pick up|grasp|take) (.+)'
        manip_matches = re.findall(manip_pattern, text, re.IGNORECASE)
        for match in manip_matches:
            commands.append({
                'type': 'manipulation',
                'target': match.strip(),
                'parameters': {}
            })

        return commands

    def validate_commands(self, commands):
        """Validate extracted commands for robot feasibility"""
        valid_commands = []

        for command in commands:
            if self.is_command_valid(command):
                valid_commands.append(command)

        return valid_commands

    def is_command_valid(self, command):
        """Check if a command is valid for the robot"""
        # Check if command type is supported
        if command.get('type') not in self.supported_commands:
            return False

        # Check if target is specified
        if not command.get('target'):
            return False

        # Additional validation logic can be added here
        return True
```

## Performance Optimization

### Caching and Batching

```python
import functools
import time
from collections import OrderedDict

class OptimizedLLMNode(Node):
    def __init__(self):
        super().__init__('optimized_llm_node')

        # Initialize cache
        self.response_cache = OrderedDict()
        self.cache_size = 50
        self.cache_timeout = 300  # 5 minutes

        # Rate limiting
        self.request_times = []
        self.max_requests_per_minute = 10

    @functools.lru_cache(maxsize=128)
    def cached_llm_query(self, query_hash):
        """Cached LLM query to avoid repeated requests"""
        # This would call the actual LLM with the original query
        # Implementation depends on the specific LLM being used
        pass

    def should_rate_limit(self):
        """Check if request should be rate limited"""
        current_time = time.time()

        # Remove old requests
        self.request_times = [
            req_time for req_time in self.request_times
            if current_time - req_time < 60
        ]

        # Check if we're over the limit
        if len(self.request_times) >= self.max_requests_per_minute:
            return True

        # Add current request
        self.request_times.append(current_time)
        return False

    def query_with_caching(self, query, context=None):
        """Query LLM with caching and rate limiting"""
        # Create cache key
        cache_key = f"{query}_{context or ''}" if context else query

        # Check cache first
        if cache_key in self.response_cache:
            cached_response, timestamp = self.response_cache[cache_key]
            if time.time() - timestamp < self.cache_timeout:
                return cached_response

        # Check rate limit
        if self.should_rate_limit():
            return "Please wait - processing too many requests"

        # Call LLM
        response = self.call_llm(query, context)

        # Cache the response
        self.cache_response(cache_key, response)

        return response

    def cache_response(self, key, response):
        """Cache LLM response"""
        if len(self.response_cache) >= self.cache_size:
            # Remove oldest entry
            self.response_cache.popitem(last=False)

        self.response_cache[key] = (response, time.time())
```

## Safety and Security Considerations

### Content Filtering

```python
class SafeLLMNode(Node):
    def __init__(self):
        super().__init__('safe_llm_node')

        # Initialize safety filters
        self.initialize_safety_filters()

    def initialize_safety_filters(self):
        """Initialize content safety filters"""
        self.blocked_keywords = [
            'harm', 'injure', 'damage', 'unsafe', 'dangerous',
            'break', 'destroy', 'hurt', 'kill', 'harmful'
        ]

        # Context-specific safety rules
        self.safety_rules = [
            # Rule: Don't allow commands that could harm humans
            {
                'pattern': r'(?:hurt|harm|injure|attack|hit|slap|push) (?:me|him|her|them|person|people|human)',
                'response': "I cannot perform actions that might harm humans. Is there something else I can help with?"
            }
        ]

    def process_safe_response(self, llm_response, original_request):
        """Process LLM response for safety"""
        # Check for safety violations
        if self.contains_blocked_content(llm_response):
            return self.get_safe_alternative(original_request)

        # Apply safety rules
        for rule in self.safety_rules:
            if re.search(rule['pattern'], llm_response, re.IGNORECASE):
                return rule['response']

        return llm_response

    def contains_blocked_content(self, text):
        """Check if text contains blocked content"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.blocked_keywords)

    def get_safe_alternative(self, original_request):
        """Get a safe alternative response"""
        safe_response = self.query_llm([{
            'role': 'user',
            'content': f'{original_request} - but respond safely and helpfully instead'
        }])

        return self.process_safe_response(safe_response, original_request)
```

## Integration with Vision and Action Systems

### Multi-Modal Prompt Engineering

```python
class MultiModalLLMNode(Node):
    def __init__(self):
        super().__init__('multi_modal_llm_node')

        # Subscribe to vision and sensor data
        self.vision_sub = self.create_subscription(
            String,
            'vision_description',
            self.vision_callback,
            10
        )

        self.sensors_sub = self.create_subscription(
            String,
            'sensor_fusion',
            self.sensors_callback,
            10
        )

        # Store multi-modal context
        self.vision_data = {}
        self.sensor_data = {}

    def vision_callback(self, msg):
        """Update vision context"""
        try:
            self.vision_data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.vision_data = {'description': msg.data}

    def sensors_callback(self, msg):
        """Update sensor context"""
        try:
            self.sensor_data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.sensor_data = {'data': msg.data}

    def create_multimodal_prompt(self, user_request):
        """Create prompt combining vision, sensor, and language inputs"""
        prompt_parts = []

        # Add user request
        prompt_parts.append(f"User Request: {user_request}")

        # Add vision context
        if self.vision_data:
            prompt_parts.append(f"Visual Context: {self.vision_data}")

        # Add sensor context
        if self.sensor_data:
            prompt_parts.append(f"Sensor Context: {self.sensor_data}")

        # Add robot state
        robot_state = self.get_robot_state()
        prompt_parts.append(f"Robot State: {robot_state}")

        # Add capabilities
        capabilities = self.get_robot_capabilities()
        prompt_parts.append(f"Robot Capabilities: {capabilities}")

        return "\n".join(prompt_parts)

    def get_robot_state(self):
        """Get current robot state"""
        # This would typically come from robot state publisher
        return {
            'battery_level': 85,
            'location': 'kitchen',
            'current_task': 'idle',
            'gripper_status': 'open'
        }
```

## Troubleshooting Common Issues

### API Connection Problems

**Issue**: LLM API calls failing due to connection issues.

**Solutions**:
1. Check API key validity and permissions
2. Verify network connectivity
3. Implement retry logic with exponential backoff
4. Use local fallback models when cloud services are unavailable

### Performance Issues

**Issue**: High latency in LLM responses affecting real-time interaction.

**Solutions**:
1. Use smaller, faster models for real-time responses
2. Implement response caching for common queries
3. Use streaming responses when supported
4. Preload frequently used prompts

### Context Window Limitations

**Issue**: Long conversations exceeding context window limits.

**Solutions**:
1. Implement conversation summarization
2. Use external memory systems
3. Implement sliding window context management
4. Compress context when necessary

## Best Practices

### System Design

- **Modular Architecture**: Keep LLM integration separate for easy replacement
- **Fallback Mechanisms**: Provide alternative responses when LLM fails
- **Error Handling**: Implement comprehensive error handling and logging
- **Performance Monitoring**: Track response times and success rates

### Privacy and Ethics

- **Data Minimization**: Only send necessary information to LLMs
- **Local Processing**: Use local models when privacy is critical
- **Consent**: Obtain user consent for LLM interactions
- **Transparency**: Inform users when LLMs are being used

## Next Steps

Continue to [Cognitive Planning](./cognitive-planning.md) to learn how to use LLMs for high-level reasoning and task planning in humanoid robots.