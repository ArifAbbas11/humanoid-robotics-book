/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  // Book-style sidebar that organizes all content in a logical flow
  bookSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'ros-fundamentals/intro',
        'ros-fundamentals/installation',
        'ros-fundamentals/architecture',
        'ros-fundamentals/nodes-topics-services',
        'ros-fundamentals/python-integration',
        'ros-fundamentals/urdf-basics',
        'ros-fundamentals/mini-project',
        'ros-fundamentals/troubleshooting',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'simulation/intro',
        'simulation/gazebo-setup',
        'simulation/unity-setup',
        'simulation/physics-concepts',
        'simulation/sensors',
        'simulation/sensor-configuration',
        'simulation/mini-project',
        'simulation/troubleshooting',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)',
      items: [
        'ai-navigation/intro',
        'ai-navigation/isaac-setup',
        'ai-navigation/isaac-sim',
        'ai-navigation/isaac-ros',
        'ai-navigation/vslam',
        'ai-navigation/navigation-planning',
        'ai-navigation/mini-project',
        'ai-navigation/troubleshooting',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'vla-integration/intro',
        'vla-integration/voice-recognition',
        'vla-integration/llm-integration',
        'vla-integration/cognitive-planning',
        'vla-integration/multi-modal',
        'vla-integration/voice-to-action',
        'vla-integration/capstone-project',
        'vla-integration/troubleshooting',
      ],
    },
    {
      type: 'category',
      label: 'Capstone Project',
      items: [
        'capstone/intro',
        'capstone/integration-guide',
        'capstone/testing-guide',
        'capstone/troubleshooting',
      ],
    },
  ],
};

export default sidebars;