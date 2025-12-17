import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/humanoid-robotics-book/__docusaurus/debug',
    component: ComponentCreator('/humanoid-robotics-book/__docusaurus/debug', 'fbb'),
    exact: true
  },
  {
    path: '/humanoid-robotics-book/__docusaurus/debug/config',
    component: ComponentCreator('/humanoid-robotics-book/__docusaurus/debug/config', '58c'),
    exact: true
  },
  {
    path: '/humanoid-robotics-book/__docusaurus/debug/content',
    component: ComponentCreator('/humanoid-robotics-book/__docusaurus/debug/content', '9ef'),
    exact: true
  },
  {
    path: '/humanoid-robotics-book/__docusaurus/debug/globalData',
    component: ComponentCreator('/humanoid-robotics-book/__docusaurus/debug/globalData', '5e0'),
    exact: true
  },
  {
    path: '/humanoid-robotics-book/__docusaurus/debug/metadata',
    component: ComponentCreator('/humanoid-robotics-book/__docusaurus/debug/metadata', 'a6b'),
    exact: true
  },
  {
    path: '/humanoid-robotics-book/__docusaurus/debug/registry',
    component: ComponentCreator('/humanoid-robotics-book/__docusaurus/debug/registry', '969'),
    exact: true
  },
  {
    path: '/humanoid-robotics-book/__docusaurus/debug/routes',
    component: ComponentCreator('/humanoid-robotics-book/__docusaurus/debug/routes', '919'),
    exact: true
  },
  {
    path: '/humanoid-robotics-book/',
    component: ComponentCreator('/humanoid-robotics-book/', 'f8c'),
    exact: true
  },
  {
    path: '/humanoid-robotics-book/',
    component: ComponentCreator('/humanoid-robotics-book/', 'f76'),
    routes: [
      {
        path: '/humanoid-robotics-book/',
        component: ComponentCreator('/humanoid-robotics-book/', '66c'),
        routes: [
          {
            path: '/humanoid-robotics-book/',
            component: ComponentCreator('/humanoid-robotics-book/', 'ead'),
            routes: [
              {
                path: '/humanoid-robotics-book/ai-navigation/control-systems',
                component: ComponentCreator('/humanoid-robotics-book/ai-navigation/control-systems', '560'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/ai-navigation/intro',
                component: ComponentCreator('/humanoid-robotics-book/ai-navigation/intro', '46e'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ai-navigation/introduction',
                component: ComponentCreator('/humanoid-robotics-book/ai-navigation/introduction', '447'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/ai-navigation/isaac-ros',
                component: ComponentCreator('/humanoid-robotics-book/ai-navigation/isaac-ros', '833'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ai-navigation/isaac-setup',
                component: ComponentCreator('/humanoid-robotics-book/ai-navigation/isaac-setup', '26d'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ai-navigation/isaac-sim',
                component: ComponentCreator('/humanoid-robotics-book/ai-navigation/isaac-sim', '903'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ai-navigation/localization',
                component: ComponentCreator('/humanoid-robotics-book/ai-navigation/localization', '92d'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/ai-navigation/mapping',
                component: ComponentCreator('/humanoid-robotics-book/ai-navigation/mapping', '8f9'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/ai-navigation/mini-project',
                component: ComponentCreator('/humanoid-robotics-book/ai-navigation/mini-project', '72e'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ai-navigation/navigation-planning',
                component: ComponentCreator('/humanoid-robotics-book/ai-navigation/navigation-planning', 'a72'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ai-navigation/path-planning',
                component: ComponentCreator('/humanoid-robotics-book/ai-navigation/path-planning', '32a'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/ai-navigation/troubleshooting',
                component: ComponentCreator('/humanoid-robotics-book/ai-navigation/troubleshooting', 'f13'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ai-navigation/vslam',
                component: ComponentCreator('/humanoid-robotics-book/ai-navigation/vslam', '80e'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/assets/ATTRIBUTION',
                component: ComponentCreator('/humanoid-robotics-book/assets/ATTRIBUTION', 'ffb'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/capstone/integration-guide',
                component: ComponentCreator('/humanoid-robotics-book/capstone/integration-guide', '71f'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/capstone/intro',
                component: ComponentCreator('/humanoid-robotics-book/capstone/intro', '7f1'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/capstone/testing-guide',
                component: ComponentCreator('/humanoid-robotics-book/capstone/testing-guide', 'a36'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/capstone/troubleshooting',
                component: ComponentCreator('/humanoid-robotics-book/capstone/troubleshooting', 'e9b'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/intro',
                component: ComponentCreator('/humanoid-robotics-book/intro', 'df1'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ros-fundamentals/architecture',
                component: ComponentCreator('/humanoid-robotics-book/ros-fundamentals/architecture', '30e'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ros-fundamentals/installation',
                component: ComponentCreator('/humanoid-robotics-book/ros-fundamentals/installation', '310'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ros-fundamentals/intro',
                component: ComponentCreator('/humanoid-robotics-book/ros-fundamentals/intro', 'b48'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ros-fundamentals/mini-project',
                component: ComponentCreator('/humanoid-robotics-book/ros-fundamentals/mini-project', 'af4'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ros-fundamentals/nodes-topics-services',
                component: ComponentCreator('/humanoid-robotics-book/ros-fundamentals/nodes-topics-services', 'cb0'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ros-fundamentals/python-integration',
                component: ComponentCreator('/humanoid-robotics-book/ros-fundamentals/python-integration', 'b37'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ros-fundamentals/troubleshooting',
                component: ComponentCreator('/humanoid-robotics-book/ros-fundamentals/troubleshooting', '132'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ros-fundamentals/urdf-basics',
                component: ComponentCreator('/humanoid-robotics-book/ros-fundamentals/urdf-basics', '228'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/simulation/gazebo-setup',
                component: ComponentCreator('/humanoid-robotics-book/simulation/gazebo-setup', '7b0'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/simulation/intro',
                component: ComponentCreator('/humanoid-robotics-book/simulation/intro', '74b'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/simulation/mini-project',
                component: ComponentCreator('/humanoid-robotics-book/simulation/mini-project', '30a'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/simulation/physics-concepts',
                component: ComponentCreator('/humanoid-robotics-book/simulation/physics-concepts', '3ce'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/simulation/sensor-configuration',
                component: ComponentCreator('/humanoid-robotics-book/simulation/sensor-configuration', '52d'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/simulation/sensors',
                component: ComponentCreator('/humanoid-robotics-book/simulation/sensors', '4f6'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/simulation/troubleshooting',
                component: ComponentCreator('/humanoid-robotics-book/simulation/troubleshooting', 'b37'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/simulation/unity-setup',
                component: ComponentCreator('/humanoid-robotics-book/simulation/unity-setup', 'f24'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/STYLE_GUIDE',
                component: ComponentCreator('/humanoid-robotics-book/STYLE_GUIDE', 'dff'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/templates/code-example-template',
                component: ComponentCreator('/humanoid-robotics-book/templates/code-example-template', '6f2'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/templates/lab-instruction-template',
                component: ComponentCreator('/humanoid-robotics-book/templates/lab-instruction-template', 'b96'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/vla-integration/action-planning',
                component: ComponentCreator('/humanoid-robotics-book/vla-integration/action-planning', '46a'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/vla-integration/capstone-project',
                component: ComponentCreator('/humanoid-robotics-book/vla-integration/capstone-project', '8cc'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/vla-integration/cognitive-planning',
                component: ComponentCreator('/humanoid-robotics-book/vla-integration/cognitive-planning', 'b83'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/vla-integration/integration-challenges',
                component: ComponentCreator('/humanoid-robotics-book/vla-integration/integration-challenges', '708'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/vla-integration/intro',
                component: ComponentCreator('/humanoid-robotics-book/vla-integration/intro', 'f08'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/vla-integration/introduction',
                component: ComponentCreator('/humanoid-robotics-book/vla-integration/introduction', '2af'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/vla-integration/language-understanding',
                component: ComponentCreator('/humanoid-robotics-book/vla-integration/language-understanding', '63d'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/vla-integration/llm-integration',
                component: ComponentCreator('/humanoid-robotics-book/vla-integration/llm-integration', 'be7'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/vla-integration/mini-project',
                component: ComponentCreator('/humanoid-robotics-book/vla-integration/mini-project', '5c5'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/vla-integration/multi-modal',
                component: ComponentCreator('/humanoid-robotics-book/vla-integration/multi-modal', '3e1'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/vla-integration/troubleshooting',
                component: ComponentCreator('/humanoid-robotics-book/vla-integration/troubleshooting', '91b'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/vla-integration/vision-systems',
                component: ComponentCreator('/humanoid-robotics-book/vla-integration/vision-systems', 'b0b'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/vla-integration/voice-recognition',
                component: ComponentCreator('/humanoid-robotics-book/vla-integration/voice-recognition', 'a70'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/vla-integration/voice-to-action',
                component: ComponentCreator('/humanoid-robotics-book/vla-integration/voice-to-action', '79d'),
                exact: true,
                sidebar: "bookSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
