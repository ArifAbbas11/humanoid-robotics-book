import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/humanoid-robotics-book/__docusaurus/debug',
    component: ComponentCreator('/humanoid-robotics-book/__docusaurus/debug', '86a'),
    exact: true
  },
  {
    path: '/humanoid-robotics-book/__docusaurus/debug/config',
    component: ComponentCreator('/humanoid-robotics-book/__docusaurus/debug/config', '6a7'),
    exact: true
  },
  {
    path: '/humanoid-robotics-book/__docusaurus/debug/content',
    component: ComponentCreator('/humanoid-robotics-book/__docusaurus/debug/content', 'c2e'),
    exact: true
  },
  {
    path: '/humanoid-robotics-book/__docusaurus/debug/globalData',
    component: ComponentCreator('/humanoid-robotics-book/__docusaurus/debug/globalData', '73c'),
    exact: true
  },
  {
    path: '/humanoid-robotics-book/__docusaurus/debug/metadata',
    component: ComponentCreator('/humanoid-robotics-book/__docusaurus/debug/metadata', '8ef'),
    exact: true
  },
  {
    path: '/humanoid-robotics-book/__docusaurus/debug/registry',
    component: ComponentCreator('/humanoid-robotics-book/__docusaurus/debug/registry', '827'),
    exact: true
  },
  {
    path: '/humanoid-robotics-book/__docusaurus/debug/routes',
    component: ComponentCreator('/humanoid-robotics-book/__docusaurus/debug/routes', 'd39'),
    exact: true
  },
  {
    path: '/humanoid-robotics-book/',
    component: ComponentCreator('/humanoid-robotics-book/', 'cb0'),
    exact: true
  },
  {
    path: '/humanoid-robotics-book/',
    component: ComponentCreator('/humanoid-robotics-book/', '78a'),
    routes: [
      {
        path: '/humanoid-robotics-book/',
        component: ComponentCreator('/humanoid-robotics-book/', 'c49'),
        routes: [
          {
            path: '/humanoid-robotics-book/',
            component: ComponentCreator('/humanoid-robotics-book/', '586'),
            routes: [
              {
                path: '/humanoid-robotics-book/ai-navigation/control-systems',
                component: ComponentCreator('/humanoid-robotics-book/ai-navigation/control-systems', 'f10'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/ai-navigation/intro',
                component: ComponentCreator('/humanoid-robotics-book/ai-navigation/intro', '645'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ai-navigation/introduction',
                component: ComponentCreator('/humanoid-robotics-book/ai-navigation/introduction', '5a0'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/ai-navigation/isaac-ros',
                component: ComponentCreator('/humanoid-robotics-book/ai-navigation/isaac-ros', '5e1'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ai-navigation/isaac-setup',
                component: ComponentCreator('/humanoid-robotics-book/ai-navigation/isaac-setup', '8f1'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ai-navigation/isaac-sim',
                component: ComponentCreator('/humanoid-robotics-book/ai-navigation/isaac-sim', 'efb'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ai-navigation/localization',
                component: ComponentCreator('/humanoid-robotics-book/ai-navigation/localization', '6ec'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/ai-navigation/mapping',
                component: ComponentCreator('/humanoid-robotics-book/ai-navigation/mapping', '55f'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/ai-navigation/mini-project',
                component: ComponentCreator('/humanoid-robotics-book/ai-navigation/mini-project', 'b0b'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ai-navigation/navigation-planning',
                component: ComponentCreator('/humanoid-robotics-book/ai-navigation/navigation-planning', '646'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ai-navigation/path-planning',
                component: ComponentCreator('/humanoid-robotics-book/ai-navigation/path-planning', '3ac'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/ai-navigation/troubleshooting',
                component: ComponentCreator('/humanoid-robotics-book/ai-navigation/troubleshooting', 'aac'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ai-navigation/vslam',
                component: ComponentCreator('/humanoid-robotics-book/ai-navigation/vslam', 'ec2'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/assets/ATTRIBUTION',
                component: ComponentCreator('/humanoid-robotics-book/assets/ATTRIBUTION', 'c26'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/capstone/integration-guide',
                component: ComponentCreator('/humanoid-robotics-book/capstone/integration-guide', '6a2'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/capstone/intro',
                component: ComponentCreator('/humanoid-robotics-book/capstone/intro', '637'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/capstone/testing-guide',
                component: ComponentCreator('/humanoid-robotics-book/capstone/testing-guide', '982'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/capstone/troubleshooting',
                component: ComponentCreator('/humanoid-robotics-book/capstone/troubleshooting', 'ddf'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/intro',
                component: ComponentCreator('/humanoid-robotics-book/intro', 'cff'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ros-fundamentals/architecture',
                component: ComponentCreator('/humanoid-robotics-book/ros-fundamentals/architecture', '0cc'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ros-fundamentals/installation',
                component: ComponentCreator('/humanoid-robotics-book/ros-fundamentals/installation', 'a5d'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ros-fundamentals/intro',
                component: ComponentCreator('/humanoid-robotics-book/ros-fundamentals/intro', 'fe7'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ros-fundamentals/mini-project',
                component: ComponentCreator('/humanoid-robotics-book/ros-fundamentals/mini-project', '84f'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ros-fundamentals/nodes-topics-services',
                component: ComponentCreator('/humanoid-robotics-book/ros-fundamentals/nodes-topics-services', '1ea'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ros-fundamentals/python-integration',
                component: ComponentCreator('/humanoid-robotics-book/ros-fundamentals/python-integration', '5e8'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ros-fundamentals/troubleshooting',
                component: ComponentCreator('/humanoid-robotics-book/ros-fundamentals/troubleshooting', '87c'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/ros-fundamentals/urdf-basics',
                component: ComponentCreator('/humanoid-robotics-book/ros-fundamentals/urdf-basics', '0e5'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/simulation/gazebo-setup',
                component: ComponentCreator('/humanoid-robotics-book/simulation/gazebo-setup', 'ab1'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/simulation/intro',
                component: ComponentCreator('/humanoid-robotics-book/simulation/intro', '076'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/simulation/mini-project',
                component: ComponentCreator('/humanoid-robotics-book/simulation/mini-project', 'b91'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/simulation/physics-concepts',
                component: ComponentCreator('/humanoid-robotics-book/simulation/physics-concepts', '53f'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/simulation/sensor-configuration',
                component: ComponentCreator('/humanoid-robotics-book/simulation/sensor-configuration', 'c5b'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/simulation/sensors',
                component: ComponentCreator('/humanoid-robotics-book/simulation/sensors', '31a'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/simulation/troubleshooting',
                component: ComponentCreator('/humanoid-robotics-book/simulation/troubleshooting', '23e'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/simulation/unity-setup',
                component: ComponentCreator('/humanoid-robotics-book/simulation/unity-setup', 'a3c'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/STYLE_GUIDE',
                component: ComponentCreator('/humanoid-robotics-book/STYLE_GUIDE', '8e9'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/templates/code-example-template',
                component: ComponentCreator('/humanoid-robotics-book/templates/code-example-template', '991'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/templates/lab-instruction-template',
                component: ComponentCreator('/humanoid-robotics-book/templates/lab-instruction-template', 'a91'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/vla-integration/action-planning',
                component: ComponentCreator('/humanoid-robotics-book/vla-integration/action-planning', '248'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/vla-integration/capstone-project',
                component: ComponentCreator('/humanoid-robotics-book/vla-integration/capstone-project', 'acf'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/vla-integration/cognitive-planning',
                component: ComponentCreator('/humanoid-robotics-book/vla-integration/cognitive-planning', '05f'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/vla-integration/integration-challenges',
                component: ComponentCreator('/humanoid-robotics-book/vla-integration/integration-challenges', 'ac3'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/vla-integration/intro',
                component: ComponentCreator('/humanoid-robotics-book/vla-integration/intro', '085'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/vla-integration/introduction',
                component: ComponentCreator('/humanoid-robotics-book/vla-integration/introduction', 'bab'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/vla-integration/language-understanding',
                component: ComponentCreator('/humanoid-robotics-book/vla-integration/language-understanding', 'ec5'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/vla-integration/llm-integration',
                component: ComponentCreator('/humanoid-robotics-book/vla-integration/llm-integration', '320'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/vla-integration/mini-project',
                component: ComponentCreator('/humanoid-robotics-book/vla-integration/mini-project', '529'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/vla-integration/multi-modal',
                component: ComponentCreator('/humanoid-robotics-book/vla-integration/multi-modal', '52a'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/vla-integration/troubleshooting',
                component: ComponentCreator('/humanoid-robotics-book/vla-integration/troubleshooting', '20a'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/vla-integration/vision-systems',
                component: ComponentCreator('/humanoid-robotics-book/vla-integration/vision-systems', 'b34'),
                exact: true
              },
              {
                path: '/humanoid-robotics-book/vla-integration/voice-recognition',
                component: ComponentCreator('/humanoid-robotics-book/vla-integration/voice-recognition', '6a0'),
                exact: true,
                sidebar: "bookSidebar"
              },
              {
                path: '/humanoid-robotics-book/vla-integration/voice-to-action',
                component: ComponentCreator('/humanoid-robotics-book/vla-integration/voice-to-action', '347'),
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
