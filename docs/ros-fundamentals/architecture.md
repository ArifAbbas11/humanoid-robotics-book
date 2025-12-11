# ROS 2 Architecture Concepts

## Overview

ROS 2 (Robot Operating System 2) is designed to be a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

## Core Concepts

### Nodes
A node is an executable that uses ROS 2 to communicate with other nodes. Nodes can publish or subscribe to messages, provide or use services, and so on.

### Packages
A package is the main unit of organization in ROS 2. It contains libraries, executables, configuration files, and other resources needed for the functionality it provides.

### Topics and Messages
Topics are named buses over which nodes exchange messages. Messages are the data exchanged between nodes.

## ROS 2 Middleware

ROS 2 uses DDS (Data Distribution Service) as its middleware, which provides the underlying communication layer.

## Next Steps

Continue to [Nodes, Topics, and Services](./nodes-topics-services.md) to learn about communication patterns.