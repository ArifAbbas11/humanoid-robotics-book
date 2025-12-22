# Lab Instruction Template

This template provides a standard format for lab instructions and practical exercises throughout the book.

## Format

```markdown
# Lab: [Title of the lab exercise]

**Duration**: [Estimated time to complete]

**Learning Objectives**:
- [ ] [Objective 1]
- [ ] [Objective 2]
- [ ] [Objective 3]

**Prerequisites**:
- [List of knowledge, tools, or previous modules required]

**Equipment/Software Required**:
- [List of hardware, software, or other materials needed]

**Instructions**:

### Step 1: [Brief description of first step]
- [Detailed instructions for the first step]
- [Expected results or output]

### Step 2: [Brief description of second step]
- [Detailed instructions for the second step]
- [Expected results or output]

### Step n: [Brief description of final step]
- [Detailed instructions for the final step]
- [Expected results or output]

**Verification**:
- [How to verify that the lab was completed successfully]
- [Expected final state or output]

**Troubleshooting**:
- [Common issues and solutions]
- [Where to look for error messages]

**Reflection Questions**:
1. [Question to help solidify learning]
2. [Question about potential improvements or extensions]

**Next Steps**:
- [How this lab connects to the next topic]
- [Suggestions for further exploration]
```

## Usage Guidelines

1. Always include clear learning objectives that align with the module goals
2. Specify required prerequisites to avoid confusion
3. Break complex tasks into numbered steps
4. Include expected results at each step when possible
5. Provide troubleshooting guidance for common issues
6. End with reflection questions to reinforce learning

## Example

# Lab: Creating Your First ROS 2 Package

**Duration**: 30-45 minutes

**Learning Objectives**:
- [ ] Create a new ROS 2 package using colcon
- [ ] Understand the basic structure of a ROS 2 package
- [ ] Build and source a ROS 2 workspace

**Prerequisites**:
- ROS 2 Humble Hawksbill installed
- Basic command-line knowledge
- Completion of Module 1, Section 1 (Installation)

**Equipment/Software Required**:
- Ubuntu 22.04 with ROS 2 Humble
- Terminal access

**Instructions**:

### Step 1: Create a workspace directory
Create a directory for your ROS 2 workspace:
```bash
mkdir -p ~/ros2_workspace/src
cd ~/ros2_workspace
```
- Expected result: A new directory structure with src subdirectory

### Step 2: Create a new package
Use the ros2 pkg command to create a new package:
```bash
cd src
ros2 pkg create --build-type ament_python my_robot_package
```
- Expected result: A new package directory with standard ROS 2 structure

### Step 3: Build the workspace
Return to the workspace root and build:
```bash
cd ~/ros2_workspace
colcon build
```
- Expected result: Successful build with no errors

### Step 4: Source the workspace
Source the setup file to use the new package:
```bash
source install/setup.bash
```
- Expected result: No output, but the package is now available in the environment

**Verification**:
- Run `ros2 pkg list` and confirm your package appears in the list
- Check that the install directory contains built files

**Troubleshooting**:
- If colcon build fails, ensure ROS 2 is properly sourced with `source /opt/ros/humble/setup.bash`
- If the package doesn't appear in the list, double-check the build command and directory structure

**Reflection Questions**:
1. What are the key components of a ROS 2 package structure?
2. Why is it important to source the setup.bash file after building?

**Next Steps**:
- Learn to add nodes to your new package
- Explore different build types available for ROS 2 packages