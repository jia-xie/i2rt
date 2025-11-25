# MuJoCo End Effector Control Example

Interactive simulator for controlling robot arm end effector pose using keyboard commands.

## Quick Start

```bash
python examples/mujoco_end_effector_control/mujoco_end_effector_control.py
```

## What It Does

This example provides an interactive MuJoCo simulator that allows you to:
- Control the robot arm's end effector position and orientation using keyboard
- Visualize the robot in real-time with MuJoCo viewer
- Use inverse kinematics to automatically compute joint angles for desired poses

## Controls

### Position Control
- `w/s` : Move forward/backward (Y axis)
- `a/d` : Move left/right (X axis)
- `t/g` : Move up/down (Z axis)

### Orientation Control
- `i/k` : Rotate around X axis (roll)
- `j/l` : Rotate around Y axis (pitch)
- `u/o` : Rotate around Z axis (yaw)

### Other Controls
- `r` : Reset to home position
- `h` : Show help message
- `q` : Quit

## Options

```bash
--xml_path PATH          # Path to MuJoCo XML model file (default: yam_no_gripper.xml)
--site_name NAME         # Name of the end effector site (default: grasp_site)
--step_size_pos FLOAT    # Position step size in meters (default: 0.01)
--step_size_rot FLOAT    # Rotation step size in radians (default: 0.05)
```

## Example Usage

```bash
# Use default YAM robot without gripper
python examples/mujoco_end_effector_control/mujoco_end_effector_control.py

# Use custom model and step sizes
python examples/mujoco_end_effector_control/mujoco_end_effector_control.py \
    --xml_path i2rt/robot_models/yam/yam.xml \
    --site_name grasp_site \
    --step_size_pos 0.02 \
    --step_size_rot 0.1
```

## Notes

- Make sure the terminal window has focus for keyboard input to work
- The simulator uses inverse kinematics to compute joint angles
- If IK fails to converge, a warning will be displayed
- The viewer shows the robot model with the end effector site highlighted

