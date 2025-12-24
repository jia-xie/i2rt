"""LeCARM teleoperation of YAM arm - Physical LeCARM controls physical YAM.

This script uses the physical LeCARM robot as a leader to teleoperate
the physical YAM arm. The physical LeCARM is put in gravity compensation
mode so it can be moved manually, and YAM follows its joint positions.

The script has two modes:
    - slow_sync: When joints are too far apart (> threshold), YAM gradually
      syncs up to avoid sudden jumps
    - following: When joints are within threshold, YAM directly follows
      LeCARM's joint positions

Usage:
    uv run python scripts/lecarm_teleop_yam.py
    
    Options:
        --lecarm-can-interface INTERFACE    CAN interface for LeCARM (default: can0)
        --yam-can-interface INTERFACE       CAN interface for YAM (default: can0)
        --rate RATE                         Control loop rate in Hz (default: 200)
        --kp KP                             Position gain for YAM PD control (default: 80.0)
        --kd KD                             Velocity damping for YAM PD control (default: 5.0)
        --slow-sync-threshold THRESHOLD     Joint error (rad) to switch to slow_sync mode (default: 0.5)
        --following-threshold THRESHOLD     Joint error (rad) to switch to following mode (default: 0.2)
        --max-sync-speed SPEED              Max joint speed for slow_sync mode in rad/s (default: 0.5)
        --max-following-speed SPEED         Max joint speed for following mode in rad/s (default: 2.0)
        --enable-visualizer                 Enable MuJoCo visualization of both robots
"""

import argparse
import os
import threading
import time

import numpy as np
import yaml

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.lecarm import LeCARMRobot, LECARM_XML_PATH
from i2rt.robots.utils import GripperType

# Get paths
I2RT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LECARM_CONFIG_PATH = os.path.join(I2RT_ROOT, "i2rt", "robot_models", "lecarm", "lecarm_config.yaml")


def load_lecarm_config(config_path: str) -> dict:
    """Load LeCARM configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"LeCARM config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def apply_joint_mapping(
    lecarm_joint_pos: np.ndarray, joint_mapping: list
) -> np.ndarray:
    """
    Apply joint mapping from LeCARM to YAM.
    
    Args:
        lecarm_joint_pos: Joint positions from LeCARM (6 elements)
        joint_mapping: List of mapping dicts with 'target_joint' and 'direction'
    
    Returns:
        Mapped joint positions for YAM (6 elements)
    """
    yam_joint_pos = np.zeros(6)
    
    for lecarm_idx, mapping in enumerate(joint_mapping):
        yam_idx = mapping["target_joint"]
        direction = mapping["direction"]
        
        if 0 <= yam_idx < 6:
            yam_joint_pos[yam_idx] = lecarm_joint_pos[lecarm_idx] * direction
    
    return yam_joint_pos


def main():
    parser = argparse.ArgumentParser(
        description="Physical LeCARM teleoperation of physical YAM arm"
    )
    parser.add_argument(
        "--lecarm-config",
        type=str,
        default=LECARM_CONFIG_PATH,
        help=f"Path to LeCARM config YAML file (default: {LECARM_CONFIG_PATH})",
    )
    parser.add_argument(
        "--lecarm-can-interface",
        type=str,
        default=None,
        help="CAN interface for LeCARM (overrides config file)",
    )
    parser.add_argument(
        "--motor-ids",
        type=int,
        nargs=6,
        default=None,
        help="Motor CAN IDs for LeCARM joints 1-6 (overrides config file)",
    )
    parser.add_argument(
        "--feedback-ids",
        type=int,
        nargs=6,
        default=None,
        help="Feedback IDs for LeCARM (overrides config file)",
    )
    parser.add_argument(
        "--bitrate",
        type=int,
        default=None,
        help="CAN bus bitrate for LeCARM (overrides config file)",
    )
    parser.add_argument(
        "--yam-can-interface",
        type=str,
        default="can0",
        help="CAN interface for YAM (default: can0)",
    )
    parser.add_argument(
        "--yam-gripper-type",
        type=str,
        default="no_gripper",
        choices=["crank_4310", "linear_3507", "linear_4310", "no_gripper"],
        help="YAM gripper type (default: no_gripper)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=200.0,
        help="Control loop rate in Hz (default: 200)",
    )
    parser.add_argument(
        "--kp",
        type=float,
        nargs="+",
        default=None,
        help="Position gains for YAM PD control (6 values, or single value for all joints). "
        "Default: [80, 80, 80, 40, 10, 10] (per-joint defaults from get_yam_robot)",
    )
    parser.add_argument(
        "--kd",
        type=float,
        nargs="+",
        default=None,
        help="Velocity damping for YAM PD control (6 values, or single value for all joints). "
        "Default: [5, 5, 5, 1.5, 1.5, 1.5] (per-joint defaults from get_yam_robot)",
    )
    parser.add_argument(
        "--gravity-coeffs",
        type=float,
        nargs=6,
        default=None,
        help="Gravity compensation coefficients for LeCARM joints 1-6 "
        "(overrides config file)",
    )
    parser.add_argument(
        "--enable-visualizer",
        action="store_true",
        help="Enable MuJoCo visualization of both robots",
    )
    parser.add_argument(
        "--slow-sync-threshold",
        type=float,
        default=0.5,
        help="Joint error (rad) to switch from following to slow_sync mode (default: 0.5)",
    )
    parser.add_argument(
        "--following-threshold",
        type=float,
        default=0.2,
        help="Joint error (rad) to switch from slow_sync to following mode (default: 0.2)",
    )
    parser.add_argument(
        "--max-sync-speed",
        type=float,
        default=0.5,
        help="Maximum joint speed for slow_sync mode in rad/s (default: 0.5)",
    )
    parser.add_argument(
        "--max-following-speed",
        type=float,
        default=2.0,
        help="Maximum joint speed for following mode in rad/s (default: 2.0)",
    )
    
    args = parser.parse_args()
    
    # Load configuration from YAML file
    print(f"Loading LeCARM configuration from: {args.lecarm_config}")
    try:
        config = load_lecarm_config(args.lecarm_config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Using default configuration values")
        config = {}
    
    # Get configuration values (command line args override config file)
    lecarm_can_interface = args.lecarm_can_interface or config.get("can_interface", "can0")
    motor_ids = args.motor_ids or config.get("motor_ids", [1, 2, 3, 4, 5, 6])
    feedback_ids = args.feedback_ids or config.get("feedback_ids", None)
    bitrate = args.bitrate or config.get("bitrate", 1000000)
    gravity_coeffs = args.gravity_coeffs or config.get("gravity_coefficients", [1.0, 1.0, 0.8, 2.35, 1.0, 1.0])
    joint_mapping = config.get("joint_mapping", [
        {"target_joint": 0, "direction": 1},
        {"target_joint": 1, "direction": 1},
        {"target_joint": 2, "direction": 1},
        {"target_joint": 3, "direction": 1},
        {"target_joint": 4, "direction": 1},
        {"target_joint": 5, "direction": 1},
    ])
    
    print("=" * 60)
    print("LeCARM Teleoperation of YAM Arm")
    print("Physical LeCARM -> Physical YAM")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  LeCARM CAN interface: {lecarm_can_interface}")
    print(f"  YAM CAN interface: {args.yam_can_interface}")
    print(f"  LeCARM Motor IDs: {motor_ids}")
    print(f"  Gravity coefficients: {gravity_coeffs}")
    print(f"  Joint mapping directions: {[m['direction'] for m in joint_mapping]}")
    
    # Connect to physical LeCARM robot
    print(f"\nConnecting to LeCARM on {lecarm_can_interface}...")
    lecarm_robot = LeCARMRobot(
        can_interface=lecarm_can_interface,
        motor_ids=motor_ids,
        feedback_ids=feedback_ids,
        bitrate=bitrate,
        gravity_coefficients=gravity_coeffs,
    )
    
    if not lecarm_robot.connect():
        print("Failed to connect to LeCARM. Exiting.")
        return
    
    print("✓ LeCARM connected successfully")
    
    # Connect to physical YAM robot
    print(f"\nConnecting to YAM on {args.yam_can_interface}...")
    yam_gripper_type = GripperType.from_string_name(args.yam_gripper_type)
    yam_robot = get_yam_robot(
        channel=args.yam_can_interface,
        gripper_type=yam_gripper_type,
        zero_gravity_mode=False,  # We want to control it actively
    )
    
    # Default YAM PD gains (from get_yam_robot defaults)
    default_yam_kp = np.array([80, 80, 80, 40, 10, 10])
    default_yam_kd = np.array([5, 5, 5, 1.5, 1.5, 1.5])
    
    # Handle PD gain arguments
    if args.kp is not None:
        if len(args.kp) == 1:
            # Single value - apply to all joints
            yam_kp = np.array([args.kp[0]] * 6)
        elif len(args.kp) == 6:
            # Per-joint gains
            yam_kp = np.array(args.kp)
        else:
            raise ValueError(f"kp must be 1 or 6 values, got {len(args.kp)}")
    else:
        yam_kp = default_yam_kp
    
    if args.kd is not None:
        if len(args.kd) == 1:
            # Single value - apply to all joints
            yam_kd = np.array([args.kd[0]] * 6)
        elif len(args.kd) == 6:
            # Per-joint gains
            yam_kd = np.array(args.kd)
        else:
            raise ValueError(f"kd must be 1 or 6 values, got {len(args.kd)}")
    else:
        yam_kd = default_yam_kd
    
    # Set PD gains for YAM (only for arm joints, preserve gripper gains if present)
    current_kp = yam_robot._kp.copy()
    current_kd = yam_robot._kd.copy()
    # Update first 6 joints (arm), preserve gripper gains
    current_kp[:6] = yam_kp
    current_kd[:6] = yam_kd
    yam_robot.update_kp_kd(current_kp, current_kd)
    
    print("✓ YAM connected successfully")
    print(f"  YAM PD gains: kp={yam_kp.tolist()}, kd={yam_kd.tolist()} (arm joints only)")
    
    # Setup MuJoCo visualization if requested
    yam_model = None
    yam_data = None
    lecarm_model = None
    lecarm_data = None
    if args.enable_visualizer:
        import mujoco
        import mujoco.viewer
        
        yam_xml_path = yam_gripper_type.get_xml_path()
        print(f"\nLoading MuJoCo models for visualization...")
        yam_model = mujoco.MjModel.from_xml_path(yam_xml_path)
        yam_data = mujoco.MjData(yam_model)
        lecarm_model = mujoco.MjModel.from_xml_path(LECARM_XML_PATH)
        lecarm_data = mujoco.MjData(lecarm_model)
        print("✓ MuJoCo models loaded")
    
    period = 1.0 / args.rate
    max_sync_step = args.max_sync_speed * period  # Maximum change per control cycle for slow_sync
    max_following_step = args.max_following_speed * period  # Maximum change per control cycle for following
    print(f"\nControl rate: {args.rate} Hz")
    print(f"Slow sync threshold: {args.slow_sync_threshold} rad (switch to slow_sync)")
    print(f"Following threshold: {args.following_threshold} rad (switch to following)")
    print(f"Max sync speed: {args.max_sync_speed} rad/s (max step: {max_sync_step:.4f} rad/cycle)")
    print(f"Max following speed: {args.max_following_speed} rad/s (max step: {max_following_step:.4f} rad/cycle)")
    print("LeCARM is in gravity compensation mode - you can move it manually")
    print("YAM will follow LeCARM's joint positions")
    print("  - slow_sync mode: when joints are far apart (> {:.2f} rad)".format(args.slow_sync_threshold))
    print("  - following mode: when joints are close (< {:.2f} rad)".format(args.following_threshold))
    print("Press Ctrl+C to stop\n")
    
    # Send initial zero commands to LeCARM and enable gravity compensation
    print("Initializing LeCARM motors...")
    for motor in lecarm_robot.motors:
        if motor is not None:
            motor.send_cmd(
                target_position=0.0,
                target_velocity=0.0,
                stiffness=0.0,
                damping=0.0,
                feedforward_torque=0.0,
                control_mode="MIT",
            )
            time.sleep(0.005)
    
    # Setup MuJoCo visualization threads if enabled
    visualizer_threads = []
    viewer_shutdown = threading.Event()
    if args.enable_visualizer:
        import mujoco.viewer
        
        def yam_viewer_thread():
            """Separate thread for YAM viewer."""
            try:
                with mujoco.viewer.launch_passive(
                    model=yam_model,
                    data=yam_data,
                    show_left_ui=False,
                    show_right_ui=False,
                ) as viewer:
                    mujoco.mjv_defaultFreeCamera(yam_model, viewer.cam)
                    viewer.cam.lookat[:] = [0.3, 0, 0.2]
                    viewer.cam.distance = 1.5
                    
                    while viewer.is_running() and not viewer_shutdown.is_set():
                        mujoco.mj_kinematics(yam_model, yam_data)
                        viewer.sync()
                        time.sleep(0.01)
            except Exception:
                # Suppress errors during shutdown
                pass
        
        def lecarm_viewer_thread():
            """Separate thread for LeCARM viewer."""
            try:
                with mujoco.viewer.launch_passive(
                    model=lecarm_model,
                    data=lecarm_data,
                    show_left_ui=False,
                    show_right_ui=False,
                ) as viewer:
                    mujoco.mjv_defaultFreeCamera(lecarm_model, viewer.cam)
                    viewer.cam.lookat[:] = [0.3, 0, 0.2]
                    viewer.cam.distance = 1.5
                    
                    while viewer.is_running() and not viewer_shutdown.is_set():
                        mujoco.mj_kinematics(lecarm_model, lecarm_data)
                        viewer.sync()
                        time.sleep(0.01)
            except Exception:
                # Suppress errors during shutdown
                pass
        
        print("Starting MuJoCo viewers...")
        yam_thread = threading.Thread(target=yam_viewer_thread, daemon=True)
        lecarm_thread = threading.Thread(target=lecarm_viewer_thread, daemon=True)
        yam_thread.start()
        lecarm_thread.start()
        visualizer_threads = [yam_thread, lecarm_thread]
        time.sleep(0.5)  # Give viewers time to start
    
    try:
        # Initialize mode tracking and rate-limited target
        current_mode = "slow_sync"  # Start in slow_sync mode
        yam_full_current_pos = yam_robot.get_joint_pos()
        rate_limited_target = yam_full_current_pos[:6].copy()  # Initialize rate-limited target to current position
        print(f"Starting in slow_sync mode (will switch to following when error < {args.following_threshold:.3f} rad)\n")
        
        # Main control loop
        while True:
            loop_start = time.time()
            
            # Read LeCARM joint positions from physical robot
            lecarm_joint_pos = lecarm_robot.get_joint_pos()
            
            if len(lecarm_joint_pos) != 6:
                print("Warning: Expected 6 joint positions, got", len(lecarm_joint_pos))
                time.sleep(period)
                continue
            
            # Apply gravity compensation to LeCARM (so it can be moved easily)
            lecarm_robot.send_gravity_compensation_only()
            
            # Map LeCARM positions to YAM using joint mapping from config
            # This is the final target from the leader
            yam_final_target = apply_joint_mapping(lecarm_joint_pos, joint_mapping)
            
            # Get current YAM position (full state, may include gripper)
            yam_full_current_pos = yam_robot.get_joint_pos()
            yam_current_pos = yam_full_current_pos[:6]  # Arm joints only
            
            # Calculate joint error between current position and final target
            # This determines which mode we should be in
            joint_error = np.abs(yam_final_target - yam_current_pos)
            max_error = np.max(joint_error)
            
            # Determine mode based on error with hysteresis
            # Use slow_sync_threshold to switch TO slow_sync (when error gets large)
            # Use following_threshold to switch TO following (when error gets small)
            if current_mode == "slow_sync":
                # In slow_sync mode, switch to following when error is small enough
                if max_error <= args.following_threshold:
                    current_mode = "following"
                    print(f"Switched to following mode (max error: {max_error:.3f} rad)")
                    # When switching to following mode, directly set rate-limited target to final target
                    rate_limited_target = yam_final_target.copy()
            else:  # current_mode == "following"
                # In following mode, switch to slow_sync when error is too large
                if max_error > args.slow_sync_threshold:
                    current_mode = "slow_sync"
                    print(f"Switched to slow_sync mode (max error: {max_error:.3f} rad)")
            
            # Apply rate limiting based on current mode
            if current_mode == "slow_sync":
                # Rate limit the target towards the final target (slow speed)
                # Calculate desired change from current rate-limited target to final target
                desired_change = yam_final_target - rate_limited_target
                # Limit the change to max_sync_step per control cycle
                change_magnitude = np.abs(desired_change)
                # Scale down if any joint exceeds max speed
                scale_factor = np.ones(6)
                exceeds_limit = change_magnitude > max_sync_step
                if np.any(exceeds_limit):
                    # For joints that exceed limit, scale to max_sync_step
                    scale_factor[exceeds_limit] = max_sync_step / change_magnitude[exceeds_limit]
                
                # Update rate-limited target (not based on current position, but on previous rate-limited target)
                rate_limited_target = rate_limited_target + desired_change * scale_factor
            else:  # following mode
                # Rate limit the target towards the final target (fast speed)
                # Calculate desired change from current rate-limited target to final target
                desired_change = yam_final_target - rate_limited_target
                # Limit the change to max_following_step per control cycle
                change_magnitude = np.abs(desired_change)
                # Scale down if any joint exceeds max speed
                scale_factor = np.ones(6)
                exceeds_limit = change_magnitude > max_following_step
                if np.any(exceeds_limit):
                    # For joints that exceed limit, scale to max_following_step
                    scale_factor[exceeds_limit] = max_following_step / change_magnitude[exceeds_limit]
                
                # Update rate-limited target
                rate_limited_target = rate_limited_target + desired_change * scale_factor
            
            # Command the rate-limited target (not the final target directly)
            yam_command_arm_pos = rate_limited_target
            
            # Build full command position (preserve gripper if present)
            if len(yam_full_current_pos) > 6:
                # Robot has gripper, preserve its current position
                yam_command_pos = np.concatenate([yam_command_arm_pos, yam_full_current_pos[6:]])
            else:
                # No gripper, just use arm positions
                yam_command_pos = yam_command_arm_pos
            
            # Command YAM to follow LeCARM's joint positions
            # YAM robot handles joint limit clamping internally
            yam_robot.command_joint_pos(yam_command_pos)
            
            # Update MuJoCo visualization if enabled
            if args.enable_visualizer:
                # Update YAM visualization (use command position, which may be interpolated)
                yam_data.qpos[:yam_model.nq] = yam_command_pos[:yam_model.nq]
                
                # Update LeCARM visualization
                lecarm_data.qpos[:lecarm_model.nq] = lecarm_joint_pos[:lecarm_model.nq]
            
            # Maintain control loop rate
            elapsed = time.time() - loop_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\n\nTeleoperation stopped by user")
    except Exception as e:
        print(f"\nError during teleoperation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nDisconnecting robots...")
        # Signal viewers to shutdown first to avoid GLX errors
        if args.enable_visualizer:
            viewer_shutdown.set()
            # Give viewer threads time to exit gracefully
            time.sleep(0.2)
        lecarm_robot.disconnect()
        yam_robot.close()
        print("Disconnected from all hardware")


if __name__ == "__main__":
    main()

