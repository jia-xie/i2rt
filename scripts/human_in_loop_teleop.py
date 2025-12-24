"""Human-in-the-loop teleoperation with mode switching.

This script provides a human-in-the-loop teleoperation system with one YAM follower
that can be controlled by either:
    - LeCARM (physical leader): Physical LeCARM controls YAM follower
    - YAM Leader (policy simulation): YAM leader robot controls YAM follower

Keyboard controls:
    '1' - Switch to LeCARM leader mode (physical teleop)
    '2' - Switch to YAM leader mode (policy simulation)
    'q' - Quit

When in YAM leader mode, the follower's joint positions are also mapped to LeCARM
with slow sync behavior, allowing the human to see what the policy is doing.

Both modes use slow sync behavior to safely follow the leader.

Usage:
    uv run python scripts/human_in_loop_teleop.py
    
    Options:
        --lecarm-can-interface INTERFACE    CAN interface for LeCARM (default: can0)
        --yam-can-interface INTERFACE       CAN interface for YAM robots (default: can0)
        --yam-leader-can-interface INTERFACE CAN interface for YAM leader (default: same as yam-can-interface)
        --yam-follower-can-interface INTERFACE CAN interface for YAM follower (default: same as yam-can-interface)
        --rate RATE                         Control loop rate in Hz (default: 200)
        --kp KP                             Position gain for YAM PD control (default: 80.0)
        --kd KD                             Velocity damping for YAM PD control (default: 5.0)
        --slow-sync-threshold THRESHOLD     Joint error (rad) to switch to slow_sync mode (default: 0.5)
        --following-threshold THRESHOLD     Joint error (rad) to switch to following mode (default: 0.2)
        --max-sync-speed SPEED              Max joint speed for slow_sync mode in rad/s (default: 0.5)
        --max-following-speed SPEED         Max joint speed for following mode in rad/s (default: 2.0)
        --enable-visualizer                 Enable MuJoCo visualization
"""

import argparse
import os
import select
import signal
import sys
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


def apply_inverse_joint_mapping(
    yam_joint_pos: np.ndarray, joint_mapping: list
) -> np.ndarray:
    """
    Apply inverse joint mapping from YAM to LeCARM.
    
    Args:
        yam_joint_pos: Joint positions from YAM (6 elements)
        joint_mapping: List of mapping dicts with 'target_joint' and 'direction'
    
    Returns:
        Mapped joint positions for LeCARM (6 elements)
    """
    lecarm_joint_pos = np.zeros(6)
    
    # Create inverse mapping: find which LeCARM joint maps to each YAM joint
    for lecarm_idx, mapping in enumerate(joint_mapping):
        yam_idx = mapping["target_joint"]
        direction = mapping["direction"]
        
        if 0 <= yam_idx < 6:
            # Inverse: YAM joint -> LeCARM joint (with same direction)
            lecarm_joint_pos[lecarm_idx] = yam_joint_pos[yam_idx] * direction
    
    return lecarm_joint_pos


class KeyboardListener:
    """Non-blocking keyboard input listener."""
    
    def __init__(self):
        self.current_key = None
        self.running = True
        self.thread = None
        self.old_settings = None
        
    def _read_keyboard(self):
        """Thread function to read keyboard input."""
        import termios
        import tty
        
        # Save terminal settings
        self.old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            
            while self.running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    # Handle Ctrl+C (0x03) and Ctrl+D (0x04)
                    if key == '\x03' or key == '\x04':  # Ctrl+C or Ctrl+D
                        self.running = False
                        break
                    self.current_key = key
                    if key == 'q':
                        self.running = False
                        break
        finally:
            # Restore terminal settings
            if self.old_settings is not None:
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
                except:
                    pass
    
    def start(self):
        """Start the keyboard listener thread."""
        self.thread = threading.Thread(target=self._read_keyboard, daemon=True)
        self.thread.start()
    
    def get_key(self):
        """Get the last pressed key and clear it."""
        key = self.current_key
        self.current_key = None
        return key
    
    def stop(self):
        """Stop the keyboard listener."""
        self.running = False
        # Restore terminal settings immediately
        if self.old_settings is not None:
            import termios
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except:
                pass
        if self.thread:
            self.thread.join(timeout=0.1)


def apply_rate_limiting(
    current_target: np.ndarray,
    final_target: np.ndarray,
    max_step: float,
) -> np.ndarray:
    """
    Apply rate limiting to smoothly move from current_target to final_target.
    
    Args:
        current_target: Current rate-limited target position
        final_target: Desired final target position
        max_step: Maximum change per control cycle (rad)
    
    Returns:
        New rate-limited target position
    """
    desired_change = final_target - current_target
    change_magnitude = np.abs(desired_change)
    
    # Scale down if any joint exceeds max speed
    scale_factor = np.ones(6)
    exceeds_limit = change_magnitude > max_step
    if np.any(exceeds_limit):
        # For joints that exceed limit, scale to max_step
        scale_factor[exceeds_limit] = max_step / change_magnitude[exceeds_limit]
    
    # Update rate-limited target
    new_target = current_target + desired_change * scale_factor
    return new_target


def main():
    parser = argparse.ArgumentParser(
        description="Human-in-the-loop teleoperation with mode switching"
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
        help="CAN interface for YAM robots (default: can0)",
    )
    parser.add_argument(
        "--yam-leader-can-interface",
        type=str,
        default=None,
        help="CAN interface for YAM leader (default: same as yam-can-interface)",
    )
    parser.add_argument(
        "--yam-follower-can-interface",
        type=str,
        default=None,
        help="CAN interface for YAM follower (default: same as yam-can-interface)",
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
        "Default: [80, 80, 80, 40, 10, 10]",
    )
    parser.add_argument(
        "--kd",
        type=float,
        nargs="+",
        default=None,
        help="Velocity damping for YAM PD control (6 values, or single value for all joints). "
        "Default: [5, 5, 5, 1.5, 1.5, 1.5]",
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
        help="Enable MuJoCo visualization of all robots",
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
    
    # Determine CAN interfaces for YAM robots
    yam_leader_can = args.yam_leader_can_interface or args.yam_can_interface
    yam_follower_can = args.yam_follower_can_interface or args.yam_can_interface
    
    print("=" * 60)
    print("Human-in-the-Loop Teleoperation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  LeCARM CAN interface: {lecarm_can_interface}")
    print(f"  YAM Leader CAN interface: {yam_leader_can}")
    print(f"  YAM Follower CAN interface: {yam_follower_can}")
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
    
    # Connect to YAM leader robot (for policy simulation mode)
    print(f"\nConnecting to YAM Leader on {yam_leader_can}...")
    yam_gripper_type = GripperType.from_string_name(args.yam_gripper_type)
    yam_leader_robot = get_yam_robot(
        channel=yam_leader_can,
        gripper_type=yam_gripper_type,
        zero_gravity_mode=True,
    )
    
    # Connect to YAM follower robot
    print(f"\nConnecting to YAM Follower on {yam_follower_can}...")
    yam_follower_robot = get_yam_robot(
        channel=yam_follower_can,
        gripper_type=yam_gripper_type,
        zero_gravity_mode=False,
    )
    
    # Default YAM PD gains
    default_yam_kp = np.array([80, 80, 80, 40, 10, 10])
    default_yam_kd = np.array([5, 5, 5, 1.5, 1.5, 1.5])
    
    # Handle PD gain arguments
    if args.kp is not None:
        if len(args.kp) == 1:
            yam_kp = np.array([args.kp[0]] * 6)
        elif len(args.kp) == 6:
            yam_kp = np.array(args.kp)
        else:
            raise ValueError(f"kp must be 1 or 6 values, got {len(args.kp)}")
    else:
        yam_kp = default_yam_kp
    
    if args.kd is not None:
        if len(args.kd) == 1:
            yam_kd = np.array([args.kd[0]] * 6)
        elif len(args.kd) == 6:
            yam_kd = np.array(args.kd)
        else:
            raise ValueError(f"kd must be 1 or 6 values, got {len(args.kd)}")
    else:
        yam_kd = default_yam_kd
    
    # Set PD gains for YAM follower (only for arm joints, preserve gripper gains if present)
    current_kp = yam_follower_robot._kp.copy()
    current_kd = yam_follower_robot._kd.copy()
    current_kp[:6] = yam_kp
    current_kd[:6] = yam_kd
    yam_follower_robot.update_kp_kd(current_kp, current_kd)
    
    print("✓ YAM Follower connected successfully")
    print(f"  YAM Follower PD gains: kp={yam_kp.tolist()}, kd={yam_kd.tolist()} (arm joints only)")
    
    # Setup MuJoCo visualization if requested
    yam_leader_model = None
    yam_leader_data = None
    yam_follower_model = None
    yam_follower_data = None
    lecarm_model = None
    lecarm_data = None
    if args.enable_visualizer:
        import mujoco
        import mujoco.viewer
        
        yam_xml_path = yam_gripper_type.get_xml_path()
        print(f"\nLoading MuJoCo models for visualization...")
        yam_leader_model = mujoco.MjModel.from_xml_path(yam_xml_path)
        yam_leader_data = mujoco.MjData(yam_leader_model)
        yam_follower_model = mujoco.MjModel.from_xml_path(yam_xml_path)
        yam_follower_data = mujoco.MjData(yam_follower_model)
        lecarm_model = mujoco.MjModel.from_xml_path(LECARM_XML_PATH)
        lecarm_data = mujoco.MjData(lecarm_model)
        print("✓ MuJoCo models loaded")
    
    period = 1.0 / args.rate
    max_sync_step = args.max_sync_speed * period
    max_following_step = args.max_following_speed * period
    
    print(f"\nControl rate: {args.rate} Hz")
    print(f"Slow sync threshold: {args.slow_sync_threshold} rad")
    print(f"Following threshold: {args.following_threshold} rad")
    print(f"Max sync speed: {args.max_sync_speed} rad/s")
    print(f"Max following speed: {args.max_following_speed} rad/s")
    print("\nKeyboard Controls:")
    print("  '1' - Switch to LeCARM leader mode (physical teleop)")
    print("  '2' - Switch to YAM leader mode (policy simulation)")
    print("  'q' - Quit")
    print("\nPress Ctrl+C to stop\n")
    
    # Initialize keyboard listener
    keyboard = KeyboardListener()
    keyboard.start()
    
    # Initialize mode
    current_mode = "lecarm"  # Start with LeCARM mode
    print(f"Starting in {current_mode} mode\n")
    
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
        
        def yam_leader_viewer_thread():
            """Separate thread for YAM leader viewer."""
            try:
                with mujoco.viewer.launch_passive(
                    model=yam_leader_model,
                    data=yam_leader_data,
                    show_left_ui=False,
                    show_right_ui=False,
                ) as viewer:
                    mujoco.mjv_defaultFreeCamera(yam_leader_model, viewer.cam)
                    viewer.cam.lookat[:] = [0.3, 0, 0.2]
                    viewer.cam.distance = 1.5
                    
                    while viewer.is_running() and not viewer_shutdown.is_set():
                        mujoco.mj_kinematics(yam_leader_model, yam_leader_data)
                        viewer.sync()
                        time.sleep(0.01)
            except Exception:
                pass
        
        def yam_follower_viewer_thread():
            """Separate thread for YAM follower viewer."""
            try:
                with mujoco.viewer.launch_passive(
                    model=yam_follower_model,
                    data=yam_follower_data,
                    show_left_ui=False,
                    show_right_ui=False,
                ) as viewer:
                    mujoco.mjv_defaultFreeCamera(yam_follower_model, viewer.cam)
                    viewer.cam.lookat[:] = [0.3, 0, 0.2]
                    viewer.cam.distance = 1.5
                    
                    while viewer.is_running() and not viewer_shutdown.is_set():
                        mujoco.mj_kinematics(yam_follower_model, yam_follower_data)
                        viewer.sync()
                        time.sleep(0.01)
            except Exception:
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
                pass
        
        print("Starting MuJoCo viewers...")
        yam_leader_thread = threading.Thread(target=yam_leader_viewer_thread, daemon=True)
        yam_follower_thread = threading.Thread(target=yam_follower_viewer_thread, daemon=True)
        lecarm_thread = threading.Thread(target=lecarm_viewer_thread, daemon=True)
        yam_leader_thread.start()
        yam_follower_thread.start()
        lecarm_thread.start()
        visualizer_threads = [yam_leader_thread, yam_follower_thread, lecarm_thread]
        time.sleep(0.5)
    
    # Setup signal handler for Ctrl+C
    shutdown_event = threading.Event()
    def signal_handler(signum, frame):
        print("\n\nReceived interrupt signal, shutting down...")
        keyboard.running = False
        shutdown_event.set()
        keyboard.stop()  # Restore terminal immediately
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize mode tracking and rate-limited targets
        yam_follower_full_current_pos = yam_follower_robot.get_joint_pos()
        yam_follower_rate_limited_target = yam_follower_full_current_pos[:6].copy()
        lecarm_rate_limited_target = None  # Will be initialized when needed
        
        # Main control loop
        while keyboard.running and not shutdown_event.is_set():
            loop_start = time.time()
            
            # Check for keyboard input to switch modes
            key = keyboard.get_key()
            if key == '1':
                if current_mode != "lecarm":
                    current_mode = "lecarm"
                    print(f"\nSwitched to LeCARM leader mode")
                    # Reset rate-limited target when switching modes
                    yam_follower_full_current_pos = yam_follower_robot.get_joint_pos()
                    yam_follower_rate_limited_target = yam_follower_full_current_pos[:6].copy()
            elif key == '2':
                if current_mode != "yam_leader":
                    current_mode = "yam_leader"
                    print(f"\nSwitched to YAM leader mode")
                    # Reset rate-limited targets when switching modes
                    yam_follower_full_current_pos = yam_follower_robot.get_joint_pos()
                    yam_follower_rate_limited_target = yam_follower_full_current_pos[:6].copy()
                    lecarm_current_pos = lecarm_robot.get_joint_pos()
                    lecarm_rate_limited_target = lecarm_current_pos.copy()
                    # Initialize LeCARM sync mode to slow_sync
                    lecarm_robot._lecarm_sync_mode = "slow_sync"
            
            # Get leader joint positions based on current mode
            if current_mode == "lecarm":
                # Read LeCARM joint positions from physical robot
                lecarm_joint_pos = lecarm_robot.get_joint_pos()
                
                if len(lecarm_joint_pos) != 6:
                    print("Warning: Expected 6 joint positions from LeCARM, got", len(lecarm_joint_pos))
                    time.sleep(period)
                    continue
                
                # Apply gravity compensation to LeCARM
                lecarm_robot.send_gravity_compensation_only()
                
                # Map LeCARM positions to YAM follower using joint mapping
                yam_follower_final_target = apply_joint_mapping(lecarm_joint_pos, joint_mapping)
                
            elif current_mode == "yam_leader":
                # Read YAM leader joint positions
                yam_leader_full_pos = yam_leader_robot.get_joint_pos()
                yam_leader_pos = yam_leader_full_pos[:6]  # Arm joints only
                
                if len(yam_leader_pos) != 6:
                    print("Warning: Expected 6 joint positions from YAM leader, got", len(yam_leader_pos))
                    time.sleep(period)
                    continue
                
                # YAM follower should follow YAM leader
                yam_follower_final_target = yam_leader_pos.copy()
            else:
                # Unknown mode, skip this cycle
                time.sleep(period)
                continue
            
            # Get current YAM follower position
            yam_follower_full_current_pos = yam_follower_robot.get_joint_pos()
            yam_follower_current_pos = yam_follower_full_current_pos[:6]  # Arm joints only
            
            # Calculate joint error for mode switching logic
            joint_error = np.abs(yam_follower_final_target - yam_follower_current_pos)
            max_error = np.max(joint_error)
            
            # Determine sync mode based on error with hysteresis
            # (This is internal to the slow sync behavior, not the main mode)
            if max_error > args.slow_sync_threshold:
                sync_mode = "slow_sync"
                max_step = max_sync_step
            elif max_error <= args.following_threshold:
                sync_mode = "following"
                max_step = max_following_step
            else:
                # Stay in current sync mode (hysteresis)
                sync_mode = "slow_sync" if max_error > (args.slow_sync_threshold + args.following_threshold) / 2 else "following"
                max_step = max_sync_step if sync_mode == "slow_sync" else max_following_step
            
            # Apply rate limiting to YAM follower target
            yam_follower_rate_limited_target = apply_rate_limiting(
                yam_follower_rate_limited_target,
                yam_follower_final_target,
                max_step,
            )
            
            # Command YAM follower
            yam_follower_command_arm_pos = yam_follower_rate_limited_target
            
            # Build full command position (preserve gripper if present)
            if len(yam_follower_full_current_pos) > 6:
                yam_follower_command_pos = np.concatenate([yam_follower_command_arm_pos, yam_follower_full_current_pos[6:]])
            else:
                yam_follower_command_pos = yam_follower_command_arm_pos
            
            yam_follower_robot.command_joint_pos(yam_follower_command_pos)
            
            # In YAM leader mode, also command LeCARM to follow YAM follower's rate-limited position
            if current_mode == "yam_leader":
                # Map YAM follower's rate-limited target position to LeCARM
                # This shows what the follower is being commanded to do (after rate limiting)
                lecarm_final_target = apply_inverse_joint_mapping(yam_follower_rate_limited_target, joint_mapping)
                
                # Get current LeCARM position (initialize if needed)
                if lecarm_rate_limited_target is None:
                    lecarm_current_pos = lecarm_robot.get_joint_pos()
                    lecarm_rate_limited_target = lecarm_current_pos.copy()
                
                # Calculate LeCARM joint error to determine sync mode
                lecarm_current_pos = lecarm_robot.get_joint_pos()
                lecarm_joint_error = np.abs(lecarm_final_target - lecarm_current_pos)
                lecarm_max_error = np.max(lecarm_joint_error)
                
                # Determine LeCARM sync mode based on error with hysteresis
                # Start in slow_sync, switch to following when error is small
                if not hasattr(lecarm_robot, '_lecarm_sync_mode'):
                    lecarm_robot._lecarm_sync_mode = "slow_sync"  # Initialize to slow_sync
                
                if lecarm_robot._lecarm_sync_mode == "slow_sync":
                    # In slow_sync mode, switch to following when error is small enough
                    if lecarm_max_error <= args.following_threshold:
                        lecarm_robot._lecarm_sync_mode = "following"
                        print(f"LeCARM switched to following mode (max error: {lecarm_max_error:.3f} rad)")
                else:  # following mode
                    # In following mode, switch to slow_sync when error is too large
                    if lecarm_max_error > args.slow_sync_threshold:
                        lecarm_robot._lecarm_sync_mode = "slow_sync"
                        print(f"LeCARM switched to slow_sync mode (max error: {lecarm_max_error:.3f} rad)")
                
                # Apply rate limiting to LeCARM target based on sync mode
                lecarm_max_step = max_sync_step if lecarm_robot._lecarm_sync_mode == "slow_sync" else max_following_step
                lecarm_rate_limited_target = apply_rate_limiting(
                    lecarm_rate_limited_target,
                    lecarm_final_target,
                    lecarm_max_step,
                )
                
                # Command LeCARM to follow YAM follower position (with gravity compensation)
                # Use small position gains for soft following
                lecarm_kp = np.array([20.0, 20.0, 20.0, 10.0, 5.0, 5.0])  # Small position control gains
                lecarm_kd = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])  # Small damping
                lecarm_robot.command_joint_pos(
                    lecarm_rate_limited_target,
                    kp=lecarm_kp,
                    kd=lecarm_kd,
                    use_gravity_comp=True,
                )
            
            # Update MuJoCo visualization if enabled
            if args.enable_visualizer:
                # Update YAM leader visualization
                if current_mode == "yam_leader":
                    yam_leader_data.qpos[:yam_leader_model.nq] = yam_leader_full_pos[:yam_leader_model.nq]
                
                # Update YAM follower visualization
                yam_follower_data.qpos[:yam_follower_model.nq] = yam_follower_command_pos[:yam_follower_model.nq]
                
                # Update LeCARM visualization
                if current_mode == "lecarm":
                    lecarm_data.qpos[:lecarm_model.nq] = lecarm_joint_pos[:lecarm_model.nq]
                elif current_mode == "yam_leader":
                    lecarm_data.qpos[:lecarm_model.nq] = lecarm_rate_limited_target[:lecarm_model.nq]
            
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
        # Restore terminal settings first
        keyboard.stop()
        # Signal viewers to shutdown first
        if args.enable_visualizer:
            viewer_shutdown.set()
            time.sleep(0.2)
        try:
            lecarm_robot.disconnect()
        except:
            pass
        try:
            yam_leader_robot.close()
        except:
            pass
        try:
            yam_follower_robot.close()
        except:
            pass
        print("Disconnected from all hardware")


if __name__ == "__main__":
    main()

