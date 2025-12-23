"""LeCARM robot module for motor state monitoring and control using damiao-motor."""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional
from damiao_motor import DaMiaoController, DaMiaoMotor

import numpy as np

from i2rt.utils.mujoco_utils import MuJoCoKDL

# Get the path to the lecarm XML file
I2RT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LECARM_XML_PATH = os.path.join(I2RT_ROOT, "robot_models", "lecarm", "lecarm.xml")


class LeCARMRobot:
    """
    LeCARM robot interface using damiao-motor for hardware control.
    
    This class provides a simple interface for monitoring and controlling
    the LeCARM robot with gravity compensation.
    """

    def __init__(
        self,
        can_interface: str = "can0",
        motor_ids: Optional[List[int]] = None,
        feedback_ids: Optional[List[int]] = None,
        bitrate: int = 1000000,
        gravity_coefficients: Optional[List[float]] = None,
    ):
        """
        Initialize LeCARM robot.

        Args:
            can_interface: CAN interface name (default: "can0")
            motor_ids: List of motor CAN IDs (default: [1, 2, 3, 4, 5, 6])
            feedback_ids: List of feedback IDs (default: same as motor_ids)
            bitrate: CAN bus bitrate (default: 1000000)
            gravity_coefficients: List of 6 gravity compensation coefficients 
                (default: [1.0, 1.0, 0.8, 2.35, 1.0, 1.0])
        """
        self.can_interface = can_interface
        self.motor_ids = motor_ids or [1, 2, 3, 4, 5, 6]
        self.feedback_ids = feedback_ids or self.motor_ids.copy()
        self.bitrate = bitrate

        # Validate inputs
        if len(self.motor_ids) != 6:
            raise ValueError(f"Expected 6 motor IDs, got {len(self.motor_ids)}")
        if len(self.feedback_ids) != 6:
            raise ValueError(f"Expected 6 feedback IDs, got {len(self.feedback_ids)}")

        # Gravity compensation coefficients (one per joint)
        if gravity_coefficients is None:
            self.gravity_coefficients = np.array([1.0, 1.0, 0.8, 2.35, 1.0, 1.0])
        else:
            if len(gravity_coefficients) != 6:
                raise ValueError(
                    f"Expected 6 gravity coefficients, got {len(gravity_coefficients)}"
                )
            self.gravity_coefficients = np.array(gravity_coefficients)

        # Initialize motor controller and motors
        self.controller = DaMiaoController(channel=self.can_interface)
        self.motors: List[Optional[DaMiaoMotor]] = [None] * 6
        self.mujoco_kdl = None  # Will be initialized when needed

        # Robot state
        self._num_dofs = 6
        self._connected = False

    def connect(self) -> bool:
        """Connect to CAN bus and initialize motors."""
        try:
            logging.info(f"Connecting to CAN interface: {self.can_interface}")

            logging.info("Initializing motors...")
            for i, (motor_id, feedback_id) in enumerate(
                zip(self.motor_ids, self.feedback_ids)
            ):
                try:
                    motor = self.controller.add_motor(motor_id, feedback_id)
                    motor.enable()
                    self.motors[i] = motor
                    logging.info(
                        f"  Motor {i+1}: ID={motor_id}, Feedback ID={feedback_id} - Connected"
                    )
                except Exception as e:
                    logging.error(f"  Motor {i+1}: Failed to connect - {e}")
                    return False

            logging.info("All motors connected successfully!")
            self._connected = True

            # Initialize MuJoCo KDL for gravity compensation
            if os.path.exists(LECARM_XML_PATH):
                logging.info("Initializing MuJoCo model for gravity compensation...")
                try:
                    self.mujoco_kdl = MuJoCoKDL(LECARM_XML_PATH)
                    logging.info(f"Loaded MuJoCo model from: {LECARM_XML_PATH}")
                except Exception as e:
                    logging.warning(f"Failed to initialize MuJoCo KDL: {e}")
                    logging.warning("Continuing without gravity compensation...")
            else:
                logging.warning(
                    f"LeCARM XML file not found at {LECARM_XML_PATH}, "
                    "gravity compensation will not work"
                )

            return True
        except Exception as e:
            logging.error(f"Failed to connect to CAN bus: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from motors and CAN bus."""
        logging.info("\nDisconnecting motors...")
        for motor in self.motors:
            if motor is not None:
                try:
                    motor.disable()
                except Exception:
                    pass

        self.motors = [None] * 6
        self._connected = False

    def read_motor_states(self) -> Optional[List[Dict]]:
        """
        Read current states from all motors.

        Returns:
            List of motor state dictionaries, or None if failed
        """
        if not self._connected:
            logging.warning("Robot not connected. Call connect() first.")
            return None

        states = []
        for i, motor in enumerate(self.motors):
            if motor is None:
                logging.error(f"Motor {i+1} is not initialized")
                return None

            try:
                # Get state directly (automatically updated)
                state = motor.get_states()
                states.append(state)
            except Exception as e:
                logging.error(f"Motor {i+1}: Failed to read state - {e}")
                return None

        return states

    def get_joint_pos(self) -> np.ndarray:
        """Get current joint positions in radians."""
        states = self.read_motor_states()
        if states is None:
            return np.zeros(6)

        positions = np.array([state.get("pos", 0.0) for state in states])
        return positions

    def get_joint_vel(self) -> np.ndarray:
        """Get current joint velocities in rad/s."""
        states = self.read_motor_states()
        if states is None:
            return np.zeros(6)

        velocities = np.array([state.get("vel", 0.0) for state in states])
        return velocities

    def get_joint_state(self) -> Dict[str, np.ndarray]:
        """Get current joint state (positions and velocities)."""
        states = self.read_motor_states()
        if states is None:
            return {"pos": np.zeros(6), "vel": np.zeros(6)}

        positions = np.array([state.get("pos", 0.0) for state in states])
        velocities = np.array([state.get("vel", 0.0) for state in states])

        return {"pos": positions, "vel": velocities}

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get current observations of the robot."""
        states = self.read_motor_states()
        if states is None:
            return {
                "joint_pos": np.zeros(6),
                "joint_vel": np.zeros(6),
                "joint_torque": np.zeros(6),
            }

        positions = np.array([state.get("pos", 0.0) for state in states])
        velocities = np.array([state.get("vel", 0.0) for state in states])
        torques = np.array([state.get("torq", 0.0) for state in states])

        return {
            "joint_pos": positions,
            "joint_vel": velocities,
            "joint_torque": torques,
        }

    def compute_gravity_compensation(
        self, joint_positions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute gravity compensation torques.

        Args:
            joint_positions: Joint positions in radians. If None, uses current positions.

        Returns:
            Array of gravity compensation torques for each joint.
        """
        if self.mujoco_kdl is None:
            return np.zeros(6)

        if joint_positions is None:
            joint_positions = self.get_joint_pos()

        try:
            # Compute inverse dynamics with zero velocities and accelerations (gravity only)
            qdot = np.zeros(6)
            qdotdot = np.zeros(6)
            raw_gravity_torques = self.mujoco_kdl.compute_inverse_dynamics(
                joint_positions, qdot, qdotdot
            )
            # Apply gravity compensation coefficients
            gravity_torques = raw_gravity_torques * self.gravity_coefficients
            return gravity_torques
        except Exception as e:
            logging.warning(f"Failed to calculate gravity torques: {e}")
            return np.zeros(6)

    def command_joint_pos(
        self,
        joint_pos: np.ndarray,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        use_gravity_comp: bool = True,
    ) -> None:
        """
        Command joint positions with optional PD control and gravity compensation.

        Args:
            joint_pos: Target joint positions in radians (6 elements)
            kp: Proportional gains (default: zeros, position control only)
            kd: Derivative gains (default: zeros, no damping)
            use_gravity_comp: Whether to apply gravity compensation
        """
        if not self._connected:
            logging.warning("Robot not connected. Call connect() first.")
            return

        if len(joint_pos) != 6:
            raise ValueError(f"Expected 6 joint positions, got {len(joint_pos)}")

        if kp is None:
            kp = np.zeros(6)
        if kd is None:
            kd = np.zeros(6)

        # Get current positions for PD control
        current_pos = self.get_joint_pos()
        current_vel = self.get_joint_vel()

        # Compute PD torques
        pos_error = joint_pos - current_pos
        pd_torques = kp * pos_error - kd * current_vel

        # Compute gravity compensation
        gravity_torques = (
            self.compute_gravity_compensation(current_pos) if use_gravity_comp else np.zeros(6)
        )

        # Send commands to motors
        for i, (motor, torque) in enumerate(zip(self.motors, gravity_torques)):
            if motor is None:
                continue
            try:
                motor.send_cmd(
                    target_position=joint_pos[i],
                    target_velocity=0.0,
                    stiffness=kp[i],
                    damping=kd[i],
                    feedforward_torque=torque,
                    control_mode="MIT",
                )
            except Exception as e:
                logging.warning(f"Motor {i+1} failed to send command: {e}")

    def send_gravity_compensation_only(self) -> None:
        """Send only gravity compensation torques (zero position/velocity targets)."""
        if not self._connected:
            logging.warning("Robot not connected. Call connect() first.")
            return

        current_pos = self.get_joint_pos()
        gravity_torques = self.compute_gravity_compensation(current_pos)

        # Send commands with gravity compensation torques
        for i, (motor, torque) in enumerate(zip(self.motors, gravity_torques)):
            if motor is None:
                continue
            try:
                motor.send_cmd(
                    target_position=0.0,
                    target_velocity=0.0,
                    stiffness=0.0,
                    damping=0.0,
                    feedforward_torque=torque,
                    control_mode="MIT",
                )
            except Exception as e:
                logging.warning(f"Motor {i+1} failed to send command: {e}")

    def num_dofs(self) -> int:
        """Get number of degrees of freedom."""
        return self._num_dofs

    def close(self) -> None:
        """Close the robot and clean up resources."""
        self.disconnect()

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


# Convenience function for backward compatibility
def get_lecarm_robot(
    can_interface: str = "can0",
    motor_ids: Optional[List[int]] = None,
    feedback_ids: Optional[List[int]] = None,
    bitrate: int = 1000000,
    gravity_coefficients: Optional[List[float]] = None,
) -> LeCARMRobot:
    """
    Initialize a LeCARM robot.

    Args:
        can_interface: CAN interface name
        motor_ids: List of motor CAN IDs
        feedback_ids: List of feedback IDs
        bitrate: CAN bus bitrate
        gravity_coefficients: List of gravity compensation coefficients

    Returns:
        LeCARMRobot instance
    """
    robot = LeCARMRobot(
        can_interface=can_interface,
        motor_ids=motor_ids,
        feedback_ids=feedback_ids,
        bitrate=bitrate,
        gravity_coefficients=gravity_coefficients,
    )
    robot.connect()
    return robot
