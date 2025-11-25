"""MuJoCo visualizer with keyboard control for end effector pose.

This module provides an interactive simulator that allows you to control
the robot arm's end effector pose using keyboard commands.
"""

import argparse
import os
import queue
import threading
import time
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np

from i2rt.robots.kinematics import Kinematics
from i2rt.robots.utils import YAM_NO_GRIPPER_PATH, YAM_XML_PATH


class EndEffectorPoseController:
    """Controller for end effector pose using keyboard input."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        kinematics: Kinematics,
        site_name: str = "grasp_site",
        step_size_pos: float = 0.01,
        step_size_rot: float = 0.05,
    ):
        """Initialize the controller.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            kinematics: Kinematics solver for IK
            site_name: Name of the end effector site
            step_size_pos: Position step size in meters
            step_size_rot: Rotation step size in radians
        """
        self.model = model
        self.data = data
        self.kinematics = kinematics
        self.site_name = site_name
        self.step_size_pos = step_size_pos
        self.step_size_rot = step_size_rot

        # Initialize joint positions
        self.current_q = np.zeros(model.nq)
        data.qpos[:] = self.current_q
        mujoco.mj_forward(model, data)

        # Get initial end effector pose
        self.target_pose = self.kinematics.fk(self.current_q, site_name)
        self.current_pose = self.target_pose.copy()

    def get_current_pose(self) -> np.ndarray:
        """Get the current end effector pose."""
        return self.kinematics.fk(self.current_q, self.site_name)

    def update_target_pose(
        self,
        delta_pos: Optional[np.ndarray] = None,
        delta_rot: Optional[np.ndarray] = None,
    ) -> bool:
        """Update target pose and solve IK.

        Args:
            delta_pos: Position delta in world frame (3,)
            delta_rot: Rotation delta as Euler angles (roll, pitch, yaw) in radians (3,)

        Returns:
            True if IK converged, False otherwise
        """
        # Update target position
        if delta_pos is not None:
            self.target_pose[:3, 3] += delta_pos

        # Update target orientation
        if delta_rot is not None:
            # Convert Euler angles to rotation matrix
            # Using ZYX (intrinsic) Euler angles: roll, pitch, yaw
            roll, pitch, yaw = delta_rot
            cx, sx = np.cos(roll), np.sin(roll)
            cy, sy = np.cos(pitch), np.sin(pitch)
            cz, sz = np.cos(yaw), np.sin(yaw)

            # Rotation matrix for delta rotation
            R_delta = np.array(
                [
                    [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
                    [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
                    [-sy, cy * sx, cy * cx],
                ]
            )

            # Apply delta rotation to current orientation
            self.target_pose[:3, :3] = self.target_pose[:3, :3] @ R_delta

        # Solve IK
        success, q_new = self.kinematics.ik(
            self.target_pose,
            self.site_name,
            init_q=self.current_q,
            max_iters=50,
            pos_threshold=1e-3,
            ori_threshold=1e-2,
        )

        if success:
            self.current_q = q_new
            self.data.qpos[: self.model.nq] = self.current_q
            mujoco.mj_forward(self.model, self.data)
            self.current_pose = self.get_current_pose()
            return True
        return False

    def reset_to_home(self) -> None:
        """Reset to home position."""
        self.current_q = np.zeros(self.model.nq)
        self.data.qpos[:] = self.current_q
        mujoco.mj_forward(self.model, self.data)
        self.target_pose = self.get_current_pose()
        self.current_pose = self.target_pose.copy()


def print_instructions() -> None:
    """Print keyboard control instructions."""
    print("\n" + "=" * 60)
    print("End Effector Pose Control - Keyboard Commands")
    print("=" * 60)
    print("Position Control:")
    print("  w/s : Move forward/backward (Y axis)")
    print("  a/d : Move left/right (X axis)")
    print("  t/g : Move up/down (Z axis)")
    print("\nOrientation Control:")
    print("  i/k : Rotate around X axis (roll)")
    print("  j/l : Rotate around Y axis (pitch)")
    print("  u/o : Rotate around Z axis (yaw)")
    print("\nOther Controls:")
    print("  r   : Reset to home position")
    print("  h   : Show this help message")
    print("  q   : Quit")
    print("=" * 60 + "\n")


def main() -> None:
    """Main function to run the simulator."""
    parser = argparse.ArgumentParser(description="MuJoCo visualizer with keyboard control")
    parser.add_argument(
        "--xml_path",
        type=str,
        default=YAM_NO_GRIPPER_PATH,
        help="Path to MuJoCo XML model file",
    )
    parser.add_argument(
        "--site_name",
        type=str,
        default="grasp_site",
        help="Name of the end effector site",
    )
    parser.add_argument(
        "--step_size_pos",
        type=float,
        default=0.01,
        help="Position step size in meters",
    )
    parser.add_argument(
        "--step_size_rot",
        type=float,
        default=0.05,
        help="Rotation step size in radians",
    )
    args = parser.parse_args()

    # Load model
    xml_path = os.path.expanduser(args.xml_path)
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Initialize kinematics
    kinematics = Kinematics(xml_path, args.site_name)

    # Initialize controller
    controller = EndEffectorPoseController(
        model=model,
        data=data,
        kinematics=kinematics,
        site_name=args.site_name,
        step_size_pos=args.step_size_pos,
        step_size_rot=args.step_size_rot,
    )

    # Print instructions
    print_instructions()

    # Keyboard input queue
    key_queue = queue.Queue()
    stop_event = threading.Event()

    def keyboard_listener():
        """Listen for keyboard input in a separate thread."""
        try:
            import sys
            import select
            import tty
            import termios

            # Set terminal to raw mode
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())

            while not stop_event.is_set():
                if select.select([sys.stdin], [], [], 0.01)[0]:
                    key = sys.stdin.read(1)
                    if key:
                        key_queue.put(ord(key))
                time.sleep(0.01)

            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except ImportError:
            # Fallback for Windows or systems without termios
            try:
                import msvcrt

                while not stop_event.is_set():
                    if msvcrt.kbhit():
                        key = msvcrt.getch()
                        if key:
                            if isinstance(key, bytes):
                                key_queue.put(key[0])
                            else:
                                key_queue.put(ord(key))
                    time.sleep(0.01)
            except ImportError:
                print("Warning: Keyboard input not supported on this platform")
                print("The viewer will open but keyboard controls won't work.")
        except Exception as e:
            print(f"Keyboard listener error: {e}")
            print("Keyboard input may not work. Continuing with viewer only...")

    # Start keyboard listener thread
    kb_thread = threading.Thread(target=keyboard_listener, daemon=True)
    kb_thread.start()

    # Control loop
    dt = 0.01
    last_key_time = time.time()
    key_repeat_delay = 0.1  # Delay before key repeat

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        # Set camera to follow the end effector
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        print("Simulator started. Use keyboard to control the end effector.")
        print("Make sure the terminal window has focus for keyboard input.")
        print("Press 'h' for help, or 'q' to quit.\n")

        while viewer.is_running():
            step_start = time.time()

            # Handle keyboard input from queue
            current_time = time.time()
            delta_pos = None
            delta_rot = None
            handled = False

            try:
                key = key_queue.get_nowait()
            except queue.Empty:
                key = None

            if key is not None:
                # Position controls
                if key == ord("w"):  # Forward (Y+)
                    delta_pos = np.array([0, controller.step_size_pos, 0])
                    handled = True
                elif key == ord("s"):  # Backward (Y-)
                    delta_pos = np.array([0, -controller.step_size_pos, 0])
                    handled = True
                elif key == ord("a"):  # Left (X-)
                    delta_pos = np.array([-controller.step_size_pos, 0, 0])
                    handled = True
                elif key == ord("d"):  # Right (X+)
                    delta_pos = np.array([controller.step_size_pos, 0, 0])
                    handled = True
                elif key == ord("q"):  # Quit
                    print("\nQuitting...")
                    break
                elif key == ord("r"):  # Reset
                    controller.reset_to_home()
                    handled = True
                elif key == ord("h"):  # Help
                    print_instructions()
                    handled = True

                # Orientation controls (using arrow keys would require special handling)
                # Using number pad or other keys
                elif key == ord("i"):  # Roll positive (X axis)
                    delta_rot = np.array([controller.step_size_rot, 0, 0])
                    handled = True
                elif key == ord("k"):  # Roll negative (X axis)
                    delta_rot = np.array([-controller.step_size_rot, 0, 0])
                    handled = True
                elif key == ord("j"):  # Pitch positive (Y axis)
                    delta_rot = np.array([0, controller.step_size_rot, 0])
                    handled = True
                elif key == ord("l"):  # Pitch negative (Y axis)
                    delta_rot = np.array([0, -controller.step_size_rot, 0])
                    handled = True
                elif key == ord("u"):  # Yaw positive (Z axis)
                    delta_rot = np.array([0, 0, controller.step_size_rot])
                    handled = True
                elif key == ord("o"):  # Yaw negative (Z axis)
                    delta_rot = np.array([0, 0, -controller.step_size_rot])
                    handled = True
                elif key == ord("t"):  # Up (Z+)
                    delta_pos = np.array([0, 0, controller.step_size_pos])
                    handled = True
                elif key == ord("g"):  # Down (Z-)
                    delta_pos = np.array([0, 0, -controller.step_size_pos])
                    handled = True

                # Update pose if needed
                if handled and (delta_pos is not None or delta_rot is not None):
                    if current_time - last_key_time > key_repeat_delay:
                        success = controller.update_target_pose(delta_pos, delta_rot)
                        if not success:
                            print("Warning: IK did not converge")
                        last_key_time = current_time

            # Sync viewer
            viewer.sync()

            # Maintain target frame rate
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # Stop keyboard listener
    stop_event.set()

    print("\nSimulator closed.")


if __name__ == "__main__":
    main()

