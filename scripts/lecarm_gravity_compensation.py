"""LeCARM gravity compensation script - continuously apply gravity compensation torques."""

import argparse
import time
from typing import List, Optional

import numpy as np

from i2rt.robots.lecarm import LeCARMRobot


def main():
    """Main function for LeCARM gravity compensation."""
    parser = argparse.ArgumentParser(
        description="LeCARM gravity compensation - continuously apply gravity compensation torques"
    )
    parser.add_argument(
        "--can-interface",
        type=str,
        default="can0",
        help="CAN interface name (default: can0)",
    )
    parser.add_argument(
        "--motor-ids",
        type=int,
        nargs=6,
        default=[1, 2, 3, 4, 5, 6],
        help="Motor CAN IDs for joints 1-6 (default: 1 2 3 4 5 6)",
    )
    parser.add_argument(
        "--feedback-ids",
        type=int,
        nargs=6,
        default=None,
        help="Feedback IDs (default: same as motor-ids)",
    )
    parser.add_argument(
        "--bitrate",
        type=int,
        default=1000000,
        help="CAN bus bitrate (default: 1000000)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=200.0,
        help="Control loop rate in Hz (default: 200)",
    )
    parser.add_argument(
        "--gravity-coeffs",
        type=float,
        nargs=6,
        default=[1.0, 1.0, 0.8, 2.35, 1.0, 1.0],
        help="Gravity compensation coefficients for joints 1-6 "
        "(default: 1.0 1.0 0.8 2.35 1.0 1.0)",
    )
    parser.add_argument(
        "--print-state",
        action="store_true",
        help="Print motor states (default: False)",
    )
    parser.add_argument(
        "--print-joint-4",
        action="store_true",
        help="Print detailed state for joint 4 (default: False)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("LeCARM GRAVITY COMPENSATION")
    print("=" * 60)

    # Create robot
    robot = LeCARMRobot(
        can_interface=args.can_interface,
        motor_ids=args.motor_ids,
        feedback_ids=args.feedback_ids,
        bitrate=args.bitrate,
        gravity_coefficients=args.gravity_coeffs,
    )

    # Connect
    if not robot.connect():
        print("Failed to connect. Exiting.")
        return

    try:
        period = 1.0 / args.rate
        print(f"\nStarting gravity compensation at {args.rate} Hz")
        print("Press Ctrl+C to stop\n")

        # Send initial zero commands
        for motor in robot.motors:
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

        while True:
            loop_start = time.time()

            # Read motor states
            states = robot.read_motor_states()
            if states is None:
                print("Warning: Failed to read motor states")
                time.sleep(period)
                continue

            # Get current joint positions
            current_pos = robot.get_joint_pos()

            # Compute gravity compensation torques
            gravity_torques = robot.compute_gravity_compensation(current_pos)

            # Send gravity compensation commands
            robot.send_gravity_compensation_only()

            # Small delay between commands
            time.sleep(0.005)

            # Print states if requested
            if args.print_state:
                print("\n" + "=" * 95)
                print(
                    f"{'M':<3} {'ID':<4} {'Pos(rad)':<10} {'Vel(rad/s)':<11} "
                    f"{'Torq(Nm)':<10} {'Grav(Nm)':<10} {'Stat':<6} {'Tmos':<6} {'Trot':<6}"
                )
                print("-" * 95)

                for i, state in enumerate(states):
                    motor_id = state.get("can_id", args.motor_ids[i])
                    pos = state.get("pos", 0.0)
                    vel = state.get("vel", 0.0)
                    torq = state.get("torq", 0.0)
                    status = str(state.get("status", "UNKNOWN"))[:5]
                    temp_mos = state.get("t_mos", 0.0)
                    temp_rotor = state.get("t_rotor", 0.0)

                    gravity_torque = (
                        gravity_torques[i] if i < len(gravity_torques) else 0.0
                    )

                    print(
                        f"{i+1:<3} {motor_id:<4} {pos:>9.3f} {vel:>10.3f} "
                        f"{torq:>9.3f} {gravity_torque:>9.3f} {status:<6} "
                        f"{temp_mos:>5.1f} {temp_rotor:>5.1f}"
                    )

                print("=" * 95)

            # Print detailed state for joint 4 if requested
            if args.print_joint_4 and len(states) > 3:
                joint_4_state = states[3]
                joint_4_gravity = (
                    gravity_torques[3] if len(gravity_torques) > 3 else 0.0
                )
                print(f"\nJoint 4 Details:")
                print(
                    f"  Position: {joint_4_state.get('pos', 0.0):.6f} rad "
                    f"({np.degrees(joint_4_state.get('pos', 0.0)):.3f} deg)"
                )
                print(f"  Velocity: {joint_4_state.get('vel', 0.0):.6f} rad/s")
                print(
                    f"  Torque: {joint_4_state.get('torq', 0.0):.6f} N⋅m  "
                    f"Gravity Compensation: {joint_4_gravity:.6f} N⋅m"
                )
                print(f"  Status: {joint_4_state.get('status', 'UNKNOWN')}")
                print(f"  MOSFET Temp: {joint_4_state.get('t_mos', 0.0):.1f} °C")
                print(f"  Rotor Temp: {joint_4_state.get('t_rotor', 0.0):.1f} °C")
                print(
                    f"  Motor ID: {joint_4_state.get('can_id', args.motor_ids[3])}"
                )

            # Maintain control loop rate
            elapsed = time.time() - loop_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\nGravity compensation stopped by user")
    except Exception as e:
        print(f"\nError during gravity compensation: {e}")
        import traceback

        traceback.print_exc()
    finally:
        robot.disconnect()
        print("\nDisconnected from hardware")


if __name__ == "__main__":
    main()


