#!/usr/bin/env python3

import subprocess
import os
import time
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))


def check_can_interface(interface):
    """Check if a CAN interface exists and is available"""
    try:
        # Check if interface exists in network interfaces
        result = subprocess.run(['ip', 'link', 'show', interface],
                              capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return False

        # Check if interface is UP
        if 'state UP' in result.stdout or 'state UNKNOWN' in result.stdout:
            return True
        else:
            print(f"Warning: CAN interface {interface} exists but is not UP")
            return False

    except Exception as e:
        print(f"Error checking CAN interface {interface}: {e}")
        return False


def check_can_interfaces(follower_channel, leader_channel):
    """Check if required CAN interfaces exist"""
    required_interfaces = [follower_channel, leader_channel]
    missing_interfaces = []

    for interface in required_interfaces:
        if not check_can_interface(interface):
            missing_interfaces.append(interface)

    if missing_interfaces:
        raise RuntimeError(f"Missing or unavailable CAN interfaces: {', '.join(missing_interfaces)}")

    print(f"✓ All CAN interfaces are available: {', '.join(required_interfaces)}")
    return True


def launch_gello_process(can_channel, gripper, mode=None, server_port=None, bilateral_kp=None):
    """Launch a single gello process with given parameters"""
    python_path = "python"
    script_path = os.path.join(current_file_path, "minimum_gello.py")

    cmd = [python_path, os.path.expanduser(script_path),
           "--can_channel", can_channel,
           "--gripper", gripper]

    if mode:
        cmd.extend(["--mode", mode])

    if server_port:
        cmd.extend(["--server_port", str(server_port)])

    if bilateral_kp is not None:
        cmd.extend(["--bilateral_kp", str(bilateral_kp)])

    print(f"Starting: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(cmd)
        return process
    except Exception as e:
        print(f"Error starting process for {can_channel}: {e}")
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Launch a single leader-follower arm setup")
    parser.add_argument("--follower_channel", type=str, default="can_follower_r",
                       help="CAN channel for the follower arm (default: can_follower_r)")
    parser.add_argument("--leader_channel", type=str, default="can_leader_r",
                       help="CAN channel for the leader arm (default: can_leader_r)")
    parser.add_argument("--follower_gripper", type=str, default="linear_4310",
                       choices=["crank_4310", "linear_3507", "linear_4310", "yam_teaching_handle", "no_gripper"],
                       help="Gripper type for follower arm (default: linear_4310)")
    parser.add_argument("--server_port", type=int, default=11333,
                       help="Server port for communication (default: 11333)")
    parser.add_argument("--bilateral_kp", type=float, default=0.2,
                       help="Bilateral force feedback gain (default: 0.2)")

    args = parser.parse_args()

    processes = []

    try:
        # Check if CAN interfaces exist
        print("Checking CAN interfaces...")
        check_can_interfaces(args.follower_channel, args.leader_channel)

        # Define the processes to launch
        process_configs = [
            {
                'can_channel': args.follower_channel,
                'gripper': args.follower_gripper,
                'mode': 'follower',
                'server_port': args.server_port,
            },
            {
                'can_channel': args.leader_channel,
                'gripper': 'yam_teaching_handle',
                'mode': 'leader',
                'server_port': args.server_port,
                'bilateral_kp': args.bilateral_kp,
            }
        ]

        # Launch all processes
        print("\nLaunching processes...")
        for config in process_configs:
            process = launch_gello_process(**config)
            if process:
                processes.append(process)
                print(f"✓ Started process {process.pid} for {config['can_channel']} ({config.get('mode', 'follower')})")
            else:
                raise RuntimeError(f"Failed to start process for {config['can_channel']}")

        print(f"\n✓ Successfully launched {len(processes)} processes")
        print("Press Ctrl+C to stop all processes")
        print("\nUsage:")
        print("  - Press the top button on the teaching handle to toggle synchronization")
        print("  - When synchronized, the follower arm will track the leader arm")

        # Wait for processes and handle termination
        try:
            while True:
                # Check if any process has died
                for i, process in enumerate(processes):
                    if process.poll() is not None:
                        print(f"Process {process.pid} has terminated")
                        processes.pop(i)
                        break

                if not processes:
                    print("All processes have terminated")
                    break

                time.sleep(1)

        except KeyboardInterrupt:
            print("\nReceived Ctrl+C, terminating all processes...")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    finally:
        # Clean up: terminate all running processes
        for process in processes:
            try:
                print(f"Terminating process {process.pid}...")
                process.terminate()

                # Wait up to 5 seconds for graceful termination
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"Force killing process {process.pid}...")
                    process.kill()
                    process.wait()

            except Exception as e:
                print(f"Error terminating process {process.pid}: {e}")

        print("All processes terminated")


if __name__ == "__main__":
    main()

