#!/usr/bin/env python3
"""
data_logger.py — Step 1: Baseline Data Logger

Subscribes to:
  /joint_states                          — actual joint positions (from hardware/mock)
  /forward_position_controller/commands  — desired joint positions (from walking_publisher)

Logs every timestep to a CSV file:
  timestamp, desired[6], actual[6], error[6]

Usage:
  ros2 run rehabrobo_control data_logger.py
  ros2 run rehabrobo_control data_logger.py --ros-args -p session_label:=trial_01

Output:
  ~/rehabrobo_logs/<session_label>_<datetime>.csv
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

import csv
import os
import math
from datetime import datetime


# Joint names must match the order in robot_controller.yaml
JOINT_NAMES = [
    'right_hip_joint',
    'right_knee_joint',
    'right_ankle_joint',
    'left_hip_joint',
    'left_knee_joint',
    'left_ankle_joint',
]

NUM_JOINTS = len(JOINT_NAMES)


class DataLogger(Node):
    def __init__(self):
        super().__init__('data_logger')

        # ── ROS2 parameter: session label for naming the CSV ──────────────────
        self.declare_parameter('session_label', 'baseline')
        self.declare_parameter('log_dir', os.path.expanduser('~/rehabrobo_logs'))

        session_label = self.get_parameter('session_label').get_parameter_value().string_value
        log_dir = self.get_parameter('log_dir').get_parameter_value().string_value

        # ── Prepare output directory and CSV file ─────────────────────────────
        os.makedirs(log_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{session_label}_{timestamp_str}.csv'
        self.csv_path = os.path.join(log_dir, filename)

        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # Write header row
        header = ['ros_time_sec']
        for name in JOINT_NAMES:
            header.append(f'desired_{name}')
        for name in JOINT_NAMES:
            header.append(f'actual_{name}')
        for name in JOINT_NAMES:
            header.append(f'error_{name}')
        header.append('rmse_all_joints')
        self.csv_writer.writerow(header)

        # ── Internal state ────────────────────────────────────────────────────
        # Latest desired positions from the walking publisher
        # Initialise to zeros — will update once first message arrives
        self.desired_positions = [0.0] * NUM_JOINTS
        self.desired_received = False

        # Latest actual positions indexed by joint name (from /joint_states)
        self.actual_positions = {name: 0.0 for name in JOINT_NAMES}
        self.actual_received = False

        self.start_time = None  # set on first log entry
        self.row_count = 0

        # ── Subscribers ───────────────────────────────────────────────────────
        self.sub_desired = self.create_subscription(
            Float64MultiArray,
            '/forward_position_controller/commands',
            self.desired_callback,
            10
        )

        self.sub_actual = self.create_subscription(
            JointState,
            '/joint_states',
            self.actual_callback,
            10
        )

        # ── Logging timer: runs at 20 Hz to match walking_publisher ──────────
        self.log_timer = self.create_timer(0.05, self.log_callback)

        self.get_logger().info(
            f'DataLogger started. Writing to: {self.csv_path}'
        )
        self.get_logger().info(
            'Waiting for /forward_position_controller/commands and /joint_states ...'
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────────

    def desired_callback(self, msg: Float64MultiArray):
        """Receive the commanded positions from walking_publisher."""
        if len(msg.data) != NUM_JOINTS:
            self.get_logger().warn(
                f'Expected {NUM_JOINTS} values in /commands, got {len(msg.data)}. Skipping.'
            )
            return
        self.desired_positions = list(msg.data)
        if not self.desired_received:
            self.desired_received = True
            self.get_logger().info('Receiving desired commands ✓')

    def actual_callback(self, msg: JointState):
        """Receive actual joint positions from joint_state_broadcaster."""
        # /joint_states may include joints in any order — index by name
        for i, name in enumerate(msg.name):
            if name in self.actual_positions:
                self.actual_positions[name] = msg.position[i]

        if not self.actual_received:
            self.actual_received = True
            self.get_logger().info('Receiving actual joint states ✓')

    def log_callback(self):
        """Write one row to CSV if both data sources are ready."""
        if not (self.desired_received and self.actual_received):
            return  # quietly wait until both topics arrive

        now = self.get_clock().now().nanoseconds / 1e9

        if self.start_time is None:
            self.start_time = now
            self.get_logger().info('Logging started.')

        t = now - self.start_time

        # Build ordered actual list to match JOINT_NAMES order
        actuals = [self.actual_positions[name] for name in JOINT_NAMES]

        # Compute per-joint errors
        errors = [
            self.desired_positions[i] - actuals[i]
            for i in range(NUM_JOINTS)
        ]

        # Compute RMSE across all joints for this timestep
        rmse = math.sqrt(sum(e ** 2 for e in errors) / NUM_JOINTS)

        row = [f'{t:.4f}']
        row += [f'{v:.6f}' for v in self.desired_positions]
        row += [f'{v:.6f}' for v in actuals]
        row += [f'{v:.6f}' for v in errors]
        row.append(f'{rmse:.6f}')

        self.csv_writer.writerow(row)
        self.row_count += 1

        # Print a brief console update every 100 rows (~5 seconds at 20 Hz)
        if self.row_count % 100 == 0:
            self.get_logger().info(
                f'[{t:.1f}s] Rows logged: {self.row_count} | RMSE: {rmse:.4f} rad'
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────────────────

    def destroy_node(self):
        self.csv_file.flush()
        self.csv_file.close()
        self.get_logger().info(
            f'Logger shut down. Total rows written: {self.row_count}'
        )
        self.get_logger().info(f'CSV saved at: {self.csv_path}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DataLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
