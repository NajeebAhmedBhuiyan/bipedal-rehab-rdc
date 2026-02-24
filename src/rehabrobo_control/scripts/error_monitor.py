#!/usr/bin/env python3
"""
error_monitor.py — Phase 3 Fixed: Clean Error Signal for RDC

KEY FIX: Now computes error as:
  error = nominal - disturbed_commands   (NOT nominal - joint_states)

This gives the RDC exactly the same error signal it was trained on:
  error ≈ -disturbance

Previously it used joint_states which included the RDC's own output,
creating a feedback loop that made the model compensate itself.

Subscribes to:
  /nominal_commands    — desired clean trajectory
  /disturbed_commands  — nominal + disturbance (from command_mixer)

Publishes:
  /joint_tracking_error — 6 error values (rad) → fed to RDC node

ROS2 Parameters:
  session_label    : str   — label for CSV log  (default: 'session')
  log_dir          : str   — log folder         (default: ~/rehabrobo_logs)
  publish_rate_hz  : float — publish rate       (default: 20.0)
  verbose          : bool  — print status       (default: False)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

import csv
import math
import os
from datetime import datetime


JOINT_NAMES = [
    'right_hip_joint',  'right_knee_joint',  'right_ankle_joint',
    'left_hip_joint',   'left_knee_joint',   'left_ankle_joint',
]
NUM_JOINTS = len(JOINT_NAMES)


class ErrorMonitor(Node):
    def __init__(self):
        super().__init__('error_monitor')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('session_label',  'session')
        self.declare_parameter('log_dir',        os.path.expanduser('~/rehabrobo_logs'))
        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('verbose',         False)

        session_label   = self.get_parameter('session_label').get_parameter_value().string_value
        log_dir         = self.get_parameter('log_dir').get_parameter_value().string_value
        publish_rate_hz = self.get_parameter('publish_rate_hz').get_parameter_value().double_value

        # ── CSV setup ─────────────────────────────────────────────────────────
        os.makedirs(log_dir, exist_ok=True)
        ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'error_monitor_{session_label}_{ts}.csv'
        self.csv_path   = os.path.join(log_dir, filename)
        self.csv_file   = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            ['time_sec'] +
            [f'nominal_{n}' for n in JOINT_NAMES] +
            [f'disturbed_{n}' for n in JOINT_NAMES] +
            [f'error_{n}' for n in JOINT_NAMES] +
            [f'abs_error_{n}' for n in JOINT_NAMES] +
            ['rmse']
        )

        # ── Internal state ────────────────────────────────────────────────────
        self.nominal   = [0.0] * NUM_JOINTS
        self.disturbed = [0.0] * NUM_JOINTS

        self.nominal_received   = False
        self.disturbed_received = False

        self.start_time     = None
        self.row_count      = 0
        self._last_print    = -1.0

        # Welford online statistics for normalization reference
        self._n    = [0]   * NUM_JOINTS
        self._mean = [0.0] * NUM_JOINTS
        self._M2   = [0.0] * NUM_JOINTS

        # ── Subscribers ───────────────────────────────────────────────────────
        self.sub_nominal = self.create_subscription(
            Float64MultiArray, '/nominal_commands',
            self.nominal_callback, 10
        )
        self.sub_disturbed = self.create_subscription(
            Float64MultiArray, '/disturbed_commands',
            self.disturbed_callback, 10
        )

        # ── Publisher ─────────────────────────────────────────────────────────
        self.pub_error = self.create_publisher(
            Float64MultiArray,
            '/joint_tracking_error',
            10
        )

        self.timer = self.create_timer(1.0 / publish_rate_hz, self.monitor_callback)

        self.get_logger().info(
            'ErrorMonitor (Phase 3 Fixed) started.\n'
            '  Computes: error = /nominal_commands - /disturbed_commands\n'
            f'  Logging to: {self.csv_path}\n'
            '  Waiting for topics...'
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Subscribers
    # ──────────────────────────────────────────────────────────────────────────

    def nominal_callback(self, msg):
        if len(msg.data) != NUM_JOINTS:
            return
        self.nominal = list(msg.data)
        if not self.nominal_received:
            self.nominal_received = True
            self.get_logger().info('  /nominal_commands    ✓')

    def disturbed_callback(self, msg):
        if len(msg.data) != NUM_JOINTS:
            return
        self.disturbed = list(msg.data)
        if not self.disturbed_received:
            self.disturbed_received = True
            self.get_logger().info('  /disturbed_commands  ✓')
            self.get_logger().info('  All topics live — monitoring started ✓')

    # ──────────────────────────────────────────────────────────────────────────
    # Monitor callback
    # ──────────────────────────────────────────────────────────────────────────

    def monitor_callback(self):
        if not (self.nominal_received and self.disturbed_received):
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if self.start_time is None:
            self.start_time = now

        t = now - self.start_time

        # ── Compute error ─────────────────────────────────────────────────────
        # error = nominal - disturbed ≈ -disturbance
        # This matches exactly what the model was trained on
        errors     = [self.nominal[i] - self.disturbed[i] for i in range(NUM_JOINTS)]
        abs_errors = [abs(e) for e in errors]
        rmse       = math.sqrt(sum(e**2 for e in errors) / NUM_JOINTS)

        # ── Welford online stats ──────────────────────────────────────────────
        for i, e in enumerate(errors):
            self._n[i] += 1
            delta = e - self._mean[i]
            self._mean[i] += delta / self._n[i]
            self._M2[i]   += delta * (e - self._mean[i])

        # ── Publish error ─────────────────────────────────────────────────────
        msg = Float64MultiArray()
        msg.data = errors
        self.pub_error.publish(msg)

        # ── CSV log ───────────────────────────────────────────────────────────
        row = (
            [f'{t:.4f}'] +
            [f'{v:.6f}' for v in self.nominal] +
            [f'{v:.6f}' for v in self.disturbed] +
            [f'{v:.6f}' for v in errors] +
            [f'{v:.6f}' for v in abs_errors] +
            [f'{rmse:.6f}']
        )
        self.csv_writer.writerow(row)
        self.row_count += 1

        # ── Verbose print every 5 seconds ─────────────────────────────────────
        verbose = self.get_parameter('verbose').get_parameter_value().bool_value
        if verbose and (t - self._last_print) >= 5.0:
            self._last_print = t
            self.get_logger().info(
                f'\n  ── ErrorMonitor @ t={t:.1f}s (rows={self.row_count}) ──\n'
                f'  {"Joint":<25} {"Error(rad)":>12} {"|Error|(rad)":>14}\n'
                f'  ' + '-' * 54 + '\n' +
                '\n'.join([
                    f'  {JOINT_NAMES[i]:<25} '
                    f'{errors[i]:>12.6f} '
                    f'{abs_errors[i]:>14.6f}'
                    for i in range(NUM_JOINTS)
                ]) +
                f'\n  ' + '-' * 54 +
                f'\n  RMSE: {rmse:.6f} rad ({math.degrees(rmse):.4f} deg)\n'
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────────────────

    def destroy_node(self):
        self.csv_file.flush()
        self.csv_file.close()
        self.get_logger().info(
            f'ErrorMonitor shut down. Rows written: {self.row_count}\n'
            f'  CSV: {self.csv_path}'
        )
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ErrorMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
