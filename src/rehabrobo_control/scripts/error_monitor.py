#!/usr/bin/env python3
"""
error_monitor.py — Phase 3 v3: Logs both disturbed and compensated signals

KEY ADDITION: Now subscribes to /forward_position_controller/commands to log
the compensated signal alongside the disturbed signal. This enables proper
Stage 2 vs Stage 3 comparison from CSV data alone.

CSV columns:
  time_sec
  nominal_*          — 6 nominal joint positions
  disturbed_*        — 6 disturbed positions (nominal + disturbance, no RDC)
  compensated_*      — 6 compensated positions (nominal + disturbance + RDC)
  error_*            — 6 disturbed errors  (nominal - disturbed)
  comp_error_*       — 6 compensated errors (nominal - compensated)
  abs_error_*        — 6 absolute disturbed errors
  abs_comp_error_*   — 6 absolute compensated errors
  dist_rmse          — overall disturbed RMSE
  comp_rmse          — overall compensated RMSE
  improvement_pct    — (dist_rmse - comp_rmse) / dist_rmse × 100

Subscribes to:
  /nominal_commands                        — desired positions
  /disturbed_commands                      — nominal + disturbance (no RDC)
  /forward_position_controller/commands    — actual robot command (with RDC)

Publishes:
  /joint_tracking_error  — error for RDC node to consume

ROS2 Parameters:
  session_label    : str   — label for CSV filename
  publish_rate_hz  : float — publishing rate
  verbose          : bool  — extra console output
  log_dir          : str   — output folder
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

import csv
import math
import os
from datetime import datetime


JOINT_NAMES = [
    'right_hip_joint',
    'right_knee_joint',
    'right_ankle_joint',
    'left_hip_joint',
    'left_knee_joint',
    'left_ankle_joint',
]
NUM_JOINTS = len(JOINT_NAMES)


class ErrorMonitor(Node):
    def __init__(self):
        super().__init__('error_monitor')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('session_label',   'phase3_live')
        self.declare_parameter('publish_rate_hz',  20.0)
        self.declare_parameter('verbose',          False)
        self.declare_parameter('log_dir',          os.path.expanduser('~/rehabrobo_logs'))

        session_label    = self.get_parameter('session_label').get_parameter_value().string_value
        publish_rate_hz  = self.get_parameter('publish_rate_hz').get_parameter_value().double_value
        self.verbose     = self.get_parameter('verbose').get_parameter_value().bool_value
        log_dir          = self.get_parameter('log_dir').get_parameter_value().string_value

        # ── CSV setup ─────────────────────────────────────────────────────────
        os.makedirs(log_dir, exist_ok=True)
        ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'error_monitor_{session_label}_{ts}.csv'
        self.csv_path = os.path.join(log_dir, filename)

        self.csv_file   = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self._write_header()

        # ── Internal state ────────────────────────────────────────────────────
        self.nominal     = [0.0] * NUM_JOINTS
        self.disturbed   = [0.0] * NUM_JOINTS
        self.compensated = [0.0] * NUM_JOINTS   # ← NEW

        self.nominal_received     = False
        self.disturbed_received   = False
        self.compensated_received = False

        self.start_time  = None
        self.row_count   = 0
        self._last_print = -1.0

        # ── Subscribers ───────────────────────────────────────────────────────
        self.sub_nominal = self.create_subscription(
            Float64MultiArray, '/nominal_commands',
            self.nominal_callback, 10
        )
        self.sub_disturbed = self.create_subscription(
            Float64MultiArray, '/disturbed_commands',
            self.disturbed_callback, 10
        )
        self.sub_compensated = self.create_subscription(
            Float64MultiArray, '/forward_position_controller/commands',
            self.compensated_callback, 10
        )

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_error = self.create_publisher(
            Float64MultiArray, '/joint_tracking_error', 10
        )

        self.timer = self.create_timer(1.0 / publish_rate_hz, self.monitor_callback)

        self.get_logger().info('=' * 55)
        self.get_logger().info('  Error Monitor v3 (logs disturbed + compensated)')
        self.get_logger().info(f'  Session : {session_label}')
        self.get_logger().info(f'  CSV     : {self.csv_path}')
        self.get_logger().info('=' * 55)
        self.get_logger().info('Waiting for topics...')

    # ──────────────────────────────────────────────────────────────────────────
    # Subscribers
    # ──────────────────────────────────────────────────────────────────────────

    def nominal_callback(self, msg):
        if len(msg.data) != NUM_JOINTS:
            return
        self.nominal = list(msg.data)
        if not self.nominal_received:
            self.nominal_received = True
            self.get_logger().info('  /nominal_commands                     ✓')

    def disturbed_callback(self, msg):
        if len(msg.data) != NUM_JOINTS:
            return
        self.disturbed = list(msg.data)
        if not self.disturbed_received:
            self.disturbed_received = True
            self.get_logger().info('  /disturbed_commands                   ✓')

    def compensated_callback(self, msg):
        if len(msg.data) != NUM_JOINTS:
            return
        self.compensated = list(msg.data)
        if not self.compensated_received:
            self.compensated_received = True
            self.get_logger().info('  /forward_position_controller/commands ✓')

    # ──────────────────────────────────────────────────────────────────────────
    # Main callback
    # ──────────────────────────────────────────────────────────────────────────

    def monitor_callback(self):
        if not (self.nominal_received and self.disturbed_received):
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if self.start_time is None:
            self.start_time = now
            self.get_logger().info('Recording started ✓')

        t = now - self.start_time

        # ── Disturbed error (nominal - disturbed) ──────────────────────────────
        dist_errors = [self.nominal[i] - self.disturbed[i] for i in range(NUM_JOINTS)]
        dist_rmse   = math.sqrt(sum(e**2 for e in dist_errors) / NUM_JOINTS)

        # ── Compensated error (nominal - compensated) ──────────────────────────
        comp_errors = [self.nominal[i] - self.compensated[i] for i in range(NUM_JOINTS)]
        comp_rmse   = math.sqrt(sum(e**2 for e in comp_errors) / NUM_JOINTS)

        # ── Improvement % ──────────────────────────────────────────────────────
        if dist_rmse > 1e-6:
            improvement = (dist_rmse - comp_rmse) / dist_rmse * 100.0
        else:
            improvement = 0.0

        # ── Publish tracking error for RDC node ───────────────────────────────
        error_msg = Float64MultiArray()
        error_msg.data = dist_errors
        self.pub_error.publish(error_msg)

        # ── Write CSV ─────────────────────────────────────────────────────────
        row = [f'{t:.4f}']
        row += [f'{v:.6f}' for v in self.nominal]
        row += [f'{v:.6f}' for v in self.disturbed]
        row += [f'{v:.6f}' for v in self.compensated]
        row += [f'{v:.6f}' for v in dist_errors]
        row += [f'{v:.6f}' for v in comp_errors]
        row += [f'{abs(v):.6f}' for v in dist_errors]
        row += [f'{abs(v):.6f}' for v in comp_errors]
        row.append(f'{dist_rmse:.6f}')
        row.append(f'{comp_rmse:.6f}')
        row.append(f'{improvement:.4f}')

        self.csv_writer.writerow(row)
        self.row_count += 1

        # ── Console status every 30s ───────────────────────────────────────────
        if (t - self._last_print) >= 30.0:
            self._last_print = t
            mins = int(t) // 60
            secs = int(t) % 60
            self.get_logger().info(
                f'\n  ── Error Monitor @ {mins:02d}:{secs:02d} | Rows: {self.row_count} ──\n'
                f'  Dist RMSE : {math.degrees(dist_rmse):.4f}°\n'
                f'  Comp RMSE : {math.degrees(comp_rmse):.4f}°\n'
                f'  Improvement: {improvement:+.2f}%'
            )

    # ──────────────────────────────────────────────────────────────────────────
    # CSV Header
    # ──────────────────────────────────────────────────────────────────────────

    def _write_header(self):
        header = ['time_sec']
        for n in JOINT_NAMES: header.append(f'nominal_{n}')
        for n in JOINT_NAMES: header.append(f'disturbed_{n}')
        for n in JOINT_NAMES: header.append(f'compensated_{n}')
        for n in JOINT_NAMES: header.append(f'error_{n}')
        for n in JOINT_NAMES: header.append(f'comp_error_{n}')
        for n in JOINT_NAMES: header.append(f'abs_error_{n}')
        for n in JOINT_NAMES: header.append(f'abs_comp_error_{n}')
        header.append('dist_rmse')
        header.append('comp_rmse')
        header.append('improvement_pct')
        self.csv_writer.writerow(header)
        self.get_logger().info(f'CSV: {len(header)} columns')

    # ──────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────────────────

    def destroy_node(self):
        self.csv_file.flush()
        self.csv_file.close()
        elapsed = (self.get_clock().now().nanoseconds / 1e9) - (self.start_time or 0)
        self.get_logger().info(
            f'\n  ── Error Monitor shut down ──\n'
            f'  Rows written : {self.row_count}\n'
            f'  CSV saved    : {self.csv_path}'
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