#!/usr/bin/env python3
"""
ml_data_collector.py — Phase 3 v2: Window Size 30

KEY CHANGE: window_size increased from 10 → 30 steps (1.5 seconds of history)
This gives the LSTM enough context to determine the phase of a spasticity
burst, which lasts ~1.5 seconds. Previously at 10 steps (0.5s), the model
could not see a full burst cycle and was guessing phase incorrectly.

Feature vector is now 187 values (was 67):
  6  nominal commands
  180 error window (30 steps × 6 joints)
  1  gait phase
  ─────────────────
  187 total

Output labels unchanged: 6 compensation values

Error computation (unchanged from v1):
  error = /nominal_commands - /disturbed_commands = -disturbance

Subscribes to:
  /nominal_commands       — desired joint positions
  /disturbed_commands     — nominal + disturbance (clean error signal)
  /disturbance            — raw disturbance (training label)

ROS2 Parameters:
  session_label    : str   — label for CSV filename     (default: 'session')
  disturbance_type : str   — logged as metadata         (default: 'spasticity')
  magnitude        : float — logged as metadata         (default: 0.15)
  log_dir          : str   — output folder              (default: ~/rehabrobo_logs/ml_data)
  window_size      : int   — number of past error steps (default: 30)
  gait_frequency   : float — must match walking_publisher (default: 0.25 Hz)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

import csv
import math
import os
from collections import deque
from datetime import datetime


JOINT_NAMES = [
    'right_hip_joint',
    'right_knee_joint',
    'right_ankle_joint',
    'left_hip_joint',
    'left_knee_joint',
    'left_ankle_joint',
]

NUM_JOINTS  = len(JOINT_NAMES)
WINDOW_SIZE = 30   # ← increased from 10


class MLDataCollector(Node):
    def __init__(self):
        super().__init__('ml_data_collector')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('session_label',    'session')
        self.declare_parameter('disturbance_type', 'spasticity')
        self.declare_parameter('magnitude',         0.15)
        self.declare_parameter('log_dir',           os.path.expanduser('~/rehabrobo_logs/ml_data'))
        self.declare_parameter('window_size',       WINDOW_SIZE)
        self.declare_parameter('gait_frequency',    0.25)

        session_label    = self.get_parameter('session_label').get_parameter_value().string_value
        disturbance_type = self.get_parameter('disturbance_type').get_parameter_value().string_value
        magnitude        = self.get_parameter('magnitude').get_parameter_value().double_value
        log_dir          = self.get_parameter('log_dir').get_parameter_value().string_value
        self.window_size = self.get_parameter('window_size').get_parameter_value().integer_value
        self.gait_freq   = self.get_parameter('gait_frequency').get_parameter_value().double_value

        # Feature dim = 6 nominal + (window_size × 6 errors) + 1 phase
        self.feature_dim = 6 + (self.window_size * NUM_JOINTS) + 1
        # e.g. window_size=30 → 6 + 180 + 1 = 187

        # ── CSV setup ─────────────────────────────────────────────────────────
        os.makedirs(log_dir, exist_ok=True)
        ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{session_label}_{ts}.csv'
        self.csv_path = os.path.join(log_dir, filename)

        self.csv_file   = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self._write_header()

        # ── Internal state ────────────────────────────────────────────────────
        self.nominal     = [0.0] * NUM_JOINTS
        self.disturbed   = [0.0] * NUM_JOINTS
        self.disturbance = [0.0] * NUM_JOINTS

        self.nominal_received     = False
        self.disturbed_received   = False
        self.disturbance_received = False

        self.error_window = deque(
            [[0.0] * NUM_JOINTS for _ in range(self.window_size)],
            maxlen=self.window_size
        )

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
        self.sub_disturbance = self.create_subscription(
            Float64MultiArray, '/disturbance',
            self.disturbance_callback, 10
        )

        self.timer = self.create_timer(0.05, self.collect_callback)

        self.get_logger().info('=' * 60)
        self.get_logger().info('  ML Data Collector v2 (window_size=30)')
        self.get_logger().info(f'  Session      : {session_label}')
        self.get_logger().info(f'  Disturbance  : {disturbance_type} @ {magnitude:.2f} rad')
        self.get_logger().info(f'  Window size  : {self.window_size} steps ({self.window_size*0.05:.1f}s history)')
        self.get_logger().info(f'  Feature dim  : {self.feature_dim} (was 67)')
        self.get_logger().info(f'  Output CSV   : {self.csv_path}')
        self.get_logger().info('=' * 60)
        self.get_logger().info('Waiting for all 3 topics...')

    # ──────────────────────────────────────────────────────────────────────────
    # Subscribers
    # ──────────────────────────────────────────────────────────────────────────

    def nominal_callback(self, msg):
        if len(msg.data) != NUM_JOINTS:
            return
        self.nominal = list(msg.data)
        if not self.nominal_received:
            self.nominal_received = True
            self.get_logger().info('  /nominal_commands      ✓')
            self._check_all_ready()

    def disturbed_callback(self, msg):
        if len(msg.data) != NUM_JOINTS:
            return
        self.disturbed = list(msg.data)
        if not self.disturbed_received:
            self.disturbed_received = True
            self.get_logger().info('  /disturbed_commands    ✓')
            self._check_all_ready()

    def disturbance_callback(self, msg):
        if len(msg.data) != NUM_JOINTS:
            return
        self.disturbance = list(msg.data)
        if not self.disturbance_received:
            self.disturbance_received = True
            self.get_logger().info('  /disturbance           ✓')
            self._check_all_ready()

    def _check_all_ready(self):
        if (self.nominal_received and
            self.disturbed_received and
            self.disturbance_received):
            self.get_logger().info('All topics live — RECORDING STARTED ✓')

    # ──────────────────────────────────────────────────────────────────────────
    # Main collection callback (20 Hz)
    # ──────────────────────────────────────────────────────────────────────────

    def collect_callback(self):
        if not (self.nominal_received and
                self.disturbed_received and
                self.disturbance_received):
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if self.start_time is None:
            self.start_time = now

        t = now - self.start_time

        # ── Compute error ──────────────────────────────────────────────────────
        errors = [self.nominal[i] - self.disturbed[i] for i in range(NUM_JOINTS)]
        rmse   = math.sqrt(sum(e ** 2 for e in errors) / NUM_JOINTS)

        # ── Gait phase ─────────────────────────────────────────────────────────
        gait_phase = (2 * math.pi * self.gait_freq * t) % (2 * math.pi)

        # ── Training label ─────────────────────────────────────────────────────
        compensation = [-d for d in self.disturbance]

        # ── Build window features (oldest first) ───────────────────────────────
        window_flat = []
        for past_errors in self.error_window:
            window_flat.extend(past_errors)

        # ── Update window ──────────────────────────────────────────────────────
        self.error_window.append(errors[:])

        # ── Write CSV row ──────────────────────────────────────────────────────
        row = [f'{t:.4f}']
        row += [f'{v:.6f}' for v in self.nominal]      # 6
        row += [f'{v:.6f}' for v in window_flat]        # window_size × 6
        row.append(f'{gait_phase:.6f}')                 # 1
        row += [f'{v:.6f}' for v in compensation]       # 6 labels
        row += [f'{v:.6f}' for v in self.disturbed]     # 6 diagnostics
        row += [f'{v:.6f}' for v in self.disturbance]   # 6 diagnostics
        row.append(f'{rmse:.6f}')                       # 1 diagnostic

        self.csv_writer.writerow(row)
        self.row_count += 1

        # ── Console status every 30 seconds ────────────────────────────────────
        if (t - self._last_print) >= 30.0:
            self._last_print = t
            self._print_status(t, errors, rmse, compensation)

    # ──────────────────────────────────────────────────────────────────────────
    # Console status
    # ──────────────────────────────────────────────────────────────────────────

    def _print_status(self, t, errors, rmse, compensation):
        mins = int(t) // 60
        secs = int(t) % 60
        self.get_logger().info(
            f'\n'
            f'  ── ML Collector @ {mins:02d}:{secs:02d} | Rows: {self.row_count} ──\n'
            f'  {"Joint":<22} {"Error (rad)":>12} {"Compensation":>14}\n'
            f'  ' + '-' * 50 + '\n' +
            '\n'.join([
                f'  {JOINT_NAMES[i]:<22} '
                f'{errors[i]:>12.6f} '
                f'{compensation[i]:>14.6f}'
                for i in range(NUM_JOINTS)
            ]) +
            f'\n  ' + '-' * 50 +
            f'\n  RMSE: {rmse:.6f} rad ({math.degrees(rmse):.4f} deg)\n'
        )

    # ──────────────────────────────────────────────────────────────────────────
    # CSV Header
    # ──────────────────────────────────────────────────────────────────────────

    def _write_header(self):
        header = ['time_sec']

        # Nominal commands (6)
        for n in JOINT_NAMES:
            header.append(f'nominal_{n}')

        # Error window (window_size × 6)
        for step in range(self.window_size - 1, -1, -1):
            for n in JOINT_NAMES:
                header.append(f'error_{n}_tminus{step}')

        # Gait phase (1)
        header.append('gait_phase')

        # Compensation labels (6)
        for n in JOINT_NAMES:
            header.append(f'compensation_{n}')

        # Diagnostics (13)
        for n in JOINT_NAMES:
            header.append(f'disturbed_{n}')
        for n in JOINT_NAMES:
            header.append(f'disturbance_{n}')
        header.append('rmse')

        self.csv_writer.writerow(header)

        total_cols = 1 + 6 + (self.window_size * NUM_JOINTS) + 1 + 6 + 6 + 6 + 1
        self.get_logger().info(
            f'CSV header: 1 time + {self.feature_dim} features + '
            f'6 labels + 13 diagnostics = {total_cols} columns'
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────────────────

    def destroy_node(self):
        self.csv_file.flush()
        self.csv_file.close()
        elapsed = (self.get_clock().now().nanoseconds / 1e9) - (self.start_time or 0)
        mins = int(elapsed) // 60
        self.get_logger().info(
            f'\n'
            f'  ── ML Collector shut down ──\n'
            f'  Duration     : ~{mins} minutes\n'
            f'  Rows written : {self.row_count}\n'
            f'  CSV saved at : {self.csv_path}\n'
        )
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MLDataCollector()
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