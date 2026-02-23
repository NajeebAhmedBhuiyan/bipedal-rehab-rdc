#!/usr/bin/env python3
"""
ml_data_collector.py — Phase 2, Step 5: ML Training Data Collector

The single node that captures EVERYTHING the LSTM needs in one CSV.
Replaces running error_monitor separately for training data purposes.

Subscribes to:
  /nominal_commands       — desired joint positions (input feature)
  /joint_states           — actual joint positions  (to compute error)
  /disturbance            — disturbance signal      (training LABEL)

Records per timestep:
  INPUT FEATURES:
  ├── nominal_commands[t]     — 6 values (what sinusoid wanted)
  ├── joint_errors[t..t-9]   — sliding window of last 10 errors (60 values)
  └── gait_phase[t]           — phase of gait cycle in radians (0 to 2π)

  OUTPUT LABEL:
  └── compensation[t]         — -disturbance[t] per joint (6 values)
                                 this is what the LSTM must learn to predict

  DIAGNOSTICS (not used in training, useful for paper plots):
  ├── actual_positions[t]     — 6 values
  ├── disturbance[t]          — raw disturbance (before negation)
  └── rmse[t]                 — instantaneous RMSE across joints

ROS2 Parameters:
  session_label        : str   — label for CSV filename     (default: 'session')
  disturbance_type     : str   — logged as metadata         (default: 'spasticity')
  magnitude            : float — logged as metadata         (default: 0.15)
  log_dir              : str   — output folder              (default: ~/rehabrobo_logs/ml_data)
  window_size          : int   — number of past error steps (default: 10)
  gait_frequency       : float — must match walking_publisher (default: 0.25 Hz)

Usage:
  ros2 run rehabrobo_control ml_data_collector.py --ros-args \
    -p session_label:=spasticity_mag015 \
    -p disturbance_type:=spasticity \
    -p magnitude:=0.15
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

import csv
import math
import os
from collections import deque
from datetime import datetime


# ── Constants ─────────────────────────────────────────────────────────────────

JOINT_NAMES = [
    'right_hip_joint',
    'right_knee_joint',
    'right_ankle_joint',
    'left_hip_joint',
    'left_knee_joint',
    'left_ankle_joint',
]

NUM_JOINTS  = len(JOINT_NAMES)
WINDOW_SIZE = 10   # default sliding window — overridden by ROS2 param


# ── ML Data Collector Node ────────────────────────────────────────────────────

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
        self.actual      = {name: 0.0 for name in JOINT_NAMES}
        self.disturbance = [0.0] * NUM_JOINTS

        self.nominal_received     = False
        self.actual_received      = False
        self.disturbance_received = False

        # Sliding window of past errors — deque auto-drops oldest
        # Initialised to zeros so window is always full from row 1
        self.error_window = deque(
            [[0.0] * NUM_JOINTS for _ in range(self.window_size)],
            maxlen=self.window_size
        )

        self.start_time = None
        self.row_count  = 0

        # Console status throttle
        self._last_print = -1.0

        # ── Subscribers ───────────────────────────────────────────────────────
        self.sub_nominal = self.create_subscription(
            Float64MultiArray, '/nominal_commands',
            self.nominal_callback, 10
        )
        self.sub_actual = self.create_subscription(
            JointState, '/joint_states',
            self.actual_callback, 10
        )
        self.sub_disturbance = self.create_subscription(
            Float64MultiArray, '/disturbance',
            self.disturbance_callback, 10
        )

        # ── Timer at 20 Hz ────────────────────────────────────────────────────
        self.timer = self.create_timer(0.05, self.collect_callback)

        # ── Startup message ───────────────────────────────────────────────────
        self.get_logger().info('=' * 55)
        self.get_logger().info('  ML Data Collector started')
        self.get_logger().info(f'  Session      : {session_label}')
        self.get_logger().info(f'  Disturbance  : {disturbance_type} @ {magnitude:.2f} rad')
        self.get_logger().info(f'  Window size  : {self.window_size} steps')
        self.get_logger().info(f'  Output CSV   : {self.csv_path}')
        self.get_logger().info('=' * 55)
        self.get_logger().info('Waiting for all 3 topics...')

    # ──────────────────────────────────────────────────────────────────────────
    # Subscribers
    # ──────────────────────────────────────────────────────────────────────────

    def nominal_callback(self, msg: Float64MultiArray):
        if len(msg.data) != NUM_JOINTS:
            return
        self.nominal = list(msg.data)
        if not self.nominal_received:
            self.nominal_received = True
            self.get_logger().info('  /nominal_commands      ✓')
            self._check_all_ready()

    def actual_callback(self, msg: JointState):
        for i, name in enumerate(msg.name):
            if name in self.actual:
                self.actual[name] = msg.position[i]
        if not self.actual_received:
            self.actual_received = True
            self.get_logger().info('  /joint_states          ✓')
            self._check_all_ready()

    def disturbance_callback(self, msg: Float64MultiArray):
        if len(msg.data) != NUM_JOINTS:
            return
        self.disturbance = list(msg.data)
        if not self.disturbance_received:
            self.disturbance_received = True
            self.get_logger().info('  /disturbance           ✓')
            self._check_all_ready()

    def _check_all_ready(self):
        if self.nominal_received and self.actual_received and self.disturbance_received:
            self.get_logger().info('All topics live — RECORDING STARTED ✓')

    # ──────────────────────────────────────────────────────────────────────────
    # Main collection callback
    # ──────────────────────────────────────────────────────────────────────────

    def collect_callback(self):
        if not (self.nominal_received and
                self.actual_received and
                self.disturbance_received):
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if self.start_time is None:
            self.start_time = now

        t = now - self.start_time

        # ── Compute current error ─────────────────────────────────────────────
        actuals = [self.actual[name] for name in JOINT_NAMES]
        errors  = [self.nominal[i] - actuals[i] for i in range(NUM_JOINTS)]
        rmse    = math.sqrt(sum(e ** 2 for e in errors) / NUM_JOINTS)

        # ── Gait phase (0 to 2π) ──────────────────────────────────────────────
        # Derived from time and gait frequency — same formula as walking_publisher
        gait_phase = (2 * math.pi * self.gait_freq * t) % (2 * math.pi)

        # ── Training label: compensation = -disturbance ───────────────────────
        compensation = [-d for d in self.disturbance]

        # ── Build the sliding window feature (oldest first) ───────────────────
        # window[0] = error at t-9, window[9] = error at t (most recent)
        window_flat = []
        for past_errors in self.error_window:     # oldest → newest
            window_flat.extend(past_errors)       # 10 × 6 = 60 values

        # ── Update the window with current error ──────────────────────────────
        self.error_window.append(errors[:])       # push current, drop oldest

        # ── Write CSV row ─────────────────────────────────────────────────────
        row = [f'{t:.4f}']

        # Input features
        row += [f'{v:.6f}' for v in self.nominal]       # nominal_commands[t]  — 6
        row += [f'{v:.6f}' for v in window_flat]        # error window          — 60
        row.append(f'{gait_phase:.6f}')                 # gait_phase[t]         — 1

        # Output label
        row += [f'{v:.6f}' for v in compensation]       # compensation[t]       — 6

        # Diagnostics
        row += [f'{v:.6f}' for v in actuals]            # actual_positions      — 6
        row += [f'{v:.6f}' for v in self.disturbance]   # raw disturbance       — 6
        row.append(f'{rmse:.6f}')                       # rmse                  — 1

        self.csv_writer.writerow(row)
        self.row_count += 1

        # ── Console status every 30 seconds ───────────────────────────────────
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
            f'  {"Joint":<12} {"Error (rad)":>12} {"Compensation":>14}\n'
            f'  ' + '-' * 42 + '\n' +
            '\n'.join([
                f'  {JOINT_NAMES[i]:<12} '
                f'{errors[i]:>12.6f} '
                f'{compensation[i]:>14.6f}'
                for i in range(NUM_JOINTS)
            ]) +
            f'\n  ' + '-' * 42 +
            f'\n  RMSE: {rmse:.6f} rad ({math.degrees(rmse):.4f} deg)\n'
        )

    # ──────────────────────────────────────────────────────────────────────────
    # CSV Header
    # ──────────────────────────────────────────────────────────────────────────

    def _write_header(self):
        header = ['time_sec']

        # Input features
        for n in JOINT_NAMES:
            header.append(f'nominal_{n}')                   # 6 cols

        for step in range(self.window_size - 1, -1, -1):    # t-9 ... t-0
            for n in JOINT_NAMES:
                header.append(f'error_{n}_tminus{step}')   # 60 cols

        header.append('gait_phase')                         # 1 col

        # Output labels
        for n in JOINT_NAMES:
            header.append(f'compensation_{n}')              # 6 cols

        # Diagnostics
        for n in JOINT_NAMES:
            header.append(f'actual_{n}')                    # 6 cols
        for n in JOINT_NAMES:
            header.append(f'disturbance_{n}')               # 6 cols
        header.append('rmse')                               # 1 col

        self.csv_writer.writerow(header)

        total_input  = 6 + 60 + 1   # = 67
        total_output = 6
        total_diag   = 6 + 6 + 1    # = 13
        self.get_logger().info(
            f'CSV columns: {1} time + {total_input} inputs + '
            f'{total_output} labels + {total_diag} diagnostics '
            f'= {1 + total_input + total_output + total_diag} total'
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────────────────

    def destroy_node(self):
        self.csv_file.flush()
        self.csv_file.close()
        mins = int((self.get_clock().now().nanoseconds / 1e9 -
                    (self.start_time or 0))) // 60
        self.get_logger().info(
            f'\n'
            f'  ── ML Collector shut down ──\n'
            f'  Duration     : ~{mins} minutes\n'
            f'  Rows written : {self.row_count}\n'
            f'  CSV saved at : {self.csv_path}\n'
        )
        super().destroy_node()


# ── Entry point ───────────────────────────────────────────────────────────────

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
