#!/usr/bin/env python3
"""
error_monitor.py — Step 3: Joint Tracking Error Monitor

Subscribes to:
  /nominal_commands        — desired joint positions from walking_publisher
                             (NOTE: in Step 4, walking_publisher will publish
                              to /nominal_commands instead of directly to
                              /forward_position_controller/commands)
  /joint_states            — actual joint positions from joint_state_broadcaster

Computes:
  - Per-joint tracking error  = desired - actual  (radians)
  - Per-joint absolute error  = |desired - actual|
  - RMSE across all joints at each timestep
  - Running mean and std of error per joint (online, no buffer needed)

Publishes:
  /joint_tracking_error    — Float64MultiArray (6 values, one per joint)
                             This is the PRIMARY INPUT to the ML model in Phase 2/3

Logs to CSV:
  ~/rehabrobo_logs/error_monitor_<label>_<datetime>.csv
  Columns: time, desired[6], actual[6], error[6], abs_error[6],
           rmse, running_mean_error[6], running_std_error[6]

ROS2 Parameters:
  session_label   : str   — label for the CSV filename  (default: 'monitor')
  log_dir         : str   — output directory            (default: ~/rehabrobo_logs)
  publish_rate_hz : float — how fast to publish error   (default: 20.0)
  verbose         : bool  — print per-joint stats every 5s (default: True)

Usage:
  ros2 run rehabrobo_control error_monitor.py
  ros2 run rehabrobo_control error_monitor.py --ros-args -p session_label:=disturbed_trial_01

Monitor live error in another terminal:
  ros2 topic echo /joint_tracking_error
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

import csv
import math
import os
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

JOINT_LABELS = [
    'R.Hip  ',
    'R.Knee ',
    'R.Ankle',
    'L.Hip  ',
    'L.Knee ',
    'L.Ankle',
]

NUM_JOINTS = len(JOINT_NAMES)


# ── Online statistics (Welford's algorithm — no buffer needed) ────────────────

class OnlineStats:
    """
    Computes running mean and variance incrementally.
    Uses Welford's online algorithm — memory-efficient, numerically stable.
    This is important because the ML model will eventually use these
    statistics for input normalisation.
    """
    def __init__(self):
        self.n    = 0
        self.mean = 0.0
        self.M2   = 0.0   # sum of squared deviations

    def update(self, value: float):
        self.n += 1
        delta      = value - self.mean
        self.mean += delta / self.n
        delta2     = value - self.mean
        self.M2   += delta * delta2

    @property
    def variance(self) -> float:
        return self.M2 / self.n if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


# ── Error Monitor Node ────────────────────────────────────────────────────────

class ErrorMonitor(Node):
    def __init__(self):
        super().__init__('error_monitor')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('session_label',   'monitor')
        self.declare_parameter('log_dir',          os.path.expanduser('~/rehabrobo_logs'))
        self.declare_parameter('publish_rate_hz',  20.0)
        self.declare_parameter('verbose',          True)

        session_label   = self.get_parameter('session_label').get_parameter_value().string_value
        log_dir         = self.get_parameter('log_dir').get_parameter_value().string_value
        publish_rate_hz = self.get_parameter('publish_rate_hz').get_parameter_value().double_value

        # ── CSV setup ─────────────────────────────────────────────────────────
        os.makedirs(log_dir, exist_ok=True)
        ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'error_monitor_{session_label}_{ts}.csv'
        self.csv_path = os.path.join(log_dir, filename)

        self.csv_file   = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self._write_csv_header()

        # ── Internal state ────────────────────────────────────────────────────
        self.desired    = [0.0] * NUM_JOINTS
        self.actual     = {name: 0.0 for name in JOINT_NAMES}

        self.desired_received = False
        self.actual_received  = False

        self.start_time = None
        self.row_count  = 0

        # Online stats per joint (for running mean/std — used later by ML model)
        self.stats = [OnlineStats() for _ in range(NUM_JOINTS)]

        # For verbose print throttle
        self._last_verbose_print = -1.0

        # ── Publisher ─────────────────────────────────────────────────────────
        self.error_publisher = self.create_publisher(
            Float64MultiArray,
            '/joint_tracking_error',
            10
        )

        # ── Subscribers ───────────────────────────────────────────────────────
        # Subscribe to /nominal_commands (Step 4: walking_publisher publishes here)
        # Falls back gracefully if walking_publisher still uses the old topic
        self.sub_desired = self.create_subscription(
            Float64MultiArray,
            '/nominal_commands',
            self.desired_callback,
            10
        )

        self.sub_actual = self.create_subscription(
            JointState,
            '/joint_states',
            self.actual_callback,
            10
        )

        # ── Main timer ────────────────────────────────────────────────────────
        period = 1.0 / publish_rate_hz
        self.timer = self.create_timer(period, self.monitor_callback)

        self.get_logger().info('ErrorMonitor started.')
        self.get_logger().info(f'Logging to: {self.csv_path}')
        self.get_logger().info(
            'Waiting for /nominal_commands and /joint_states ...\n'
            'NOTE: Make sure walking_publisher publishes to /nominal_commands\n'
            '      (this will be set up in Step 4 via command_mixer).'
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Subscribers
    # ──────────────────────────────────────────────────────────────────────────

    def desired_callback(self, msg: Float64MultiArray):
        if len(msg.data) != NUM_JOINTS:
            self.get_logger().warn(
                f'Expected {NUM_JOINTS} values on /nominal_commands, '
                f'got {len(msg.data)}. Skipping.',
                throttle_duration_sec=5.0
            )
            return
        self.desired = list(msg.data)
        if not self.desired_received:
            self.desired_received = True
            self.get_logger().info('Receiving /nominal_commands ✓')

    def actual_callback(self, msg: JointState):
        for i, name in enumerate(msg.name):
            if name in self.actual:
                self.actual[name] = msg.position[i]
        if not self.actual_received:
            self.actual_received = True
            self.get_logger().info('Receiving /joint_states ✓')

    # ──────────────────────────────────────────────────────────────────────────
    # Main monitor callback
    # ──────────────────────────────────────────────────────────────────────────

    def monitor_callback(self):
        if not (self.desired_received and self.actual_received):
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if self.start_time is None:
            self.start_time = now
            self.get_logger().info('Error monitoring started.')

        t = now - self.start_time

        # Build ordered actual list
        actuals = [self.actual[name] for name in JOINT_NAMES]

        # Per-joint error
        errors     = [self.desired[i] - actuals[i] for i in range(NUM_JOINTS)]
        abs_errors = [abs(e) for e in errors]

        # RMSE across all joints
        rmse = math.sqrt(sum(e ** 2 for e in errors) / NUM_JOINTS)

        # Update online statistics
        for i in range(NUM_JOINTS):
            self.stats[i].update(errors[i])

        running_means = [self.stats[i].mean for i in range(NUM_JOINTS)]
        running_stds  = [self.stats[i].std  for i in range(NUM_JOINTS)]

        # ── Publish error to /joint_tracking_error ────────────────────────────
        msg = Float64MultiArray()
        msg.data = errors          # 6 floats, radians
        self.error_publisher.publish(msg)

        # ── Write CSV row ─────────────────────────────────────────────────────
        row = [f'{t:.4f}']
        row += [f'{v:.6f}' for v in self.desired]
        row += [f'{v:.6f}' for v in actuals]
        row += [f'{v:.6f}' for v in errors]
        row += [f'{v:.6f}' for v in abs_errors]
        row.append(f'{rmse:.6f}')
        row += [f'{v:.6f}' for v in running_means]
        row += [f'{v:.6f}' for v in running_stds]
        self.csv_writer.writerow(row)
        self.row_count += 1

        # ── Verbose console print every 5 seconds ─────────────────────────────
        verbose = self.get_parameter('verbose').get_parameter_value().bool_value
        if verbose and (t - self._last_verbose_print) >= 5.0:
            self._last_verbose_print = t
            self._print_status(t, errors, abs_errors, rmse, running_means, running_stds)

    # ──────────────────────────────────────────────────────────────────────────
    # Console status print
    # ──────────────────────────────────────────────────────────────────────────

    def _print_status(self, t, errors, abs_errors, rmse,
                      running_means, running_stds):
        self.get_logger().info(
            f'\n'
            f'  ── Error Monitor Status @ t={t:.1f}s '
            f'(rows={self.row_count}) ──\n'
            f'  {"Joint":<12} {"Error(rad)":>12} {"|Error|(rad)":>13} '
            f'{"RunMean":>10} {"RunSTD":>10}\n'
            f'  ' + '-' * 60 + '\n' +
            '\n'.join([
                f'  {JOINT_LABELS[i]:<12} '
                f'{errors[i]:>12.6f} '
                f'{abs_errors[i]:>13.6f} '
                f'{running_means[i]:>10.6f} '
                f'{running_stds[i]:>10.6f}'
                for i in range(NUM_JOINTS)
            ]) +
            f'\n  ' + '-' * 60 +
            f'\n  RMSE (all joints): {rmse:.6f} rad'
            f'  ({math.degrees(rmse):.4f} degrees)\n'
        )

    # ──────────────────────────────────────────────────────────────────────────
    # CSV header
    # ──────────────────────────────────────────────────────────────────────────

    def _write_csv_header(self):
        header = ['ros_time_sec']
        for n in JOINT_NAMES: header.append(f'desired_{n}')
        for n in JOINT_NAMES: header.append(f'actual_{n}')
        for n in JOINT_NAMES: header.append(f'error_{n}')
        for n in JOINT_NAMES: header.append(f'abs_error_{n}')
        header.append('rmse')
        for n in JOINT_NAMES: header.append(f'running_mean_{n}')
        for n in JOINT_NAMES: header.append(f'running_std_{n}')
        self.csv_writer.writerow(header)

    # ──────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────────────────

    def destroy_node(self):
        self.csv_file.flush()
        self.csv_file.close()
        self.get_logger().info(
            f'ErrorMonitor shut down. '
            f'Rows written: {self.row_count} | CSV: {self.csv_path}'
        )
        super().destroy_node()


# ── Entry point ───────────────────────────────────────────────────────────────

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
