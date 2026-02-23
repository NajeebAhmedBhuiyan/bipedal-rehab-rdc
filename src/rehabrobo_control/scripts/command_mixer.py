#!/usr/bin/env python3
"""
command_mixer.py — Step 4: Command Mixer Node

The central hub of the pipeline. Combines:
  /nominal_commands   (from walking_publisher)   — clean sinusoidal trajectory
  /disturbance        (from disturbance_injector) — synthetic disturbance signal

And outputs:
  /forward_position_controller/commands          — final command sent to the robot

Formula:
  u_total[i] = clamp(u_nominal[i] + u_disturbance[i], joint_min[i], joint_max[i])

Joint limits (from model.urdf) are enforced so the mixer never commands
an unsafe position even with large disturbances.

Full pipeline after this node:

  walking_publisher  →  /nominal_commands  ──────────────────────────┐
                                                                      ↓
  disturbance_injector  →  /disturbance  ──────────→  command_mixer  →  /forward_position_controller/commands  →  RViz
                                                            ↑
                                              (also forwards /nominal_commands
                                               so error_monitor can compare)

ROS2 Parameters:
  disturbance_enabled : bool  — include disturbance in output  (default: True)
  publish_rate_hz     : float — output rate in Hz              (default: 20.0)
  verbose             : bool  — print mixed command every 5s   (default: False)

Usage:
  ros2 run rehabrobo_control command_mixer.py

  Disable disturbance at runtime (robot returns to clean gait):
  ros2 param set /command_mixer disturbance_enabled false
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

import math


# ── Joint limits from model.urdf ──────────────────────────────────────────────
# [right_hip, right_knee, right_ankle, left_hip, left_knee, left_ankle]
JOINT_MIN = [-1.57, -0.10, -0.50, -1.57, -0.10, -0.50]
JOINT_MAX = [ 1.57,  2.00,  0.50,  1.57,  2.00,  0.50]

JOINT_LABELS = [
    'R.Hip  ', 'R.Knee ', 'R.Ankle',
    'L.Hip  ', 'L.Knee ', 'L.Ankle',
]

NUM_JOINTS = 6


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ── Command Mixer Node ────────────────────────────────────────────────────────

class CommandMixer(Node):
    def __init__(self):
        super().__init__('command_mixer')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('disturbance_enabled', True)
        self.declare_parameter('publish_rate_hz',     20.0)
        self.declare_parameter('verbose',             False)

        publish_rate_hz = self.get_parameter('publish_rate_hz').get_parameter_value().double_value

        # ── Internal state ────────────────────────────────────────────────────
        self.nominal     = [0.0] * NUM_JOINTS
        self.disturbance = [0.0] * NUM_JOINTS

        self.nominal_received     = False
        self.disturbance_received = False

        self.start_time          = None
        self._last_verbose_print = -1.0

        # ── Subscribers ───────────────────────────────────────────────────────
        self.sub_nominal = self.create_subscription(
            Float64MultiArray,
            '/nominal_commands',
            self.nominal_callback,
            10
        )

        self.sub_disturbance = self.create_subscription(
            Float64MultiArray,
            '/disturbance',
            self.disturbance_callback,
            10
        )

        # ── Publishers ────────────────────────────────────────────────────────
        # Final command to the controller
        self.pub_commands = self.create_publisher(
            Float64MultiArray,
            '/forward_position_controller/commands',
            10
        )

        # ── Timer ─────────────────────────────────────────────────────────────
        period = 1.0 / publish_rate_hz
        self.timer = self.create_timer(period, self.mix_callback)

        self.get_logger().info('CommandMixer started.')
        self.get_logger().info(
            'Waiting for /nominal_commands and /disturbance ...'
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Subscribers
    # ──────────────────────────────────────────────────────────────────────────

    def nominal_callback(self, msg: Float64MultiArray):
        if len(msg.data) != NUM_JOINTS:
            self.get_logger().warn(
                f'Expected {NUM_JOINTS} values on /nominal_commands, '
                f'got {len(msg.data)}. Skipping.',
                throttle_duration_sec=5.0
            )
            return
        self.nominal = list(msg.data)
        if not self.nominal_received:
            self.nominal_received = True
            self.get_logger().info('Receiving /nominal_commands ✓')

    def disturbance_callback(self, msg: Float64MultiArray):
        if len(msg.data) != NUM_JOINTS:
            self.get_logger().warn(
                f'Expected {NUM_JOINTS} values on /disturbance, '
                f'got {len(msg.data)}. Skipping.',
                throttle_duration_sec=5.0
            )
            return
        self.disturbance = list(msg.data)
        if not self.disturbance_received:
            self.disturbance_received = True
            self.get_logger().info('Receiving /disturbance ✓')

    # ──────────────────────────────────────────────────────────────────────────
    # Main mix callback
    # ──────────────────────────────────────────────────────────────────────────

    def mix_callback(self):
        # Wait until at least nominal commands are coming in
        # Disturbance is optional — if not yet received, treat as zeros
        if not self.nominal_received:
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if self.start_time is None:
            self.start_time = now
            self.get_logger().info(
                'CommandMixer publishing to '
                '/forward_position_controller/commands ✓'
            )

        t = now - self.start_time

        disturbance_enabled = self.get_parameter(
            'disturbance_enabled'
        ).get_parameter_value().bool_value

        # ── Mix ───────────────────────────────────────────────────────────────
        mixed = []
        for i in range(NUM_JOINTS):
            d = self.disturbance[i] if disturbance_enabled else 0.0
            raw = self.nominal[i] + d
            mixed.append(clamp(raw, JOINT_MIN[i], JOINT_MAX[i]))

        # ── Publish to controller ─────────────────────────────────────────────
        msg = Float64MultiArray()
        msg.data = mixed
        self.pub_commands.publish(msg)

        # ── Verbose print every 5 seconds ─────────────────────────────────────
        verbose = self.get_parameter('verbose').get_parameter_value().bool_value
        if verbose and (t - self._last_verbose_print) >= 5.0:
            self._last_verbose_print = t
            d_enabled = disturbance_enabled
            self.get_logger().info(
                f'\n  ── CommandMixer @ t={t:.1f}s '
                f'[disturbance: {"ON" if d_enabled else "OFF"}] ──\n'
                f'  {"Joint":<12} {"Nominal":>10} {"Disturbance":>13} {"Mixed":>10}\n'
                f'  ' + '-' * 48 + '\n' +
                '\n'.join([
                    f'  {JOINT_LABELS[i]:<12} '
                    f'{self.nominal[i]:>10.4f} '
                    f'{(self.disturbance[i] if d_enabled else 0.0):>13.4f} '
                    f'{mixed[i]:>10.4f}'
                    for i in range(NUM_JOINTS)
                ]) + '\n'
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────────────────

    def destroy_node(self):
        self.get_logger().info('CommandMixer shut down.')
        super().destroy_node()


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = CommandMixer()
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
