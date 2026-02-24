#!/usr/bin/env python3
"""
command_mixer.py — Phase 3 Fixed: Command Mixer with Clean Error Signal

KEY FIX: Publishes TWO output topics:
  /disturbed_commands  — nominal + disturbance ONLY (no RDC)
                         → error_monitor uses this to compute clean error
  /forward_position_controller/commands — nominal + disturbance + rdc
                         → actual robot command

This separates the error signal from the RDC output, preventing the
feedback loop contamination where RDC was compensating itself.

Pipeline:
  walking_publisher  →  /nominal_commands  ──────────────────────────────┐
                                                                          ↓
  disturbance_injector → /disturbance ──→  command_mixer ──→ /disturbed_commands
                                                │                  ↓
                                                │            error_monitor
                                                │                  ↓
  rdc_node → /rdc_commands ───────────────────→ │         /joint_tracking_error
                                                │                  ↓
                                                └──→ /forward_position_controller/commands → RViz

ROS2 Parameters:
  disturbance_enabled : bool  — include disturbance  (default: True)
  rdc_enabled         : bool  — include RDC          (default: True)
  publish_rate_hz     : float — output rate in Hz    (default: 20.0)
  verbose             : bool  — print table every 5s (default: False)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


JOINT_MIN = [-1.57, -0.10, -0.50, -1.57, -0.10, -0.50]
JOINT_MAX = [ 1.57,  2.00,  0.50,  1.57,  2.00,  0.50]

JOINT_LABELS = [
    'R.Hip  ', 'R.Knee ', 'R.Ankle',
    'L.Hip  ', 'L.Knee ', 'L.Ankle',
]

NUM_JOINTS = 6


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


class CommandMixer(Node):
    def __init__(self):
        super().__init__('command_mixer')

        self.declare_parameter('disturbance_enabled', True)
        self.declare_parameter('rdc_enabled',         True)
        self.declare_parameter('publish_rate_hz',     20.0)
        self.declare_parameter('verbose',             False)

        publish_rate_hz = self.get_parameter('publish_rate_hz').get_parameter_value().double_value

        self.nominal     = [0.0] * NUM_JOINTS
        self.disturbance = [0.0] * NUM_JOINTS
        self.rdc         = [0.0] * NUM_JOINTS

        self.nominal_received     = False
        self.disturbance_received = False
        self.rdc_received         = False

        self.start_time          = None
        self._last_verbose_print = -1.0

        # ── Subscribers ───────────────────────────────────────────────────────
        self.sub_nominal = self.create_subscription(
            Float64MultiArray, '/nominal_commands',
            self.nominal_callback, 10
        )
        self.sub_disturbance = self.create_subscription(
            Float64MultiArray, '/disturbance',
            self.disturbance_callback, 10
        )
        self.sub_rdc = self.create_subscription(
            Float64MultiArray, '/rdc_commands',
            self.rdc_callback, 10
        )

        # ── Publishers ────────────────────────────────────────────────────────
        # Topic 1: nominal + disturbance ONLY — for error_monitor (clean signal)
        self.pub_disturbed = self.create_publisher(
            Float64MultiArray,
            '/disturbed_commands',
            10
        )
        # Topic 2: nominal + disturbance + rdc — for the actual robot
        self.pub_commands = self.create_publisher(
            Float64MultiArray,
            '/forward_position_controller/commands',
            10
        )

        self.timer = self.create_timer(1.0 / publish_rate_hz, self.mix_callback)

        self.get_logger().info(
            'CommandMixer (Phase 3 Fixed) started.\n'
            '  Publishes /disturbed_commands  → clean error signal for RDC\n'
            '  Publishes /forward_position_controller/commands → robot\n'
            '  Waiting for topics...'
        )

    def nominal_callback(self, msg):
        if len(msg.data) != NUM_JOINTS:
            return
        self.nominal = list(msg.data)
        if not self.nominal_received:
            self.nominal_received = True
            self.get_logger().info('  /nominal_commands   ✓')

    def disturbance_callback(self, msg):
        if len(msg.data) != NUM_JOINTS:
            return
        self.disturbance = list(msg.data)
        if not self.disturbance_received:
            self.disturbance_received = True
            self.get_logger().info('  /disturbance        ✓')

    def rdc_callback(self, msg):
        if len(msg.data) != NUM_JOINTS:
            return
        self.rdc = list(msg.data)
        if not self.rdc_received:
            self.rdc_received = True
            self.get_logger().info('  /rdc_commands       ✓')

    def mix_callback(self):
        if not self.nominal_received:
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if self.start_time is None:
            self.start_time = now
            self.get_logger().info('CommandMixer publishing ✓')

        t = now - self.start_time

        dist_enabled = self.get_parameter('disturbance_enabled').get_parameter_value().bool_value
        rdc_enabled  = self.get_parameter('rdc_enabled').get_parameter_value().bool_value

        # ── Signal 1: nominal + disturbance (no RDC) ─────────────────────────
        # This is the CLEAN error signal — what the RDC was trained on
        disturbed = []
        for i in range(NUM_JOINTS):
            d = self.disturbance[i] if dist_enabled else 0.0
            disturbed.append(clamp(self.nominal[i] + d, JOINT_MIN[i], JOINT_MAX[i]))

        msg_disturbed = Float64MultiArray()
        msg_disturbed.data = disturbed
        self.pub_disturbed.publish(msg_disturbed)

        # ── Signal 2: nominal + disturbance + rdc (full command) ──────────────
        total = []
        for i in range(NUM_JOINTS):
            d   = self.disturbance[i] if dist_enabled else 0.0
            rdc = self.rdc[i]         if rdc_enabled  else 0.0
            total.append(clamp(self.nominal[i] + d + rdc, JOINT_MIN[i], JOINT_MAX[i]))

        msg_total = Float64MultiArray()
        msg_total.data = total
        self.pub_commands.publish(msg_total)

        # ── Verbose print ─────────────────────────────────────────────────────
        verbose = self.get_parameter('verbose').get_parameter_value().bool_value
        if verbose and (t - self._last_verbose_print) >= 5.0:
            self._last_verbose_print = t
            mode = (
                'BASELINE'    if not dist_enabled and not rdc_enabled else
                'DISTURBED'   if dist_enabled and not rdc_enabled     else
                'COMPENSATED' if dist_enabled and rdc_enabled         else
                'RDC ONLY'
            )
            self.get_logger().info(
                f'\n  ── CommandMixer @ t={t:.1f}s  [{mode}] ──\n'
                f'  {"Joint":<12} {"Nominal":>10} {"Disturb":>10} '
                f'{"RDC":>10} {"Total":>10}\n'
                f'  ' + '-' * 46 + '\n' +
                '\n'.join([
                    f'  {JOINT_LABELS[i]:<12} '
                    f'{self.nominal[i]:>10.4f} '
                    f'{(self.disturbance[i] if dist_enabled else 0.0):>10.4f} '
                    f'{(self.rdc[i] if rdc_enabled else 0.0):>10.4f} '
                    f'{total[i]:>10.4f}'
                    for i in range(NUM_JOINTS)
                ]) + '\n'
            )

    def destroy_node(self):
        self.get_logger().info('CommandMixer shut down.')
        super().destroy_node()


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
