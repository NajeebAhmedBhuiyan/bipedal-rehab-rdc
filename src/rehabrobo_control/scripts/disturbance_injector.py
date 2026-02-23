#!/usr/bin/env python3
"""
disturbance_injector.py — Step 2: Disturbance Injector Node

Publishes synthetic disturbances to /disturbance (Float64MultiArray).
The disturbance values are additive joint-position offsets in radians,
matching the same 6-joint order as the walking_publisher and controller.

Joint order (must match robot_controller.yaml):
  [right_hip, right_knee, right_ankle, left_hip, left_knee, left_ankle]

Three disturbance types (switchable via ROS2 parameter at runtime):
  1. spasticity  — random short bursts of sinusoidal noise on affected joints
  2. bias        — constant offset simulating limb weight or muscle stiffness
  3. fatigue     — slowly growing sinusoidal drift that worsens over session time

ROS2 Parameters (all changeable at runtime via `ros2 param set`):
  disturbance_type      : str   — 'spasticity' | 'bias' | 'fatigue'  (default: 'spasticity')
  magnitude             : float — base disturbance amplitude in radians (default: 0.15)
  affected_side         : str   — 'right' | 'left' | 'both'           (default: 'both')
  affected_joints       : str   — 'all' | 'hip' | 'knee' | 'ankle'   (default: 'all')
  enabled               : bool  — master on/off switch                 (default: True)

  # Spasticity-specific
  burst_duration        : float — seconds each burst lasts             (default: 0.4)
  burst_interval_min    : float — min seconds between bursts           (default: 1.5)
  burst_interval_max    : float — max seconds between bursts           (default: 4.0)
  burst_frequency       : float — Hz of sinusoid within the burst      (default: 6.0)

  # Fatigue-specific
  fatigue_time_constant : float — seconds to reach full magnitude      (default: 60.0)

Usage:
  ros2 run rehabrobo_control disturbance_injector.py

  Switch type at runtime (in a separate terminal):
  ros2 param set /disturbance_injector disturbance_type bias
  ros2 param set /disturbance_injector magnitude 0.10
  ros2 param set /disturbance_injector enabled false
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

import math
import random


# ── Joint index map ───────────────────────────────────────────────────────────
# Matches the order in robot_controller.yaml / walking_publisher.py
JOINT_INDEX = {
    'right_hip':    0,
    'right_knee':   1,
    'right_ankle':  2,
    'left_hip':     3,
    'left_knee':    4,
    'left_ankle':   5,
}

NUM_JOINTS = 6


def build_joint_mask(affected_side: str, affected_joints: str) -> list:
    """
    Returns a list of 6 booleans indicating which joints receive disturbance.
    """
    sides  = []
    joints = []

    if affected_side == 'right':
        sides = ['right']
    elif affected_side == 'left':
        sides = ['left']
    else:  # 'both'
        sides = ['right', 'left']

    if affected_joints == 'hip':
        joints = ['hip']
    elif affected_joints == 'knee':
        joints = ['knee']
    elif affected_joints == 'ankle':
        joints = ['ankle']
    else:  # 'all'
        joints = ['hip', 'knee', 'ankle']

    mask = [False] * NUM_JOINTS
    for side in sides:
        for joint in joints:
            key = f'{side}_{joint}'
            if key in JOINT_INDEX:
                mask[JOINT_INDEX[key]] = True
    return mask


# ── Disturbance Injector Node ─────────────────────────────────────────────────

class DisturbanceInjector(Node):
    def __init__(self):
        super().__init__('disturbance_injector')

        # ── Declare parameters ────────────────────────────────────────────────
        self.declare_parameter('disturbance_type',      'spasticity')
        self.declare_parameter('magnitude',              0.15)
        self.declare_parameter('affected_side',         'both')
        self.declare_parameter('affected_joints',       'all')
        self.declare_parameter('enabled',                True)

        # Spasticity params
        self.declare_parameter('burst_duration',         0.4)
        self.declare_parameter('burst_interval_min',     1.5)
        self.declare_parameter('burst_interval_max',     4.0)
        self.declare_parameter('burst_frequency',        6.0)

        # Fatigue params
        self.declare_parameter('fatigue_time_constant',  60.0)

        # ── Publisher ─────────────────────────────────────────────────────────
        self.publisher_ = self.create_publisher(
            Float64MultiArray,
            '/disturbance',
            10
        )

        # ── Timer: publish at 20 Hz to match walking_publisher ───────────────
        self.timer = self.create_timer(0.05, self.timer_callback)

        # ── Internal state ────────────────────────────────────────────────────
        self.start_time = self.get_clock().now().nanoseconds / 1e9

        # Spasticity state
        self.in_burst           = False
        self.burst_start_time   = 0.0
        self.next_burst_time    = self._random_interval()  # first burst delay

        self.get_logger().info('DisturbanceInjector started.')
        self._log_current_params()

    # ──────────────────────────────────────────────────────────────────────────
    # Timer callback
    # ──────────────────────────────────────────────────────────────────────────

    def timer_callback(self):
        # Read current params (supports runtime changes via ros2 param set)
        enabled          = self.get_parameter('enabled').get_parameter_value().bool_value
        disturbance_type = self.get_parameter('disturbance_type').get_parameter_value().string_value
        magnitude        = self.get_parameter('magnitude').get_parameter_value().double_value
        affected_side    = self.get_parameter('affected_side').get_parameter_value().string_value
        affected_joints  = self.get_parameter('affected_joints').get_parameter_value().string_value

        now = self.get_clock().now().nanoseconds / 1e9
        t   = now - self.start_time

        disturbance = [0.0] * NUM_JOINTS

        if enabled:
            mask = build_joint_mask(affected_side, affected_joints)

            if disturbance_type == 'spasticity':
                disturbance = self._spasticity(t, magnitude, mask)

            elif disturbance_type == 'bias':
                disturbance = self._bias(magnitude, mask)

            elif disturbance_type == 'fatigue':
                fatigue_tc = self.get_parameter('fatigue_time_constant').get_parameter_value().double_value
                disturbance = self._fatigue(t, magnitude, fatigue_tc, mask)

            else:
                self.get_logger().warn(
                    f"Unknown disturbance_type '{disturbance_type}'. "
                    "Use: spasticity | bias | fatigue",
                    throttle_duration_sec=5.0
                )

        msg = Float64MultiArray()
        msg.data = disturbance
        self.publisher_.publish(msg)

    # ──────────────────────────────────────────────────────────────────────────
    # Disturbance type implementations
    # ──────────────────────────────────────────────────────────────────────────

    def _spasticity(self, t: float, magnitude: float, mask: list) -> list:
        """
        Random burst pattern:
        - Stays at 0 between bursts
        - During a burst: sinusoidal noise at burst_frequency Hz
        - Burst duration and inter-burst interval are randomised
        """
        burst_duration   = self.get_parameter('burst_duration').get_parameter_value().double_value
        burst_freq       = self.get_parameter('burst_frequency').get_parameter_value().double_value

        disturbance = [0.0] * NUM_JOINTS

        if self.in_burst:
            elapsed = t - self.burst_start_time
            if elapsed < burst_duration:
                # Sinusoidal burst with a smooth onset/offset envelope
                phase     = 2 * math.pi * burst_freq * elapsed
                envelope  = math.sin(math.pi * elapsed / burst_duration)  # 0→1→0
                amplitude = magnitude * envelope * math.sin(phase)
                for j in range(NUM_JOINTS):
                    if mask[j]:
                        # Slight per-joint phase variation for realism
                        phase_j = phase + j * 0.3
                        disturbance[j] = magnitude * envelope * math.sin(phase_j)
            else:
                # Burst ended
                self.in_burst        = False
                self.next_burst_time = t + self._random_interval()
                self.get_logger().info(
                    f'[Spasticity] Burst ended at t={t:.2f}s. '
                    f'Next burst at t={self.next_burst_time:.2f}s'
                )
        else:
            if t >= self.next_burst_time:
                # Start a new burst
                self.in_burst         = True
                self.burst_start_time = t
                self.get_logger().info(
                    f'[Spasticity] Burst started at t={t:.2f}s '
                    f'(duration={burst_duration:.2f}s, freq={burst_freq:.1f}Hz)'
                )

        return disturbance

    def _bias(self, magnitude: float, mask: list) -> list:
        """
        Constant offset on affected joints.
        Simulates: limb weight asymmetry, persistent muscle stiffness.
        Uses a small sinusoidal wobble (5% of magnitude) for slight realism.
        """
        t = self.get_clock().now().nanoseconds / 1e9 - self.start_time
        wobble = 0.05 * magnitude * math.sin(2 * math.pi * 0.1 * t)

        disturbance = [0.0] * NUM_JOINTS
        for j in range(NUM_JOINTS):
            if mask[j]:
                disturbance[j] = magnitude + wobble
        return disturbance

    def _fatigue(self, t: float, magnitude: float,
                 time_constant: float, mask: list) -> list:
        """
        Slowly growing drift that increases over session time.
        Models patient fatigue: the robot needs progressively more force
        to move the limb as the session goes on.

        Growth follows: magnitude * (1 - exp(-t / time_constant))
        At t = time_constant  → ~63% of full magnitude
        At t = 3*time_constant → ~95% of full magnitude

        A slow sinusoidal modulation (0.05 Hz) adds realistic variability.
        """
        growth      = magnitude * (1.0 - math.exp(-t / max(time_constant, 0.1)))
        modulation  = 1.0 + 0.2 * math.sin(2 * math.pi * 0.05 * t)
        amplitude   = growth * modulation

        disturbance = [0.0] * NUM_JOINTS
        for j in range(NUM_JOINTS):
            if mask[j]:
                # Slow drift oscillation — different frequency per joint
                freq = 0.08 + j * 0.01
                disturbance[j] = amplitude * math.sin(2 * math.pi * freq * t)

        # Log fatigue level every 10 seconds
        if int(t) % 10 == 0 and int(t) != getattr(self, '_last_fatigue_log', -1):
            self._last_fatigue_log = int(t)
            self.get_logger().info(
                f'[Fatigue] t={t:.1f}s | growth={growth:.4f} rad '
                f'({100*growth/magnitude:.1f}% of max magnitude)'
            )

        return disturbance

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _random_interval(self) -> float:
        """Random wait time between spasticity bursts."""
        lo = self.get_parameter('burst_interval_min').get_parameter_value().double_value
        hi = self.get_parameter('burst_interval_max').get_parameter_value().double_value
        return random.uniform(lo, hi)

    def _log_current_params(self):
        dtype  = self.get_parameter('disturbance_type').get_parameter_value().string_value
        mag    = self.get_parameter('magnitude').get_parameter_value().double_value
        side   = self.get_parameter('affected_side').get_parameter_value().string_value
        joints = self.get_parameter('affected_joints').get_parameter_value().string_value
        self.get_logger().info(
            f'Config → type={dtype} | magnitude={mag:.3f} rad | '
            f'side={side} | joints={joints}'
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = DisturbanceInjector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
