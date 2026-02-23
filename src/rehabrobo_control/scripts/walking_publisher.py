#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import math
import time

class WalkingPublisher(Node):
    def __init__(self):
        super().__init__('walking_publisher')

        self.publisher_ = self.create_publisher(
            Float64MultiArray,
            '/nominal_commands',
            10
        )

        timer_period = 0.05  # 20 Hz (slower)
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.start_time = self.get_clock().now().nanoseconds / 1e9

        # Gait parameters (adjustable)
        self.hip_amplitude = math.radians(30)  # ~0.52 rad
        self.knee_amplitude = math.radians(25)  # ~0.44 rad
        self.ankle_amplitude = math.radians(10)  # ~0.17 rad

        self.frequency = 0.25  # Hz (slower walking cycle)
        self.phase_offset = math.pi  # Left/right leg phase diff (180 degrees)

    def timer_callback(self):
        now = self.get_clock().now().nanoseconds / 1e9
        t = now - self.start_time
        omega = 2 * math.pi * self.frequency

        # Right leg
        right_hip = self.hip_amplitude * math.sin(omega * t)
        right_knee = self.knee_amplitude * (math.sin(omega * t) > 0) * math.sin(omega * t)**2
        right_ankle = -self.ankle_amplitude * math.sin(omega * t)

        # Left leg (180 deg phase shifted)
        left_hip = self.hip_amplitude * math.sin(omega * t + self.phase_offset)
        left_knee = self.knee_amplitude * (math.sin(omega * t + self.phase_offset) > 0) * math.sin(omega * t + self.phase_offset)**2
        left_ankle = -self.ankle_amplitude * math.sin(omega * t + self.phase_offset)

        msg = Float64MultiArray()
        msg.data = [
            right_hip, right_knee, right_ankle,
            left_hip, left_knee, left_ankle
        ]

        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = WalkingPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()