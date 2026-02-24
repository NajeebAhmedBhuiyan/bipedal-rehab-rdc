#!/home/nabq/miniconda3/envs/rdenv/bin/python3
"""
live_plotter.py — Phase 3 Fixed: Real-Time Joint Trajectory Plotter

Correct RMSE metrics:
  Dist RMSE  = RMSE(/disturbed_commands vs /nominal_commands)
               → pure disturbance effect, no RDC
  Comp RMSE  = RMSE(/forward_position_controller/commands vs /nominal_commands)
               → full signal with RDC applied

Both metrics are computed on the SAME time window so the comparison
is always apples-to-apples regardless of when you toggle rdc_enabled.

Signals plotted:
  GREEN  — /nominal_commands              (what we want)
  RED    — /disturbed_commands            (nominal + disturbance, no RDC)
  BLUE   — /forward_position_controller/commands  (actual robot command)
           When rdc=off: blue = red
           When rdc=on:  blue = red + rdc (should track green)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading
import math
import numpy as np


JOINT_LABELS = [
    'Right Hip', 'Right Knee', 'Right Ankle',
    'Left Hip',  'Left Knee',  'Left Ankle',
]
NUM_JOINTS  = 6
WINDOW_SECS = 10
RATE_HZ     = 20.0
BUFFER_SIZE = int(WINDOW_SECS * RATE_HZ)


class LivePlotterNode(Node):
    def __init__(self):
        super().__init__('live_plotter')

        self.times = deque([0.0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)

        self.nominal_buf   = [deque([0.0]*BUFFER_SIZE, maxlen=BUFFER_SIZE) for _ in range(NUM_JOINTS)]
        self.disturbed_buf = [deque([0.0]*BUFFER_SIZE, maxlen=BUFFER_SIZE) for _ in range(NUM_JOINTS)]
        self.actual_buf    = [deque([0.0]*BUFFER_SIZE, maxlen=BUFFER_SIZE) for _ in range(NUM_JOINTS)]

        self.nominal   = [0.0] * NUM_JOINTS
        self.disturbed = [0.0] * NUM_JOINTS
        self.actual    = [0.0] * NUM_JOINTS

        self.start_time = None
        self._lock      = threading.Lock()

        # Subscribe to all three signal topics
        self.create_subscription(
            Float64MultiArray, '/nominal_commands',
            self._nominal_cb, 10
        )
        self.create_subscription(
            Float64MultiArray, '/disturbed_commands',
            self._disturbed_cb, 10
        )
        self.create_subscription(
            Float64MultiArray, '/forward_position_controller/commands',
            self._actual_cb, 10
        )

        self.create_timer(1.0 / RATE_HZ, self._buffer_callback)
        self.get_logger().info('LivePlotter (Fixed) started.')

    def _nominal_cb(self, msg):
        if len(msg.data) == NUM_JOINTS:
            self.nominal = list(msg.data)

    def _disturbed_cb(self, msg):
        if len(msg.data) == NUM_JOINTS:
            self.disturbed = list(msg.data)

    def _actual_cb(self, msg):
        if len(msg.data) == NUM_JOINTS:
            self.actual = list(msg.data)

    def _buffer_callback(self):
        now = self.get_clock().now().nanoseconds / 1e9
        if self.start_time is None:
            self.start_time = now
        t = now - self.start_time

        with self._lock:
            self.times.append(t)
            for i in range(NUM_JOINTS):
                self.nominal_buf[i].append(math.degrees(self.nominal[i]))
                self.disturbed_buf[i].append(math.degrees(self.disturbed[i]))
                self.actual_buf[i].append(math.degrees(self.actual[i]))

    def get_data(self):
        with self._lock:
            return (
                list(self.times),
                [list(b) for b in self.nominal_buf],
                [list(b) for b in self.disturbed_buf],
                [list(b) for b in self.actual_buf],
            )


def run_plotter(ros_node):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle(
        'Live Joint Trajectories\n'
        '  ● Green = Nominal (desired)    '
        '● Red = Disturbed (no RDC)    '
        '● Blue = Actual (with RDC when enabled)',
        fontsize=11, color='white', fontweight='bold'
    )

    axes_flat    = axes.flatten()
    lines_nom    = []
    lines_dist   = []
    lines_actual = []
    status_texts = []

    for i, ax in enumerate(axes_flat):
        ax.set_facecolor('#0f0f23')
        ax.set_title(JOINT_LABELS[i], color='white', fontsize=10, fontweight='bold')
        ax.set_xlabel('Time (s)', color='#aaaaaa', fontsize=8)
        ax.set_ylabel('Angle (degrees)', color='#aaaaaa', fontsize=8)
        ax.tick_params(colors='#aaaaaa', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#444444')
        ax.grid(True, alpha=0.2, color='#444444')

        ln_nom,  = ax.plot([], [], color='#00e676', linewidth=1.8,
                           linestyle='-',  label='Nominal',   alpha=0.95)
        ln_dist, = ax.plot([], [], color='#ff5252', linewidth=1.2,
                           linestyle='-',  label='Disturbed', alpha=0.80)
        ln_act,  = ax.plot([], [], color='#40c4ff', linewidth=1.5,
                           linestyle='--', label='Actual',    alpha=0.90)

        lines_nom.append(ln_nom)
        lines_dist.append(ln_dist)
        lines_actual.append(ln_act)

        ax.legend(fontsize=7, loc='upper right',
                  facecolor='#1a1a2e', labelcolor='white', framealpha=0.7)

        # Two RMSE values — same time window, apples to apples
        txt = ax.text(0.02, 0.05, '', transform=ax.transAxes,
                      color='#ffcc00', fontsize=8, fontweight='bold',
                      verticalalignment='bottom')
        status_texts.append(txt)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    def animate(_frame):
        t, nom, dist, actual = ros_node.get_data()
        t_arr = np.array(t)

        for i in range(NUM_JOINTS):
            nom_arr  = np.array(nom[i])
            dist_arr = np.array(dist[i])
            act_arr  = np.array(actual[i])

            lines_nom[i].set_data(t_arr, nom_arr)
            lines_dist[i].set_data(t_arr, dist_arr)
            lines_actual[i].set_data(t_arr, act_arr)

            ax = axes_flat[i]
            if len(t_arr) > 1:
                ax.set_xlim(t_arr[-1] - WINDOW_SECS, t_arr[-1])

            all_vals = np.concatenate([nom_arr, dist_arr, act_arr])
            ymin, ymax = all_vals.min(), all_vals.max()
            margin = max(2.0, (ymax - ymin) * 0.15)
            ax.set_ylim(ymin - margin, ymax + margin)

            # ── Correct RMSE metrics ──────────────────────────────────────────
            # Both computed on the SAME window — truly apples to apples
            dist_rmse = np.sqrt(np.mean((dist_arr - nom_arr) ** 2))   # disturbance effect
            comp_rmse = np.sqrt(np.mean((act_arr  - nom_arr) ** 2))   # after RDC

            # Compute improvement percentage
            if dist_rmse > 0:
                improvement = (dist_rmse - comp_rmse) / dist_rmse * 100
                imp_str = f'{improvement:+.1f}%'
                imp_color = '#00e676' if improvement > 0 else '#ff5252'
            else:
                imp_str   = 'N/A'
                imp_color = '#ffcc00'

            status_texts[i].set_text(
                f'Dist RMSE : {dist_rmse:.2f}°\n'
                f'Comp RMSE : {comp_rmse:.2f}°\n'
                f'Improvement: {imp_str}'
            )
            status_texts[i].set_color(imp_color)

        return lines_nom + lines_dist + lines_actual + status_texts

    ani = animation.FuncAnimation(
        fig, animate,
        interval=100,
        blit=False,
        cache_frame_data=False
    )

    plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = LivePlotterNode()

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    try:
        run_plotter(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()