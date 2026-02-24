#!/home/nabq/miniconda3/envs/rdenv/bin/python3
"""
residual_dynamics_compensator.py — Phase 3 v2: Window Size 30

KEY CHANGE: window_size increased from 10 → 30 steps (1.5 seconds of history)
            input_dim increased from 67 → 187 to match new training data

Feature vector (187 values):
  6   nominal commands
  180 error window (30 steps × 6 joints)
  1   gait phase
  ──────────────────
  187 total

Must be used with model trained on window_size=30 data (ml_data_collector_v2).

Subscribes to:
  /joint_tracking_error   — 6 joint errors (from error_monitor)
  /nominal_commands       — 6 desired positions (for gait phase)

Publishes:
  /rdc_commands           — 6 smoothed, scaled compensation values (rad)

ROS2 Parameters:
  model_path         : str   — path to rdc_lstm_best.pt
  scaler_x_path      : str   — path to scaler_X.pkl
  scaler_y_path      : str   — path to scaler_y.pkl
  enabled            : bool  — master on/off switch   (default: True)
  window_size        : int   — must match training     (default: 30)
  gait_frequency     : float — must match walking_pub  (default: 0.25)
  compensation_scale : float — output scale factor     (default: 1.0)
  ema_alpha          : float — EMA smoothing factor    (default: 0.5)
                               0.3 = smooth, 0.8 = responsive
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

import torch
import torch.nn as nn
import numpy as np
import pickle
import math
from collections import deque


# ── LSTM Model Definition — must match training exactly ───────────────────────

class LSTMResidualCompensator(nn.Module):
    def __init__(self,
                 input_dim=187,    # ← 6 + 180 + 1 (window_size=30)
                 hidden_dim=128,
                 num_layers=2,
                 output_dim=6,
                 dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.output_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.output_head(out)


NUM_JOINTS  = 6
JOINT_NAMES = [
    'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
    'left_hip_joint',  'left_knee_joint',  'left_ankle_joint',
]


class RDCNode(Node):
    def __init__(self):
        super().__init__('residual_dynamics_compensator')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('model_path',
            '/home/nabq/rehabrobo_logs/MODELS-n-FIGS/rdc_lstm_best.pt')
        self.declare_parameter('scaler_x_path',
            '/home/nabq/rehabrobo_logs/MODELS-n-FIGS/scaler_X.pkl')
        self.declare_parameter('scaler_y_path',
            '/home/nabq/rehabrobo_logs/MODELS-n-FIGS/scaler_y.pkl')
        self.declare_parameter('enabled',             True)
        self.declare_parameter('window_size',         30)    # ← changed from 10
        self.declare_parameter('gait_frequency',      0.25)
        self.declare_parameter('compensation_scale',  1.0)   # no scaling needed
        self.declare_parameter('ema_alpha',           0.5)   # balanced smoothing

        model_path    = self.get_parameter('model_path').get_parameter_value().string_value
        scaler_x_path = self.get_parameter('scaler_x_path').get_parameter_value().string_value
        scaler_y_path = self.get_parameter('scaler_y_path').get_parameter_value().string_value
        self.window_size = self.get_parameter('window_size').get_parameter_value().integer_value
        self.gait_freq   = self.get_parameter('gait_frequency').get_parameter_value().double_value

        # Feature dim: 6 nominal + (window_size × 6) + 1 phase
        self.feature_dim = 6 + (self.window_size * NUM_JOINTS) + 1

        # ── Load model ────────────────────────────────────────────────────────
        self.device = torch.device('cpu')
        self.model  = LSTMResidualCompensator(input_dim=self.feature_dim).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()
        self.get_logger().info(f'LSTM model loaded: {model_path}')
        self.get_logger().info(f'Input dim: {self.feature_dim} (window_size={self.window_size})')

        # ── Load scalers ──────────────────────────────────────────────────────
        with open(scaler_x_path, 'rb') as f:
            self.scaler_X = pickle.load(f)
        with open(scaler_y_path, 'rb') as f:
            self.scaler_y = pickle.load(f)
        self.get_logger().info('Scalers loaded.')

        # ── Internal state ────────────────────────────────────────────────────
        self.error_window = deque(
            [[0.0] * NUM_JOINTS for _ in range(self.window_size)],
            maxlen=self.window_size
        )
        self.nominal    = [0.0] * NUM_JOINTS
        self.ema_output = [0.0] * NUM_JOINTS

        self.start_time         = None
        self.error_received     = False
        self.nominal_received   = False
        self._last_status_print = -1.0
        self._inference_count   = 0

        # ── Subscribers ───────────────────────────────────────────────────────
        self.sub_error = self.create_subscription(
            Float64MultiArray, '/joint_tracking_error',
            self.error_callback, 10
        )
        self.sub_nominal = self.create_subscription(
            Float64MultiArray, '/nominal_commands',
            self.nominal_callback, 10
        )

        # ── Publisher ─────────────────────────────────────────────────────────
        self.pub_rdc = self.create_publisher(
            Float64MultiArray, '/rdc_commands', 10
        )

        self.timer = self.create_timer(0.05, self.infer_callback)

        self.get_logger().info(
            f'RDC Node v2 ready.\n'
            f'  window_size        = {self.window_size} steps '
            f'({self.window_size * 0.05:.1f}s history)\n'
            f'  feature_dim        = {self.feature_dim}\n'
            f'  compensation_scale = 1.0\n'
            f'  ema_alpha          = 0.5\n'
            f'  Waiting for topics...'
        )

    def error_callback(self, msg):
        if len(msg.data) != NUM_JOINTS:
            return
        self.error_window.append(list(msg.data))
        if not self.error_received:
            self.error_received = True
            self.get_logger().info('  /joint_tracking_error  ✓')

    def nominal_callback(self, msg):
        if len(msg.data) != NUM_JOINTS:
            return
        self.nominal = list(msg.data)
        if not self.nominal_received:
            self.nominal_received = True
            self.get_logger().info('  /nominal_commands      ✓')

    def infer_callback(self):
        if not (self.error_received and self.nominal_received):
            return

        enabled = self.get_parameter('enabled').get_parameter_value().bool_value
        scale   = self.get_parameter('compensation_scale').get_parameter_value().double_value
        alpha   = self.get_parameter('ema_alpha').get_parameter_value().double_value

        now = self.get_clock().now().nanoseconds / 1e9
        if self.start_time is None:
            self.start_time = now
            self.get_logger().info('RDC inference started ✓')

        t = now - self.start_time

        if not enabled:
            msg = Float64MultiArray()
            msg.data = [0.0] * NUM_JOINTS
            self.pub_rdc.publish(msg)
            self.ema_output = [0.0] * NUM_JOINTS
            return

        # ── Build feature vector (187 values) ─────────────────────────────────
        nominal_feat = self.nominal[:]

        window_flat = []
        for past_errors in self.error_window:
            window_flat.extend(past_errors)

        gait_phase = (2 * math.pi * self.gait_freq * t) % (2 * math.pi)
        features   = nominal_feat + window_flat + [gait_phase]  # 187 values

        # ── Normalize ─────────────────────────────────────────────────────────
        features_np     = np.array(features, dtype=np.float32).reshape(1, -1)
        features_scaled = self.scaler_X.transform(features_np)

        # ── Inference ─────────────────────────────────────────────────────────
        with torch.no_grad():
            x_tensor    = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)
            pred_scaled = self.model(x_tensor).cpu().numpy()

        # ── Inverse transform ─────────────────────────────────────────────────
        raw_compensation = self.scaler_y.inverse_transform(pred_scaled)[0].tolist()

        # ── Scale ─────────────────────────────────────────────────────────────
        scaled_compensation = [v * scale for v in raw_compensation]

        # ── EMA smoothing ─────────────────────────────────────────────────────
        smoothed = []
        for i in range(NUM_JOINTS):
            ema_val = alpha * scaled_compensation[i] + (1.0 - alpha) * self.ema_output[i]
            self.ema_output[i] = ema_val
            smoothed.append(ema_val)

        # ── Publish ───────────────────────────────────────────────────────────
        msg = Float64MultiArray()
        msg.data = smoothed
        self.pub_rdc.publish(msg)

        self._inference_count += 1

        # ── Status print every 10 seconds ─────────────────────────────────────
        if (t - self._last_status_print) >= 10.0:
            self._last_status_print = t
            self.get_logger().info(
                f'\n  ── RDC v2 @ t={t:.1f}s | '
                f'scale={scale:.2f} | alpha={alpha:.2f} | '
                f'Inferences: {self._inference_count} ──\n' +
                '\n'.join([
                    f'  {JOINT_NAMES[i]:<22}: '
                    f'raw={raw_compensation[i]:>+.4f}  '
                    f'smoothed={smoothed[i]:>+.4f}'
                    for i in range(NUM_JOINTS)
                ])
            )

    def destroy_node(self):
        self.get_logger().info(
            f'RDC Node shut down. Total inferences: {self._inference_count}'
        )
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RDCNode()
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