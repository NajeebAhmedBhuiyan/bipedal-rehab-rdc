#!/home/nabq/miniconda3/envs/rdenv/bin/python3
"""
residual_dynamics_compensator.py — Phase 3 Fixed: With Scale + EMA Smoothing

Two key fixes over previous version:
  1. compensation_scale — multiplies raw LSTM output before publishing
                          prevents over-compensation (start at 0.3, tune up)
  2. ema_alpha          — exponential moving average smoothing on output
                          reduces high-frequency jumps between timesteps

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
  window_size        : int   — must match training     (default: 10)
  gait_frequency     : float — must match walking_pub  (default: 0.25)
  compensation_scale : float — output scale factor     (default: 0.3)
                               tune this up from 0.3 toward 1.0 gradually
  ema_alpha          : float — EMA smoothing factor    (default: 0.3)
                               0.0 = frozen, 1.0 = no smoothing

Runtime tuning:
  ros2 param set /residual_dynamics_compensator compensation_scale 0.3
  ros2 param set /residual_dynamics_compensator compensation_scale 0.5
  ros2 param set /residual_dynamics_compensator compensation_scale 0.7
  ros2 param set /residual_dynamics_compensator ema_alpha 0.3
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


# ── LSTM Model Definition (must match training exactly) ───────────────────────

class LSTMResidualCompensator(nn.Module):
    def __init__(self, input_dim=67, hidden_dim=128,
                 num_layers=2, output_dim=6, dropout=0.2):
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
        self.declare_parameter('window_size',         10)
        self.declare_parameter('gait_frequency',      0.25)
        self.declare_parameter('compensation_scale',  0.3)   # ← KEY: start conservative
        self.declare_parameter('ema_alpha',           0.3)   # ← KEY: smoothing

        model_path    = self.get_parameter('model_path').get_parameter_value().string_value
        scaler_x_path = self.get_parameter('scaler_x_path').get_parameter_value().string_value
        scaler_y_path = self.get_parameter('scaler_y_path').get_parameter_value().string_value
        self.window_size = self.get_parameter('window_size').get_parameter_value().integer_value
        self.gait_freq   = self.get_parameter('gait_frequency').get_parameter_value().double_value

        # ── Load model ────────────────────────────────────────────────────────
        self.device = torch.device('cpu')
        self.model  = LSTMResidualCompensator().to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()
        self.get_logger().info(f'LSTM model loaded: {model_path}')

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
        self.ema_output = [0.0] * NUM_JOINTS   # EMA state

        self.start_time  = None
        self.error_received   = False
        self.nominal_received = False
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
            'RDC Node ready.\n'
            '  compensation_scale = 0.3  (tune up gradually with ros2 param set)\n'
            '  ema_alpha          = 0.3  (smoothing factor)\n'
            '  Waiting for topics...'
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
            self.get_logger().info(
                f'RDC inference started ✓\n'
                f'  compensation_scale = {scale:.2f}\n'
                f'  ema_alpha          = {alpha:.2f}'
            )

        t = now - self.start_time

        if not enabled:
            msg = Float64MultiArray()
            msg.data = [0.0] * NUM_JOINTS
            self.pub_rdc.publish(msg)
            self.ema_output = [0.0] * NUM_JOINTS  # reset EMA when disabled
            return

        # ── Build feature vector (67 values) ──────────────────────────────────
        nominal_feat = self.nominal[:]
        window_flat  = []
        for past_errors in self.error_window:
            window_flat.extend(past_errors)
        gait_phase = (2 * math.pi * self.gait_freq * t) % (2 * math.pi)
        features   = nominal_feat + window_flat + [gait_phase]

        # ── Normalize ─────────────────────────────────────────────────────────
        features_np     = np.array(features, dtype=np.float32).reshape(1, -1)
        features_scaled = self.scaler_X.transform(features_np)

        # ── Inference ─────────────────────────────────────────────────────────
        with torch.no_grad():
            x_tensor    = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)
            pred_scaled = self.model(x_tensor).cpu().numpy()

        # ── Inverse transform ─────────────────────────────────────────────────
        raw_compensation = self.scaler_y.inverse_transform(pred_scaled)[0].tolist()

        # ── Scale down to prevent over-compensation ───────────────────────────
        scaled_compensation = [v * scale for v in raw_compensation]

        # ── EMA smoothing to reduce high-frequency jumps ──────────────────────
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
            scale_now = self.get_parameter('compensation_scale').get_parameter_value().double_value
            self.get_logger().info(
                f'\n  ── RDC Status @ t={t:.1f}s | '
                f'scale={scale_now:.2f} | '
                f'Inferences: {self._inference_count} ──\n' +
                '\n'.join([
                    f'  {JOINT_NAMES[i]:<22}: '
                    f'raw={raw_compensation[i]:>+.4f}  '
                    f'scaled={scaled_compensation[i]:>+.4f}  '
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