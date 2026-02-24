"""
rehabrobo_full.launch.py — Full Phase 3 RDC Pipeline Launch File v3

Launches everything in one command. Starts at Stage 1 (clean baseline).
Progress through stages manually from a separate terminal:

  STAGE 1 — Clean baseline (default on launch):
    ros2 param set /command_mixer disturbance_enabled false
    ros2 param set /command_mixer rdc_enabled false

  STAGE 2 — Disturbance on, RDC off:
    ros2 param set /command_mixer disturbance_enabled true
    ros2 param set /command_mixer rdc_enabled false

  STAGE 3 — RDC compensating:
    ros2 param set /command_mixer rdc_enabled true

  Optional RDC tuning:
    ros2 param set /residual_dynamics_compensator compensation_scale 1.0
    ros2 param set /residual_dynamics_compensator ema_alpha 0.5

  Launch arguments:
    disturbance_type  : spasticity | bias | fatigue  (default: spasticity)
    magnitude         : float                        (default: 0.15)
    burst_duration    : float                        (default: 1.5)
    burst_interval_min: float                        (default: 0.5)
    burst_interval_max: float                        (default: 1.5)
    session_label     : str                          (default: phase3_live)
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    TimerAction,
    LogInfo,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# ── Paths ─────────────────────────────────────────────────────────────────────

PKG_NAME   = 'rehabrobo_control'
MODELS_DIR = os.path.expanduser('~/rehabrobo_logs/MODELS-n-FIGS')


def generate_launch_description():

    pkg_share = get_package_share_directory(PKG_NAME)
    urdf_path = os.path.join(pkg_share, 'model', 'model.urdf')
    rviz_path = os.path.join(pkg_share, 'config', 'config.rviz')
    ctrl_path = os.path.join(pkg_share, 'config', 'robot_controller.yaml')

    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    # ── Launch Arguments ──────────────────────────────────────────────────────

    arg_disturbance_type = DeclareLaunchArgument(
        'disturbance_type',
        default_value='spasticity',
        description='Type of disturbance: spasticity | bias | fatigue'
    )
    arg_magnitude = DeclareLaunchArgument(
        'magnitude',
        default_value='0.15',
        description='Disturbance magnitude in radians'
    )
    arg_burst_duration = DeclareLaunchArgument(
        'burst_duration',
        default_value='1.5',
        description='Spasticity burst duration (seconds)'
    )
    arg_burst_min = DeclareLaunchArgument(
        'burst_interval_min',
        default_value='0.5',
        description='Minimum interval between spasticity bursts (seconds)'
    )
    arg_burst_max = DeclareLaunchArgument(
        'burst_interval_max',
        default_value='1.5',
        description='Maximum interval between spasticity bursts (seconds)'
    )
    arg_session_label = DeclareLaunchArgument(
        'session_label',
        default_value='phase3_live',
        description='Label for error monitor CSV log'
    )

    # ── Nodes ─────────────────────────────────────────────────────────────────

    # 1 — Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'robot_description': robot_description}],
        output='screen'
    )

    # 2 — RViz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_path],
        output='screen'
    )

    # 3 — ros2_control node
    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            {'robot_description': robot_description},
            ctrl_path
        ],
        output='screen'
    )

    # 4 — Joint State Broadcaster
    joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen'
    )

    # 5 — Forward Position Controller
    forward_position_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['forward_position_controller'],
        output='screen'
    )

    # 6 — Walking Publisher (t+3s)
    walking_publisher = TimerAction(
        period=3.0,
        actions=[
            Node(
                package=PKG_NAME,
                executable='walking_publisher.py',
                name='walking_publisher',
                output='screen'
            )
        ]
    )

    # 7 — Disturbance Injector (t+3s)
    disturbance_injector = TimerAction(
        period=3.0,
        actions=[
            Node(
                package=PKG_NAME,
                executable='disturbance_injector.py',
                name='disturbance_injector',
                parameters=[{
                    'disturbance_type':   LaunchConfiguration('disturbance_type'),
                    'magnitude':          LaunchConfiguration('magnitude'),
                    'burst_duration':     LaunchConfiguration('burst_duration'),
                    'burst_interval_min': LaunchConfiguration('burst_interval_min'),
                    'burst_interval_max': LaunchConfiguration('burst_interval_max'),
                }],
                output='screen'
            )
        ]
    )

    # 8 — Command Mixer (t+4s)
    # ── Starts at STAGE 1: disturbance OFF, rdc OFF ──
    # Progress manually:
    #   Stage 2: ros2 param set /command_mixer disturbance_enabled true
    #   Stage 3: ros2 param set /command_mixer rdc_enabled true
    command_mixer = TimerAction(
        period=4.0,
        actions=[
            Node(
                package=PKG_NAME,
                executable='command_mixer.py',
                name='command_mixer',
                parameters=[{
                    'disturbance_enabled': False,   # ← Stage 1: OFF
                    'rdc_enabled':         False,   # ← Stage 1: OFF
                    'publish_rate_hz':     20.0,
                    'verbose':             False,
                }],
                output='screen'
            )
        ]
    )

    # 9 — Error Monitor (t+5s)
    error_monitor = TimerAction(
        period=5.0,
        actions=[
            Node(
                package=PKG_NAME,
                executable='error_monitor.py',
                name='error_monitor',
                parameters=[{
                    'session_label':   LaunchConfiguration('session_label'),
                    'publish_rate_hz': 20.0,
                    'verbose':         False,
                }],
                output='screen'
            )
        ]
    )

    # 10 — RDC Node (t+6s)
    # window_size=30 matches the retrained v3 model (input_dim=187)
    rdc_node = TimerAction(
        period=6.0,
        actions=[
            Node(
                package=PKG_NAME,
                executable='residual_dynamics_compensator.py',
                name='residual_dynamics_compensator',
                parameters=[{
                    'model_path':         os.path.join(MODELS_DIR, 'rdc_lstm_best.pt'),
                    'scaler_x_path':      os.path.join(MODELS_DIR, 'scaler_X.pkl'),
                    'scaler_y_path':      os.path.join(MODELS_DIR, 'scaler_y.pkl'),
                    'enabled':            True,
                    'window_size':        30,       # ← v3: was 10
                    'gait_frequency':     0.25,
                    'compensation_scale': 1.0,
                    'ema_alpha':          0.5,
                }],
                output='screen'
            )
        ]
    )

    # 11 — Live Plotter (t+7s)
    live_plotter = TimerAction(
        period=7.0,
        actions=[
            Node(
                package=PKG_NAME,
                executable='live_plotter.py',
                name='live_plotter',
                output='screen'
            )
        ]
    )

    # ── Startup banner ────────────────────────────────────────────────────────

    startup_msg = LogInfo(msg=(
        '\n'
        '╔══════════════════════════════════════════════════════════╗\n'
        '║       Rehab Exo RDC v3 — Full Pipeline Launching         ║\n'
        '╠══════════════════════════════════════════════════════════╣\n'
        '║  Startup sequence:                                       ║\n'
        '║    t+0s : robot_state_publisher, rviz, controllers       ║\n'
        '║    t+3s : walking_publisher, disturbance_injector        ║\n'
        '║    t+4s : command_mixer        (Stage 1 — all OFF)       ║\n'
        '║    t+5s : error_monitor                                  ║\n'
        '║    t+6s : residual_dynamics_compensator (window=30)      ║\n'
        '║    t+7s : live_plotter                                   ║\n'
        '╠══════════════════════════════════════════════════════════╣\n'
        '║  3-Stage experiment (run from separate terminal):        ║\n'
        '║                                                          ║\n'
        '║  STAGE 1 — Clean baseline (default):                     ║\n'
        '║    ros2 param set /command_mixer disturbance_enabled false║\n'
        '║    ros2 param set /command_mixer rdc_enabled false       ║\n'
        '║                                                          ║\n'
        '║  STAGE 2 — Disturbance on, RDC off:                      ║\n'
        '║    ros2 param set /command_mixer disturbance_enabled true ║\n'
        '║    ros2 param set /command_mixer rdc_enabled false       ║\n'
        '║                                                          ║\n'
        '║  STAGE 3 — RDC compensating:                             ║\n'
        '║    ros2 param set /command_mixer rdc_enabled true        ║\n'
        '╚══════════════════════════════════════════════════════════╝'
    ))

    # ── Assemble ──────────────────────────────────────────────────────────────

    return LaunchDescription([
        arg_disturbance_type,
        arg_magnitude,
        arg_burst_duration,
        arg_burst_min,
        arg_burst_max,
        arg_session_label,
        startup_msg,
        robot_state_publisher,
        rviz,
        ros2_control_node,
        joint_state_broadcaster,
        forward_position_controller,
        walking_publisher,
        disturbance_injector,
        command_mixer,
        error_monitor,
        rdc_node,
        live_plotter,
    ])