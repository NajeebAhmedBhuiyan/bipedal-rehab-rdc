"""
rehabrobo_full.launch.py — Full Phase 3 RDC Pipeline Launch File

Launches everything in one command:
  - robot_state_publisher  (URDF)
  - rviz2                  (visualization)
  - ros2_control_node      (hardware interface)
  - joint_state_broadcaster
  - forward_position_controller
  - walking_publisher      (sinusoidal gait → /nominal_commands)
  - disturbance_injector   (synthetic disturbance → /disturbance)
  - command_mixer          (mixes all signals → controller)
  - error_monitor          (computes error → /joint_tracking_error)
  - residual_dynamics_compensator  (LSTM inference → /rdc_commands)
  - live_plotter           (real-time visualization)

Usage:
  ros2 launch rehabrobo_control rehabrobo_full.launch.py

  With custom disturbance type and magnitude:
  ros2 launch rehabrobo_control rehabrobo_full.launch.py \
    disturbance_type:=bias magnitude:=0.10

  Available launch arguments:
    disturbance_type  : spasticity | bias | fatigue  (default: spasticity)
    magnitude         : float                        (default: 0.15)
    burst_duration    : float                        (default: 1.5)
    burst_interval_min: float                        (default: 0.5)
    burst_interval_max: float                        (default: 1.5)
    session_label     : str                          (default: phase3_live)

  Runtime control (after launch):
    ros2 param set /command_mixer disturbance_enabled false  ← clean baseline
    ros2 param set /command_mixer disturbance_enabled true   ← disturbance on
    ros2 param set /command_mixer rdc_enabled false          ← RDC off
    ros2 param set /command_mixer rdc_enabled true           ← RDC on
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    TimerAction,
    LogInfo,
)
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    Command,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


# ── Paths ─────────────────────────────────────────────────────────────────────

PKG_NAME    = 'rehabrobo_control'
MODELS_DIR  = os.path.expanduser('~/rehabrobo_logs/MODELS-n-FIGS')


def generate_launch_description():

    pkg_share = get_package_share_directory(PKG_NAME)
    urdf_path  = os.path.join(pkg_share, 'model', 'model.urdf')
    rviz_path  = os.path.join(pkg_share, 'config', 'config.rviz')
    ctrl_path  = os.path.join(pkg_share, 'config', 'robot_controller.yaml')

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

    # ── Node Definitions ──────────────────────────────────────────────────────

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

    # 4 — Joint State Broadcaster spawner
    joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen'
    )

    # 5 — Forward Position Controller spawner
    forward_position_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['forward_position_controller'],
        output='screen'
    )

    # 6 — Walking Publisher (delayed 3s — wait for controller to be ready)
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

    # 7 — Disturbance Injector (delayed 3s)
    disturbance_injector = TimerAction(
        period=3.0,
        actions=[
            Node(
                package=PKG_NAME,
                executable='disturbance_injector.py',
                name='disturbance_injector',
                parameters=[{
                    'disturbance_type':    LaunchConfiguration('disturbance_type'),
                    'magnitude':           LaunchConfiguration('magnitude'),
                    'burst_duration':      LaunchConfiguration('burst_duration'),
                    'burst_interval_min':  LaunchConfiguration('burst_interval_min'),
                    'burst_interval_max':  LaunchConfiguration('burst_interval_max'),
                }],
                output='screen'
            )
        ]
    )

    # 8 — Command Mixer (delayed 4s — after walking_publisher is up)
    # Starts with disturbance ON, RDC ON — control via ros2 param set
    command_mixer = TimerAction(
        period=4.0,
        actions=[
            Node(
                package=PKG_NAME,
                executable='command_mixer.py',
                name='command_mixer',
                parameters=[{
                    'disturbance_enabled': True,
                    'rdc_enabled':         True,
                    'publish_rate_hz':     20.0,
                    'verbose':             False,
                }],
                output='screen'
            )
        ]
    )

    # 9 — Error Monitor (delayed 5s — after command_mixer is publishing)
    error_monitor = TimerAction(
        period=5.0,
        actions=[
            Node(
                package=PKG_NAME,
                executable='error_monitor.py',
                name='error_monitor',
                parameters=[{
                    'session_label': LaunchConfiguration('session_label'),
                    'publish_rate_hz': 20.0,
                    'verbose': False,
                }],
                output='screen'
            )
        ]
    )

    # 10 — RDC Node (delayed 6s — after error_monitor is publishing)
    rdc_node = TimerAction(
        period=6.0,
        actions=[
            Node(
                package=PKG_NAME,
                executable='residual_dynamics_compensator.py',
                name='residual_dynamics_compensator',
                parameters=[{
                    'model_path':    os.path.join(MODELS_DIR, 'rdc_lstm_best.pt'),
                    'scaler_x_path': os.path.join(MODELS_DIR, 'scaler_X.pkl'),
                    'scaler_y_path': os.path.join(MODELS_DIR, 'scaler_y.pkl'),
                    'enabled':       True,
                    'window_size':   10,
                    'gait_frequency': 0.25,
                }],
                output='screen'
            )
        ]
    )

    # 11 — Live Plotter (delayed 7s — after everything is publishing)
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

    # ── Status messages ───────────────────────────────────────────────────────

    startup_msg = LogInfo(msg=(
        '\n'
        '╔══════════════════════════════════════════════════════╗\n'
        '║     Rehab Exo RDC — Full Pipeline Launching          ║\n'
        '╠══════════════════════════════════════════════════════╣\n'
        '║  Nodes start with staggered delays:                  ║\n'
        '║    t+0s  : robot_state_publisher, rviz, controllers  ║\n'
        '║    t+3s  : walking_publisher, disturbance_injector   ║\n'
        '║    t+4s  : command_mixer                             ║\n'
        '║    t+5s  : error_monitor                             ║\n'
        '║    t+6s  : residual_dynamics_compensator (LSTM)      ║\n'
        '║    t+7s  : live_plotter                              ║\n'
        '╠══════════════════════════════════════════════════════╣\n'
        '║  Runtime control:                                    ║\n'
        '║    # Clean baseline:                                 ║\n'
        '║    ros2 param set /command_mixer                     ║\n'
        '║                   disturbance_enabled false          ║\n'
        '║                                                      ║\n'
        '║    # Disturbed only:                                 ║\n'
        '║    ros2 param set /command_mixer                     ║\n'
        '║                   disturbance_enabled true           ║\n'
        '║    ros2 param set /command_mixer rdc_enabled false   ║\n'
        '║                                                      ║\n'
        '║    # RDC compensating:                               ║\n'
        '║    ros2 param set /command_mixer rdc_enabled true    ║\n'
        '╚══════════════════════════════════════════════════════╝'
    ))

    # ── Assemble LaunchDescription ────────────────────────────────────────────

    return LaunchDescription([
        # Arguments
        arg_disturbance_type,
        arg_magnitude,
        arg_burst_duration,
        arg_burst_min,
        arg_burst_max,
        arg_session_label,

        # Startup banner
        startup_msg,

        # Nodes (in startup order)
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
