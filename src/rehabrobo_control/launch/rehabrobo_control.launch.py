import launch
from launch.substitutions import Command, LaunchConfiguration
import launch_ros
import os

# define the package name
packageName = 'rehabrobo_control'

# urdf relative file path with respect to the package path
urdfRelativePath = 'model/model.urdf'

# RViz config file path with respect to the package path
rvizRelativePath = 'config/config.rviz'

# ROS2 Control file path with respect to the package path
ros2controlRelativePath = 'config/robot_controller.yaml'

def generate_launch_description():
    # absolute model path
    pkgPath = launch_ros.substitutions.FindPackageShare(package=packageName).find(packageName)
    # absolute urdf model path
    urdfModelPath = os.path.join(pkgPath, urdfRelativePath)
    # absolute rviz config file path
    rvizConfigPath = os.path.join(pkgPath, rvizRelativePath)

    # absolute ros2 controller file path
    ros2controlPath = os.path.join(pkgPath, ros2controlRelativePath)

    # here, for verfication, print the urdf model path
    print(urdfModelPath)

    # load the robot description
    with open(urdfModelPath,'r') as infp:
        robot_desc = infp.read()

    # define a parameter with the robot URDF description
    robot_description = {'robot_description': robot_desc}

    # robot state publisher node
    robot_state_publisher_node = launch_ros.actions.Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters = [robot_description]
    )

    # rviz node
    rviz_node = launch_ros.actions.Node(
        package = 'rviz2',
        executable = 'rviz2',
        name = 'rviz2',
        output = 'screen',
        arguments = ['-d', rvizConfigPath]
    )

    # ros2 control node
    control_node = launch_ros.actions.Node(
        package = 'controller_manager',
        executable = 'ros2_control_node',
        parameters=[ros2controlPath],
        output = 'both',
    )

    # joint state broadcaster
    joint_state_broadcaster_spawner = launch_ros.actions.Node(
        package = 'controller_manager',
        executable = 'spawner',
        arguments=["joint_state_broadcaster"],
    )

    # forward position controller
    robot_controller_spawner = launch_ros.actions.Node(
        package = 'controller_manager',
        executable = 'spawner',
        arguments=["forward_position_controller", "--param-file", ros2controlPath],
    )

    # finally, combine everything and return
    return launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(name='gui', default_value='True', description='This is a flag for joint_state_publisher_gui'),
        robot_state_publisher_node,
        rviz_node,
        control_node,
        joint_state_broadcaster_spawner,
        robot_controller_spawner
    ])
