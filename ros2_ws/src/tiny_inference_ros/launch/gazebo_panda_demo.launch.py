import os
import tempfile

import xacro
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    SetLaunchConfiguration,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def prepare_robot_description(context):
    package_share = FindPackageShare("tiny_inference_ros").perform(context)
    robot_xacro = os.path.join(package_share, "urdf", "simple_panda.urdf.xacro")
    controllers_file = os.path.join(package_share, "config", "panda_gazebo_controllers.yaml")

    robot_doc = xacro.process_file(
        robot_xacro,
        mappings={"controllers_file": controllers_file},
    )
    robot_description = robot_doc.toprettyxml(indent="  ")

    robot_description_file = os.path.join(
        tempfile.gettempdir(),
        "tiny_inference_ros_simple_panda.urdf",
    )
    with open(robot_description_file, "w", encoding="utf-8") as output_file:
        output_file.write(robot_description)

    return [
        LogInfo(msg=f"Generated robot URDF: {robot_description_file}"),
        SetLaunchConfiguration("robot_description", robot_description),
        SetLaunchConfiguration("robot_description_file", robot_description_file),
    ]


def generate_launch_description():
    package_share = FindPackageShare("tiny_inference_ros")
    world = PathJoinSubstitution([package_share, "worlds", "panda_demo.world.sdf"])
    demo_plan = PathJoinSubstitution([package_share, "config", "demo_plan.json"])

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [PathJoinSubstitution([FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"])]
        ),
        launch_arguments={"gz_args": ["-r -v 3 ", world]}.items(),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": ParameterValue(
                    LaunchConfiguration("robot_description"),
                    value_type=str,
                ),
                "use_sim_time": True,
            }
        ],
    )

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-world",
            "default",
            "-name",
            "tiny_panda",
            "-file",
            LaunchConfiguration("robot_description_file"),
            "-x",
            "0.0",
            "-y",
            "0.0",
            "-z",
            "0.0",
        ],
    )

    spawn_joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "60",
        ],
    )

    spawn_arm_controller = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=[
            "panda_arm_controller",
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "60",
        ],
    )

    spawn_hand_controller = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=[
            "panda_hand_controller",
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "60",
        ],
    )

    scripted_pick_place = Node(
        package="tiny_inference_ros",
        executable="scripted_pick_place",
        condition=IfCondition(LaunchConfiguration("run_script")),
        output="screen",
        parameters=[
            {
                "dry_run": False,
                "plan_file": LaunchConfiguration("plan_file"),
                "arm_action": "/panda_arm_controller/follow_joint_trajectory",
                "hand_action": "/panda_hand_controller/follow_joint_trajectory",
                "controller_timeout_sec": 60.0,
                "arm_step_duration_sec": 2.0,
                "hand_step_duration_sec": 0.8,
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("plan_file", default_value=demo_plan),
            DeclareLaunchArgument(
                "run_script",
                default_value="true",
                description="Whether to start the scripted pick/place node after controllers are active.",
            ),
            OpaqueFunction(function=prepare_robot_description),
            gazebo,
            robot_state_publisher,
            TimerAction(period=2.0, actions=[spawn_robot]),
            TimerAction(period=5.0, actions=[spawn_joint_state_broadcaster]),
            TimerAction(period=6.0, actions=[spawn_arm_controller, spawn_hand_controller]),
            TimerAction(period=9.0, actions=[scripted_pick_place]),
        ]
    )
