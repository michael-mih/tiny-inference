from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


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

    set_pose_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=["/world/default/set_pose@ros_gz_interfaces/srv/SetEntityPose"],
    )

    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
    )

    scripted_pick_place = Node(
        package="tiny_inference_ros",
        executable="scripted_pick_place",
        output="screen",
        parameters=[
            {
                "dry_run": False,
                "plan_file": LaunchConfiguration("plan_file"),
                "command_mode": "symbolic",
                "use_gazebo_object_moves": True,
                "gazebo_set_pose_service": "/world/default/set_pose",
                "controller_timeout_sec": 30.0,
                "arm_step_duration_sec": 0.8,
                "hand_step_duration_sec": 0.25,
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("plan_file", default_value=demo_plan),
            gazebo,
            clock_bridge,
            set_pose_bridge,
            TimerAction(period=2.0, actions=[scripted_pick_place]),
        ]
    )
