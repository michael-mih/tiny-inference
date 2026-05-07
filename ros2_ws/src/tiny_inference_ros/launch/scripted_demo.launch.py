from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_plan_file = PathJoinSubstitution(
        [FindPackageShare("tiny_inference_ros"), "config", "demo_plan.json"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("dry_run", default_value="true"),
            DeclareLaunchArgument("plan_file", default_value=default_plan_file),
            DeclareLaunchArgument(
                "arm_action",
                default_value="/panda_arm_controller/follow_joint_trajectory",
            ),
            DeclareLaunchArgument(
                "hand_action",
                default_value="/panda_hand_controller/follow_joint_trajectory",
            ),
            DeclareLaunchArgument("command_mode", default_value="topic"),
            DeclareLaunchArgument("arm_topic", default_value="/panda_arm_controller/joint_trajectory"),
            DeclareLaunchArgument("hand_topic", default_value="/panda_hand_controller/joint_trajectory"),
            DeclareLaunchArgument("use_gazebo_object_moves", default_value="true"),
            DeclareLaunchArgument("gazebo_set_pose_service", default_value="/world/default/set_pose"),
            DeclareLaunchArgument("controller_timeout_sec", default_value="20.0"),
            DeclareLaunchArgument("trajectory_result_timeout_sec", default_value="15.0"),
            Node(
                package="tiny_inference_ros",
                executable="scripted_pick_place",
                output="screen",
                parameters=[
                    {
                        "dry_run": ParameterValue(LaunchConfiguration("dry_run"), value_type=bool),
                        "plan_file": LaunchConfiguration("plan_file"),
                        "arm_action": LaunchConfiguration("arm_action"),
                        "hand_action": LaunchConfiguration("hand_action"),
                        "command_mode": LaunchConfiguration("command_mode"),
                        "arm_topic": LaunchConfiguration("arm_topic"),
                        "hand_topic": LaunchConfiguration("hand_topic"),
                        "use_gazebo_object_moves": ParameterValue(
                            LaunchConfiguration("use_gazebo_object_moves"),
                            value_type=bool,
                        ),
                        "gazebo_set_pose_service": LaunchConfiguration("gazebo_set_pose_service"),
                        "controller_timeout_sec": LaunchConfiguration("controller_timeout_sec"),
                        "trajectory_result_timeout_sec": LaunchConfiguration(
                            "trajectory_result_timeout_sec"
                        ),
                    }
                ],
            ),
        ]
    )
