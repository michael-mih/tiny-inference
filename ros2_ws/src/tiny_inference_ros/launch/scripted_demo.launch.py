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
                    }
                ],
            ),
        ]
    )
