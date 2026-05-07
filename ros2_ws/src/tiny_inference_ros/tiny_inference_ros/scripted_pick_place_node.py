import shlex
import subprocess
import threading
from pathlib import Path

import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from tiny_inference_ros.plan_schema import (
    DEMO_PLAN,
    build_motion_sequence,
    load_plan_file,
    load_plan_json,
)


ARM_JOINTS = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]

HAND_JOINTS = [
    "panda_finger_joint1",
    "panda_finger_joint2",
]

# These are intentionally simple waypoint poses for a demo. Tune them in RViz/Gazebo
# to match your actual table/object layout.
ARM_POSES = {
    "home": [0.0, -0.70, 0.0, -2.30, 0.0, 1.60, 0.80],
    "red_box_pregrasp": [-0.65, -0.85, 0.35, -2.25, -0.20, 1.70, 0.75],
    "red_box_grasp": [-0.65, -0.55, 0.35, -2.05, -0.20, 1.55, 0.75],
    "red_box_lift": [-0.65, -0.85, 0.35, -2.25, -0.20, 1.70, 0.75],
    "blue_box_pregrasp": [0.55, -0.85, -0.30, -2.25, 0.20, 1.70, 0.75],
    "blue_box_grasp": [0.55, -0.55, -0.30, -2.05, 0.20, 1.55, 0.75],
    "blue_box_lift": [0.55, -0.85, -0.30, -2.25, 0.20, 1.70, 0.75],
    "round_table_preplace": [-0.20, -0.45, 0.20, -1.95, 0.0, 1.55, 0.80],
    "round_table_place": [-0.20, -0.20, 0.20, -1.75, 0.0, 1.35, 0.80],
    "square_table_preplace": [0.35, -0.45, -0.20, -1.95, 0.0, 1.55, 0.80],
    "square_table_place": [0.35, -0.20, -0.20, -1.75, 0.0, 1.35, 0.80],
}

HAND_POSES = {
    "open": [0.04, 0.04],
    "closed": [0.0, 0.0],
}

SYMBOLIC_HAND_MODEL = "symbolic_hand"
SYMBOLIC_HAND_POSES = {
    "home": (0.10, 0.00, 0.85),
    "red_box_pregrasp": (0.45, 0.45, 0.35),
    "red_box_grasp": (0.45, 0.45, 0.17),
    "red_box_lift": (0.45, 0.45, 0.55),
    "blue_box_pregrasp": (0.45, -0.30, 0.35),
    "blue_box_grasp": (0.45, -0.30, 0.17),
    "blue_box_lift": (0.45, -0.30, 0.55),
    "round_table_preplace": (0.75, 0.15, 0.72),
    "round_table_place": (0.75, 0.15, 0.53),
    "square_table_preplace": (0.75, -0.55, 0.72),
    "square_table_place": (0.75, -0.55, 0.53),
}

# Gazebo model poses for the fake-but-reliable pick/place visual. The arm moves
# by controllers; these service calls make the lightweight boxes complete the demo.
GAZEBO_OBJECT_POSES = {
    "red_box_held": ("red_box", (0.45, 0.45, 0.55)),
    "blue_box_held": ("blue_box", (0.45, -0.30, 0.55)),
    "red_box_over_round_table": ("red_box", (0.75, 0.15, 0.72)),
    "red_box_at_round_table": ("red_box", (0.75, 0.15, 0.46)),
    "red_box_over_square_table": ("red_box", (0.75, -0.55, 0.72)),
    "red_box_at_square_table": ("red_box", (0.75, -0.55, 0.46)),
    "blue_box_over_round_table": ("blue_box", (0.75, 0.15, 0.72)),
    "blue_box_at_round_table": ("blue_box", (0.75, 0.15, 0.46)),
    "blue_box_over_square_table": ("blue_box", (0.75, -0.55, 0.72)),
    "blue_box_at_square_table": ("blue_box", (0.75, -0.55, 0.46)),
}


class ScriptedPickPlaceNode(Node):
    def __init__(self):
        super().__init__("scripted_pick_place")
        self.declare_parameter("dry_run", True)
        self.declare_parameter("plan_file", "")
        self.declare_parameter("plan_json", "")
        self.declare_parameter("prompt_server_command", "")
        self.declare_parameter("prompt_file", "")
        self.declare_parameter("arm_action", "/panda_arm_controller/follow_joint_trajectory")
        self.declare_parameter("hand_action", "/panda_hand_controller/follow_joint_trajectory")
        self.declare_parameter("command_mode", "topic")
        self.declare_parameter("arm_topic", "/panda_arm_controller/joint_trajectory")
        self.declare_parameter("hand_topic", "/panda_hand_controller/joint_trajectory")
        self.declare_parameter("use_gazebo_object_moves", True)
        self.declare_parameter("gazebo_set_pose_service", "/world/default/set_pose")
        self.declare_parameter("controller_timeout_sec", 20.0)
        self.declare_parameter("trajectory_result_timeout_sec", 15.0)
        self.declare_parameter("arm_step_duration_sec", 2.0)
        self.declare_parameter("hand_step_duration_sec", 0.8)

        self.dry_run = self.get_bool_parameter("dry_run")
        self.arm_action_name = str(self.get_parameter("arm_action").value)
        self.hand_action_name = str(self.get_parameter("hand_action").value)
        self.command_mode = str(self.get_parameter("command_mode").value).strip().lower()
        self.arm_topic = str(self.get_parameter("arm_topic").value)
        self.hand_topic = str(self.get_parameter("hand_topic").value)
        self.use_gazebo_object_moves = self.get_bool_parameter("use_gazebo_object_moves")
        self.gazebo_set_pose_service = str(self.get_parameter("gazebo_set_pose_service").value)
        self.controller_timeout_sec = float(self.get_parameter("controller_timeout_sec").value)
        self.trajectory_result_timeout_sec = float(
            self.get_parameter("trajectory_result_timeout_sec").value
        )
        self.arm_step_duration_sec = float(self.get_parameter("arm_step_duration_sec").value)
        self.hand_step_duration_sec = float(self.get_parameter("hand_step_duration_sec").value)

        self.arm_client = None
        self.hand_client = None
        self.arm_publisher = None
        self.hand_publisher = None
        self.set_pose_client = None
        self.last_symbolic_hand_target = "home"
        if not self.dry_run:
            if self.command_mode == "action":
                self.arm_client = ActionClient(
                    self,
                    FollowJointTrajectory,
                    self.arm_action_name,
                )
                self.hand_client = ActionClient(
                    self,
                    FollowJointTrajectory,
                    self.hand_action_name,
                )
            elif self.command_mode == "topic":
                self.arm_publisher = self.create_publisher(JointTrajectory, self.arm_topic, 10)
                self.hand_publisher = self.create_publisher(JointTrajectory, self.hand_topic, 10)
            elif self.command_mode == "symbolic":
                self.use_gazebo_object_moves = True
            else:
                raise ValueError("command_mode must be 'symbolic', 'topic', or 'action'.")
            if self.use_gazebo_object_moves:
                self.set_pose_client = self.create_client(
                    SetEntityPose,
                    self.gazebo_set_pose_service,
                )

    def get_bool_parameter(self, name):
        value = self.get_parameter(name).value
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def load_plan(self):
        plan_json = str(self.get_parameter("plan_json").value).strip()
        plan_file = str(self.get_parameter("plan_file").value).strip()
        prompt_server_command = str(self.get_parameter("prompt_server_command").value).strip()

        if plan_json:
            return load_plan_json(plan_json)
        if plan_file:
            return load_plan_file(plan_file)
        if prompt_server_command:
            return self.request_plan_from_prompt_server(prompt_server_command)

        self.get_logger().warn("No plan source configured; using the built-in red/blue box demo plan.")
        return DEMO_PLAN

    def request_plan_from_prompt_server(self, command):
        prompt_file = str(self.get_parameter("prompt_file").value).strip()
        if not prompt_file:
            raise ValueError("prompt_file must be set when prompt_server_command is set.")
        if not Path(prompt_file).expanduser().is_file():
            raise FileNotFoundError(f"Prompt file does not exist: {prompt_file}")

        self.get_logger().info(f"Starting prompt server: {command}")
        process = subprocess.Popen(
            shlex.split(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stderr_lines = []

        def drain_stderr():
            for stderr_line in process.stderr:
                line = stderr_line.rstrip()
                stderr_lines.append(line)
                self.get_logger().info(f"prompt server: {line}")

        stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
        stderr_thread.start()

        try:
            process.stdin.write(f"{prompt_file}\nquit\n")
            process.stdin.flush()
            line = process.stdout.readline().strip()
            if not line:
                stderr = "\n".join(stderr_lines[-20:])
                raise RuntimeError(f"Prompt server did not return a plan. stderr: {stderr}")
            return load_plan_json(line)
        finally:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()

    def execute(self):
        plan = self.load_plan()
        commands = build_motion_sequence(plan)
        self.get_logger().info(f"Loaded {len(plan['plan'])} symbolic steps.")
        self.get_logger().info(f"Expanded to {len(commands)} low-level demo commands.")

        if self.dry_run:
            for index, command in enumerate(commands, start=1):
                self.get_logger().info(
                    f"{index:02d}. {command.subsystem}: {command.target} ({command.label})"
                )
            return

        if self.command_mode == "action":
            self.wait_for_controller(self.arm_client, "arm", self.arm_action_name)
            self.wait_for_controller(self.hand_client, "hand", self.hand_action_name)
        elif self.command_mode == "symbolic":
            self.get_logger().info(
                f"Using symbolic command mode: moving Gazebo model '{SYMBOLIC_HAND_MODEL}'"
            )
        else:
            self.get_logger().info(
                f"Using topic command mode: arm={self.arm_topic} hand={self.hand_topic}"
            )
        if self.use_gazebo_object_moves:
            self.wait_for_gazebo_set_pose_service()

        for index, command in enumerate(commands, start=1):
            self.get_logger().info(
                f"{index:02d}/{len(commands)} {command.subsystem}: {command.target} ({command.label})"
            )
            if command.subsystem == "arm":
                if self.command_mode == "symbolic":
                    self.set_symbolic_hand_pose(command.target, self.arm_step_duration_sec)
                else:
                    self.send_trajectory_command(
                        self.arm_client if self.command_mode == "action" else self.arm_publisher,
                        self.arm_action_name if self.command_mode == "action" else self.arm_topic,
                        ARM_JOINTS,
                        ARM_POSES[command.target],
                        self.arm_step_duration_sec,
                    )
            elif command.subsystem == "hand":
                if self.command_mode == "symbolic":
                    self.get_logger().info(f"Symbolic hand state: {command.target}")
                    self.sleep_for_seconds(self.hand_step_duration_sec)
                else:
                    self.send_trajectory_command(
                        self.hand_client if self.command_mode == "action" else self.hand_publisher,
                        self.hand_action_name if self.command_mode == "action" else self.hand_topic,
                        HAND_JOINTS,
                        HAND_POSES[command.target],
                        self.hand_step_duration_sec,
                    )
            elif command.subsystem == "object":
                if self.use_gazebo_object_moves:
                    self.set_gazebo_object_pose(command.target)
            else:
                raise ValueError(f"Unknown subsystem: {command.subsystem}")

    def wait_for_controller(self, client, label, action_name):
        self.get_logger().info(f"Waiting for {label} trajectory action server: {action_name}")
        if client.wait_for_server(timeout_sec=self.controller_timeout_sec):
            return

        available_actions = self.format_available_actions()
        raise RuntimeError(
            f"Timed out waiting for {label} trajectory action server: {action_name}\n"
            "This usually means the robot simulation/controllers are not running yet, "
            "or the action name does not match your controller.\n"
            f"Available action servers:\n{available_actions}"
        )

    def format_available_actions(self):
        action_names_and_types = self.get_action_names_and_types()
        if not action_names_and_types:
            return "  none"

        lines = []
        for name, action_types in sorted(action_names_and_types):
            lines.append(f"  {name} [{', '.join(action_types)}]")
        return "\n".join(lines)

    def wait_for_gazebo_set_pose_service(self):
        self.get_logger().info(f"Waiting for Gazebo set-pose service: {self.gazebo_set_pose_service}")
        if self.set_pose_client.wait_for_service(timeout_sec=self.controller_timeout_sec):
            return

        raise RuntimeError(
            f"Timed out waiting for Gazebo set-pose service: {self.gazebo_set_pose_service}\n"
            "Start the ros_gz_bridge service bridge, for example:\n"
            "  ros2 run ros_gz_bridge parameter_bridge "
            "/world/default/set_pose@ros_gz_interfaces/srv/SetEntityPose\n"
            "Or launch gazebo_panda_demo.launch.py, which starts that bridge."
        )

    def set_gazebo_object_pose(self, target):
        if target not in GAZEBO_OBJECT_POSES:
            raise ValueError(f"No Gazebo object pose is defined for target: {target}")

        model_name, position = GAZEBO_OBJECT_POSES[target]
        self.set_gazebo_model_pose(model_name, position)

    def set_symbolic_hand_pose(self, target, duration_sec):
        if target not in SYMBOLIC_HAND_POSES:
            raise ValueError(f"No symbolic hand pose is defined for target: {target}")

        self.last_symbolic_hand_target = target
        self.set_gazebo_model_pose(SYMBOLIC_HAND_MODEL, SYMBOLIC_HAND_POSES[target])
        self.sleep_for_seconds(duration_sec)

    def set_gazebo_model_pose(self, model_name, position):
        request = SetEntityPose.Request()
        request.entity = Entity()
        request.entity.name = model_name
        request.entity.type = Entity.MODEL
        request.pose = self.build_pose(position)

        future = self.set_pose_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError(f"Gazebo failed to set pose for {model_name}.")

    def sleep_for_seconds(self, seconds):
        self.get_clock().sleep_for(Duration(seconds=float(seconds)))

    def build_pose(self, position):
        pose = Pose()
        pose.position.x = float(position[0])
        pose.position.y = float(position[1])
        pose.position.z = float(position[2])
        pose.orientation.w = 1.0
        return pose

    def send_trajectory_command(self, target, target_name, joint_names, positions, duration_sec):
        if self.command_mode == "topic":
            self.publish_trajectory(target, target_name, joint_names, positions, duration_sec)
            return
        self.send_trajectory_action(target, joint_names, positions, duration_sec)

    def build_trajectory(self, joint_names, positions, duration_sec):
        trajectory = JointTrajectory()
        trajectory.joint_names = list(joint_names)
        trajectory.header.stamp = (
            self.get_clock().now() + Duration(seconds=0.2)
        ).to_msg()

        point = JointTrajectoryPoint()
        point.positions = [float(position) for position in positions]
        point.time_from_start = Duration(seconds=duration_sec).to_msg()
        trajectory.points.append(point)
        return trajectory

    def publish_trajectory(self, publisher, topic_name, joint_names, positions, duration_sec):
        self.get_logger().info(
            f"Publishing trajectory to {topic_name} "
            f"positions={positions} duration={duration_sec:.2f}s"
        )
        publisher.publish(self.build_trajectory(joint_names, positions, duration_sec))
        rclpy.spin_once(self, timeout_sec=0.1)
        self.get_clock().sleep_for(Duration(seconds=duration_sec + 0.3))

    def send_trajectory_action(self, client, joint_names, positions, duration_sec):
        self.get_logger().info(
            f"Sending trajectory to {joint_names[0]}..{joint_names[-1]} "
            f"positions={positions} duration={duration_sec:.2f}s"
        )
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = self.build_trajectory(joint_names, positions, duration_sec)

        send_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("Trajectory goal was rejected.")
        self.get_logger().info("Trajectory goal accepted.")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self,
            result_future,
            timeout_sec=self.trajectory_result_timeout_sec,
        )
        if not result_future.done():
            self.get_logger().error(
                "Trajectory goal did not finish within "
                f"{self.trajectory_result_timeout_sec:.1f}s; canceling goal."
            )
            cancel_future = goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=5.0)
            raise RuntimeError(
                "Trajectory goal was accepted but never completed. "
                "Check /joint_states and Gazebo/controller logs."
            )

        result = result_future.result().result
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(
                f"Trajectory failed with error_code={result.error_code}: {result.error_string}"
            )
        self.get_logger().info("Trajectory goal completed successfully.")


def main(args=None):
    rclpy.init(args=args)
    node = ScriptedPickPlaceNode()
    try:
        node.execute()
    finally:
        node.destroy_node()
        rclpy.shutdown()
