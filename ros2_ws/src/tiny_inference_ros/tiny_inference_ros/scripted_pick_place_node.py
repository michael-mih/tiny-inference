import shlex
import subprocess
import threading
from pathlib import Path

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
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
        self.declare_parameter("arm_step_duration_sec", 2.0)
        self.declare_parameter("hand_step_duration_sec", 0.8)

        self.dry_run = self.get_bool_parameter("dry_run")
        self.arm_step_duration_sec = float(self.get_parameter("arm_step_duration_sec").value)
        self.hand_step_duration_sec = float(self.get_parameter("hand_step_duration_sec").value)

        self.arm_client = None
        self.hand_client = None
        if not self.dry_run:
            self.arm_client = ActionClient(
                self,
                FollowJointTrajectory,
                str(self.get_parameter("arm_action").value),
            )
            self.hand_client = ActionClient(
                self,
                FollowJointTrajectory,
                str(self.get_parameter("hand_action").value),
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

        self.wait_for_controller(self.arm_client, "arm")
        self.wait_for_controller(self.hand_client, "hand")

        for index, command in enumerate(commands, start=1):
            self.get_logger().info(
                f"{index:02d}/{len(commands)} {command.subsystem}: {command.target} ({command.label})"
            )
            if command.subsystem == "arm":
                self.send_trajectory(
                    self.arm_client,
                    ARM_JOINTS,
                    ARM_POSES[command.target],
                    self.arm_step_duration_sec,
                )
            elif command.subsystem == "hand":
                self.send_trajectory(
                    self.hand_client,
                    HAND_JOINTS,
                    HAND_POSES[command.target],
                    self.hand_step_duration_sec,
                )
            else:
                raise ValueError(f"Unknown subsystem: {command.subsystem}")

    def wait_for_controller(self, client, label):
        self.get_logger().info(f"Waiting for {label} trajectory action server...")
        if not client.wait_for_server(timeout_sec=20.0):
            raise RuntimeError(f"Timed out waiting for {label} trajectory action server.")

    def send_trajectory(self, client, joint_names, positions, duration_sec):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = list(joint_names)
        goal.trajectory.header.stamp = (
            self.get_clock().now() + Duration(seconds=0.2)
        ).to_msg()

        point = JointTrajectoryPoint()
        point.positions = [float(position) for position in positions]
        point.time_from_start = Duration(seconds=duration_sec).to_msg()
        goal.trajectory.points.append(point)

        send_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("Trajectory goal was rejected.")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(
                f"Trajectory failed with error_code={result.error_code}: {result.error_string}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = ScriptedPickPlaceNode()
    try:
        node.execute()
    finally:
        node.destroy_node()
        rclpy.shutdown()
