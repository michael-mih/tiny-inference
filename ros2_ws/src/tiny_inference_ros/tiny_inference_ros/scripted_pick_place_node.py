import shlex
import subprocess
import threading
import math
from pathlib import Path

import rclpy
from geometry_msgs.msg import Pose
from rclpy.duration import Duration
from rclpy.node import Node
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose

from tiny_inference_ros.plan_schema import (
    DEMO_PLAN,
    build_motion_sequence,
    load_plan_file,
    load_plan_json,
)


SYMBOLIC_ARM_MODEL = "symbolic_arm"
SYMBOLIC_CARRIED_OBJECT_OFFSET = (0.0, 0.0, -0.12)
SYMBOLIC_MOVE_STEP_SEC = 0.08
SYMBOLIC_ARM_POSES = {
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

# Gazebo model poses for the symbolic pick/place visual.
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
        self.declare_parameter("gazebo_set_pose_service", "/world/default/set_pose")
        self.declare_parameter("service_timeout_sec", 30.0)
        self.declare_parameter("arm_step_duration_sec", 2.0)
        self.declare_parameter("hand_step_duration_sec", 0.8)

        self.dry_run = self.get_bool_parameter("dry_run")
        self.gazebo_set_pose_service = str(self.get_parameter("gazebo_set_pose_service").value)
        self.service_timeout_sec = float(self.get_parameter("service_timeout_sec").value)
        self.arm_step_duration_sec = float(self.get_parameter("arm_step_duration_sec").value)
        self.hand_step_duration_sec = float(self.get_parameter("hand_step_duration_sec").value)

        self.set_pose_client = None
        self.prompt_server_command = None
        self.prompt_server_process = None
        self.prompt_server_stderr_lines = []
        self.prompt_server_stderr_thread = None
        self.last_symbolic_arm_position = SYMBOLIC_ARM_POSES["home"]
        self.symbolic_held_object = None
        if not self.dry_run:
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

    def prompt_server_is_running(self):
        return (
            self.prompt_server_process is not None
            and self.prompt_server_process.poll() is None
        )

    def ensure_prompt_server(self, command):
        if self.prompt_server_is_running() and self.prompt_server_command == command:
            return self.prompt_server_process

        self.stop_prompt_server()
        self.get_logger().info(f"Starting prompt server: {command}")
        process = subprocess.Popen(
            shlex.split(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.prompt_server_command = command
        self.prompt_server_process = process
        self.prompt_server_stderr_lines = []

        def drain_stderr():
            for stderr_line in process.stderr:
                line = stderr_line.rstrip()
                self.prompt_server_stderr_lines.append(line)
                self.get_logger().info(f"prompt server: {line}")

        self.prompt_server_stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
        self.prompt_server_stderr_thread.start()
        return process

    def request_plan_from_prompt_server(self, command):
        prompt_file = str(self.get_parameter("prompt_file").value).strip()
        if not prompt_file:
            raise ValueError("prompt_file must be set when prompt_server_command is set.")
        if not Path(prompt_file).expanduser().is_file():
            raise FileNotFoundError(f"Prompt file does not exist: {prompt_file}")

        process = self.ensure_prompt_server(command)

        try:
            process.stdin.write(f"{prompt_file}\n")
            process.stdin.flush()
        except BrokenPipeError as exc:
            stderr = "\n".join(self.prompt_server_stderr_lines[-20:])
            self.stop_prompt_server()
            raise RuntimeError(f"Prompt server exited before accepting a prompt. stderr: {stderr}") from exc

        line = process.stdout.readline().strip()
        if not line:
            stderr = "\n".join(self.prompt_server_stderr_lines[-20:])
            exit_code = process.poll()
            self.stop_prompt_server()
            raise RuntimeError(
                f"Prompt server did not return a plan. exit_code={exit_code} stderr: {stderr}"
            )
        return load_plan_json(line)

    def stop_prompt_server(self):
        process = self.prompt_server_process
        self.prompt_server_process = None
        self.prompt_server_command = None
        if process is None:
            return

        if process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write("quit\n")
                    process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()

    def destroy_node(self):
        self.stop_prompt_server()
        return super().destroy_node()

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

        self.get_logger().info(
            f"Using symbolic arm: moving Gazebo model '{SYMBOLIC_ARM_MODEL}'"
        )
        self.wait_for_gazebo_set_pose_service()

        for index, command in enumerate(commands, start=1):
            self.get_logger().info(
                f"{index:02d}/{len(commands)} {command.subsystem}: {command.target} ({command.label})"
            )
            if command.subsystem == "arm":
                self.set_symbolic_arm_pose(command.target, self.arm_step_duration_sec)
            elif command.subsystem == "hand":
                self.get_logger().info(f"Symbolic gripper state: {command.target}")
                self.sleep_for_seconds(self.hand_step_duration_sec)
            elif command.subsystem == "object":
                self.set_gazebo_object_pose(command.target)
            else:
                raise ValueError(f"Unknown subsystem: {command.subsystem}")

    def wait_for_gazebo_set_pose_service(self):
        self.get_logger().info(f"Waiting for Gazebo set-pose service: {self.gazebo_set_pose_service}")
        if self.set_pose_client.wait_for_service(timeout_sec=self.service_timeout_sec):
            return

        raise RuntimeError(
            f"Timed out waiting for Gazebo set-pose service: {self.gazebo_set_pose_service}\n"
            "Start the ros_gz_bridge service bridge, for example:\n"
            "  ros2 run ros_gz_bridge parameter_bridge "
            "/world/default/set_pose@ros_gz_interfaces/srv/SetEntityPose"
        )

    def set_gazebo_object_pose(self, target):
        if target not in GAZEBO_OBJECT_POSES:
            raise ValueError(f"No Gazebo object pose is defined for target: {target}")

        model_name, position = GAZEBO_OBJECT_POSES[target]
        self.set_gazebo_model_pose(model_name, position)
        self.update_symbolic_grasp_state(target, model_name)

    def update_symbolic_grasp_state(self, target, model_name):
        if target.endswith("_held"):
            self.symbolic_held_object = model_name
            self.get_logger().info(f"Symbolic grasp attached: {model_name}")
            return

        if "_at_" in target:
            self.get_logger().info(f"Symbolic grasp released: {model_name}")
            self.symbolic_held_object = None

    def set_symbolic_arm_pose(self, target, duration_sec):
        if target not in SYMBOLIC_ARM_POSES:
            raise ValueError(f"No symbolic arm pose is defined for target: {target}")

        start_position = self.last_symbolic_arm_position
        end_position = SYMBOLIC_ARM_POSES[target]
        step_count = self.symbolic_step_count(start_position, end_position, duration_sec)

        for step_index in range(1, step_count + 1):
            fraction = step_index / step_count
            arm_position = self.interpolate_position(start_position, end_position, fraction)
            self.set_gazebo_model_pose(SYMBOLIC_ARM_MODEL, arm_position)
            if self.symbolic_held_object is not None:
                self.set_gazebo_model_pose(
                    self.symbolic_held_object,
                    self.apply_offset(arm_position, SYMBOLIC_CARRIED_OBJECT_OFFSET),
                )
            self.sleep_for_seconds(SYMBOLIC_MOVE_STEP_SEC)

        self.last_symbolic_arm_position = end_position

    def symbolic_step_count(self, start_position, end_position, duration_sec):
        distance = math.dist(start_position, end_position)
        distance_steps = max(1, math.ceil(distance / 0.04))
        time_steps = max(1, math.ceil(float(duration_sec) / SYMBOLIC_MOVE_STEP_SEC))
        return max(distance_steps, time_steps)

    def interpolate_position(self, start_position, end_position, fraction):
        return tuple(
            start_value + (end_value - start_value) * fraction
            for start_value, end_value in zip(start_position, end_position)
        )

    def apply_offset(self, position, offset):
        return tuple(position_value + offset_value for position_value, offset_value in zip(position, offset))

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


def main(args=None):
    rclpy.init(args=args)
    node = ScriptedPickPlaceNode()
    try:
        node.execute()
    finally:
        node.destroy_node()
        rclpy.shutdown()
