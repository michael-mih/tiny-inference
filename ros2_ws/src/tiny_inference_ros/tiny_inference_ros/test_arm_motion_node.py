import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


ARM_JOINTS = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]


class TestArmMotionNode(Node):
    def __init__(self):
        super().__init__("test_arm_motion")
        self.declare_parameter("arm_action", "/panda_arm_controller/follow_joint_trajectory")
        self.declare_parameter("duration_sec", 5.0)
        self.declare_parameter("timeout_sec", 20.0)
        self.declare_parameter("result_timeout_sec", 12.0)

        self.arm_action = str(self.get_parameter("arm_action").value)
        self.duration_sec = float(self.get_parameter("duration_sec").value)
        self.timeout_sec = float(self.get_parameter("timeout_sec").value)
        self.result_timeout_sec = float(self.get_parameter("result_timeout_sec").value)
        self.client = ActionClient(self, FollowJointTrajectory, self.arm_action)

    def execute(self):
        self.get_logger().info(f"Waiting for arm trajectory action server: {self.arm_action}")
        if not self.client.wait_for_server(timeout_sec=self.timeout_sec):
            raise RuntimeError(f"Timed out waiting for {self.arm_action}")

        self.send_goal("pose_a", [0.80, -0.80, 0.40, -1.70, 0.40, 1.30, 0.20])
        self.send_goal("pose_b", [-0.80, -0.35, -0.40, -2.20, -0.40, 1.80, 1.10])
        self.send_goal("home", [0.0, -0.70, 0.0, -2.30, 0.0, 1.60, 0.80])

    def send_goal(self, label, positions):
        self.get_logger().info(f"Sending {label}: {positions}")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = list(ARM_JOINTS)
        goal.trajectory.header.stamp = (
            self.get_clock().now() + Duration(seconds=0.2)
        ).to_msg()

        point = JointTrajectoryPoint()
        point.positions = [float(position) for position in positions]
        point.time_from_start = Duration(seconds=self.duration_sec).to_msg()
        goal.trajectory.points.append(point)

        send_future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(f"{label} was rejected by the trajectory controller.")

        self.get_logger().info(f"{label} accepted; waiting for result")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self,
            result_future,
            timeout_sec=self.result_timeout_sec,
        )
        if not result_future.done():
            self.get_logger().error(
                f"{label} did not finish within {self.result_timeout_sec:.1f}s; canceling goal."
            )
            cancel_future = goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=5.0)
            raise RuntimeError(
                "Trajectory goal was accepted but never completed. "
                "Check /joint_states and Gazebo/controller logs."
            )

        action_result = result_future.result()
        result = action_result.result
        self.get_logger().info(
            f"{label} result: status={action_result.status} "
            f"error_code={result.error_code} error_string='{result.error_string}'"
        )
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(f"{label} failed with error_code={result.error_code}")


def main(args=None):
    rclpy.init(args=args)
    node = TestArmMotionNode()
    try:
        node.execute()
    finally:
        node.destroy_node()
        rclpy.shutdown()
