# tiny_inference_ros

This package bridges the tiny-inference JSON plan into a deliberately small Panda-arm demo.

The first node expands symbolic actions:

```json
{ "action": "PICK", "object": "red_box" }
```

into low-level demo commands:

```text
move above red_box -> open hand -> lower -> close hand -> lift
```

It can run in `dry_run` mode with no robot controllers, or send `FollowJointTrajectory`
goals to Panda arm and hand controllers when a MoveIt/Gazebo setup exposes them.

## 1. Install ROS 2 and MoveIt 2

On Ubuntu 24.04, use ROS 2 Jazzy:

```bash
sudo apt update
sudo apt install ros-jazzy-desktop ros-dev-tools
source /opt/ros/jazzy/setup.bash
```

Install MoveIt and common controller messages:

```bash
sudo apt install \
  ros-jazzy-moveit \
  ros-jazzy-moveit-py \
  ros-jazzy-moveit-resources-panda-moveit-config \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-ros2controlcli \
  ros-jazzy-controller-manager \
  ros-jazzy-control-msgs \
  ros-jazzy-ros-gz-bridge \
  ros-jazzy-ros-gz-interfaces \
  ros-jazzy-rosgraph-msgs \
  ros-jazzy-joint-state-broadcaster \
  ros-jazzy-joint-trajectory-controller
```

For Gazebo, prefer modern Gazebo/GZ rather than Gazebo Classic:

```bash
sudo apt install \
  ros-jazzy-ros-gz \
  ros-jazzy-gz-ros2-control \
  ros-jazzy-robot-state-publisher \
  ros-jazzy-xacro
```

Package names can vary by ROS distribution. If a package is not found, check:

```bash
apt search ros-jazzy | grep -E 'moveit|panda|gz-ros2-control'
```

## 2. Build this demo package

From the repo root:

```bash
cd /home/michael/hpml-project/tiny-inference/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 3. Dry-run the symbolic-to-motion expansion

This checks the ROS node and plan expansion before any controller exists:

```bash
ros2 launch tiny_inference_ros scripted_demo.launch.py dry_run:=true
```

You should see a sequence like:

```text
arm: red_box_pregrasp
hand: open
arm: red_box_grasp
hand: closed
...
```

## 4. Connect your LLM plan

Use your existing prompt runner to generate one compact JSON plan:

```bash
cd /home/michael/hpml-project/tiny-inference
source .venv/bin/activate
printf '%s\n' etc/transform_prompt quit | \
  python src/prompt_inference_server.py --scenario optimized --device cuda --warmup-runs 0 \
  > /tmp/tiny_inference_plan.json
```

Pass that generated plan, or use the checked-in demo plan:

```bash
ros2 launch tiny_inference_ros scripted_demo.launch.py \
  dry_run:=true \
  plan_file:=/tmp/tiny_inference_plan.json
```

You can also let the ROS node start the prompt server itself:

```bash
ros2 run tiny_inference_ros scripted_pick_place --ros-args \
  -p dry_run:=true \
  -p prompt_file:=/home/michael/hpml-project/tiny-inference/etc/transform_prompt \
  -p prompt_server_command:="/home/michael/hpml-project/tiny-inference/.venv/bin/python /home/michael/hpml-project/tiny-inference/src/prompt_inference_server.py --scenario optimized --device cuda --warmup-runs 0"
```

## 5. Run against controllers

The simplest demo is intentionally symbolic. It starts Gazebo, shows two tables,
two boxes, and a small primitive yellow "hand", then moves the hand and boxes with
Gazebo's `/world/default/set_pose` service. After a pick, the selected box follows
the yellow hand on each motion step until the place step releases it.

```bash
ros2 launch tiny_inference_ros symbolic_demo.launch.py
```

Use your generated LLM plan:

```bash
ros2 launch tiny_inference_ros symbolic_demo.launch.py \
  plan_file:=/tmp/tiny_inference_plan.json
```

The older Panda-like controller demo is still available. It starts Gazebo, spawns a
simple Panda-like arm, loads `gz_ros2_control`, activates arm and hand trajectory
controllers, bridges Gazebo's set-pose service, then runs the scripted pick/place
node. The current default command mode is still symbolic so the demo completes
reliably.

```bash
ros2 launch tiny_inference_ros gazebo_panda_demo.launch.py
```

Use your generated LLM plan instead of the checked-in demo plan:

```bash
ros2 launch tiny_inference_ros gazebo_panda_demo.launch.py \
  plan_file:=/tmp/tiny_inference_plan.json
```

To start Gazebo and the controllers without running the pick/place script yet:

```bash
ros2 launch tiny_inference_ros gazebo_panda_demo.launch.py run_script:=false
```

Then verify the controllers:

```bash
ros2 topic echo /clock --once
ros2 control list_controllers
ros2 action list | grep follow_joint_trajectory
```

Expected action servers:

```text
/panda_arm_controller/follow_joint_trajectory
/panda_hand_controller/follow_joint_trajectory
```

Before running the whole pick/place sequence, test one obvious arm motion:

```bash
ros2 run tiny_inference_ros test_arm_motion
```

If the test logs accepted/successful goals but the arm does not visibly move, inspect
joint state changes:

```bash
ros2 topic echo /joint_states
```

If `/joint_states` changes but Gazebo visuals do not, the controller is updating
state but the spawned model visuals are not following the simulated joints.

The pick/place script defaults to publishing trajectories on controller command
topics, which avoids waiting forever for action-result semantics in a demo:

```text
/panda_arm_controller/joint_trajectory
/panda_hand_controller/joint_trajectory
```

To force action-client mode instead:

```bash
ros2 launch tiny_inference_ros scripted_demo.launch.py \
  dry_run:=false \
  command_mode:=action
```

To use the primitive yellow hand with no robot controllers:

```bash
ros2 launch tiny_inference_ros scripted_demo.launch.py \
  dry_run:=false \
  command_mode:=symbolic
```

If `ros2 control list_controllers` waits for `/controller_manager/list_controllers`,
the Gazebo launch is not currently exposing a controller manager. Keep the Gazebo
launch running in one terminal, and inspect from a second terminal:

```bash
ros2 node list | grep controller
ros2 service list | grep controller_manager
ros2 topic list | grep robot_description
ros2 action list
```

Also check the Gazebo launch terminal for errors from `create`, `gz_ros2_control`,
or `spawner`. The common failure path is: robot did not spawn, so the
`gz_ros2_control` plugin never loaded, so `/controller_manager` never appeared.
The launch writes the generated robot file here, which is useful for debugging
spawn issues:

```bash
ls -lh /tmp/tiny_inference_ros_simple_panda.urdf
```

Run the script manually after the controllers are active:

```bash
ros2 launch tiny_inference_ros scripted_demo.launch.py dry_run:=false
```

If you run the script manually and want the boxes to move too, start the set-pose
bridge in another terminal:

```bash
ros2 run ros_gz_bridge parameter_bridge \
  /world/default/set_pose@ros_gz_interfaces/srv/SetEntityPose
```

Then run:

```bash
ros2 launch tiny_inference_ros scripted_demo.launch.py \
  dry_run:=false \
  use_gazebo_object_moves:=true
```

If your controller action names differ, list them:

```bash
ros2 action list
```

Then override the launch arguments:

```bash
ros2 launch tiny_inference_ros scripted_demo.launch.py \
  dry_run:=false \
  arm_action:=/your_arm_controller/follow_joint_trajectory \
  hand_action:=/your_hand_controller/follow_joint_trajectory
```

## 6. What to tune next

The waypoint joint poses live in:

```text
tiny_inference_ros/scripted_pick_place_node.py
```

Tune `ARM_POSES` in RViz until each symbolic target lines up with your world:

- `red_box_pregrasp`
- `red_box_grasp`
- `blue_box_pregrasp`
- `blue_box_grasp`
- `round_table_preplace`
- `round_table_place`
- `square_table_preplace`
- `square_table_place`

This scripted version is intentionally simple. Once it works, the natural upgrade is
MoveIt Task Constructor, where each symbolic `PICK`/`PLACE` becomes a real grasp and
place task with collision objects, attach/detach stages, and IK-generated poses.
