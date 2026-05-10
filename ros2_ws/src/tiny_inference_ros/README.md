# tiny_inference_ros

This package is the symbolic ROS 2/Gazebo interaction layer for the Tiny Inference project. It consumes the JSON action plans produced by `src/prompt_inference_server.py`, expands them into simple pick/place motions, and moves a lightweight symbolic arm plus objects in Gazebo through `/world/default/set_pose`.

Supported symbolic actions:

```json
{ "action": "PICK", "object": "red_box" }
```

The checked-in demo world contains:

- `symbolic_arm`
- `red_box`
- `blue_box`
- `round_table`
- `square_table`

## 1. Install ROS 2 and Gazebo

On Ubuntu 24.04 with ROS 2 Jazzy:

```bash
sudo apt update
sudo apt install \
  ros-jazzy-desktop \
  ros-dev-tools \
  ros-jazzy-ros-gz \
  ros-jazzy-ros-gz-bridge \
  ros-jazzy-ros-gz-interfaces \
  ros-jazzy-rosgraph-msgs
source /opt/ros/jazzy/setup.bash
```

## 2. Build the ROS Package

From the repo root:

```bash
cd /home/michael/hpml-project/tiny-inference/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 3. Run the Symbolic Demo

Launch Gazebo, bridge the set-pose service, and execute the checked-in demo plan:

```bash
ros2 launch tiny_inference_ros symbolic_demo.launch.py
```

Use a generated plan instead:

```bash
ros2 launch tiny_inference_ros symbolic_demo.launch.py \
  plan_file:=/tmp/tiny_inference_plan.json
```

## 4. Connect the Optimized Inference Path

Generate a compact JSON plan with the project prompt server:

```bash
cd /home/michael/hpml-project/tiny-inference
source .venv/bin/activate
printf '%s\n' etc/transform_prompt quit | \
  python src/prompt_inference_server.py --scenario optimized --device cuda --warmup-runs 0 \
  > /tmp/tiny_inference_plan.json
```

Then launch the symbolic ROS interaction:

```bash
cd /home/michael/hpml-project/tiny-inference/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch tiny_inference_ros symbolic_demo.launch.py \
  plan_file:=/tmp/tiny_inference_plan.json
```

The ROS node can also start the prompt server itself:

```bash
ros2 run tiny_inference_ros scripted_pick_place --ros-args \
  -p dry_run:=true \
  -p prompt_file:=/home/michael/hpml-project/tiny-inference/etc/transform_prompt \
  -p prompt_server_command:="/home/michael/hpml-project/tiny-inference/.venv/bin/python /home/michael/hpml-project/tiny-inference/src/prompt_inference_server.py --scenario optimized --device cuda --warmup-runs 0"
```

Set `dry_run:=false` only when Gazebo and the set-pose bridge are already running.

## 5. Dry Run

To validate plan parsing and motion expansion without Gazebo:

```bash
ros2 run tiny_inference_ros scripted_pick_place --ros-args \
  -p dry_run:=true \
  -p plan_file:=/home/michael/hpml-project/tiny-inference/ros2_ws/src/tiny_inference_ros/config/demo_plan.json
```

You should see low-level symbolic commands such as:

```text
arm: red_box_pregrasp
hand: open
arm: red_box_grasp
hand: closed
object: red_box_held
```

## Package Surface

The ROS package intentionally contains only the symbolic interaction path:

- `launch/symbolic_demo.launch.py`
- `worlds/symbolic_demo.world.sdf`
- `tiny_inference_ros/scripted_pick_place_node.py`
- `tiny_inference_ros/plan_schema.py`
- `config/demo_plan.json`
