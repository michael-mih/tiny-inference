import json
from dataclasses import dataclass
from pathlib import Path


VALID_ACTIONS = {
    "PICK": {"object"},
    "MOVE_TO": {"location"},
    "PLACE": {"object", "location"},
}

DEMO_PLAN = {
    "plan": [
        {"action": "PICK", "object": "red_box"},
        {"action": "MOVE_TO", "location": "round_table"},
        {"action": "PLACE", "object": "red_box", "location": "round_table"},
        {"action": "PICK", "object": "blue_box"},
        {"action": "MOVE_TO", "location": "square_table"},
        {"action": "PLACE", "object": "blue_box", "location": "square_table"},
    ]
}

KNOWN_OBJECTS = {"red_box", "blue_box"}
KNOWN_LOCATIONS = {"round_table", "square_table"}


@dataclass(frozen=True)
class MotionCommand:
    """One low-level demo command derived from a symbolic plan step."""

    subsystem: str
    target: str
    label: str


def validate_plan(plan):
    if not isinstance(plan, dict):
        raise ValueError("Plan must be a JSON object.")

    steps = plan.get("plan")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Plan must contain a non-empty 'plan' list.")

    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"Step {index} must be an object.")

        action = step.get("action")
        if action not in VALID_ACTIONS:
            raise ValueError(f"Step {index} has invalid action: {action}")

        required = VALID_ACTIONS[action]
        missing = [
            field
            for field in required
            if not isinstance(step.get(field), str) or not step[field].strip()
        ]
        extras = sorted(set(step) - (required | {"action"}))
        if missing:
            raise ValueError(f"Step {index} is missing fields for {action}: {', '.join(missing)}")
        if extras:
            raise ValueError(f"Step {index} has unexpected fields: {', '.join(extras)}")

        obj = step.get("object")
        if obj is not None and obj not in KNOWN_OBJECTS:
            raise ValueError(f"Step {index} references unknown object: {obj}")

        location = step.get("location")
        if location is not None and location not in KNOWN_LOCATIONS:
            raise ValueError(f"Step {index} references unknown location: {location}")


def load_plan_file(path):
    plan = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    validate_plan(plan)
    return plan


def load_plan_json(text):
    plan = json.loads(text)
    validate_plan(plan)
    return plan


def build_motion_sequence(plan):
    validate_plan(plan)
    commands = [MotionCommand("arm", "home", "move arm home")]
    held_object = None

    for step in plan["plan"]:
        action = step["action"]

        if action == "PICK":
            obj = step["object"]
            if held_object is not None:
                raise ValueError(f"Cannot pick {obj}; already holding {held_object}.")

            commands.extend(
                [
                    MotionCommand("arm", f"{obj}_pregrasp", f"move above {obj}"),
                    MotionCommand("hand", "open", "open gripper"),
                    MotionCommand("arm", f"{obj}_grasp", f"lower to {obj}"),
                    MotionCommand("hand", "closed", f"close gripper on {obj}"),
                    MotionCommand("arm", f"{obj}_lift", f"lift {obj}"),
                    MotionCommand("object", f"{obj}_held", f"lift {obj} in Gazebo"),
                ]
            )
            held_object = obj
            continue

        if action == "MOVE_TO":
            location = step["location"]
            commands.append(MotionCommand("arm", f"{location}_preplace", f"move above {location}"))
            if held_object is not None:
                commands.append(
                    MotionCommand(
                        "object",
                        f"{held_object}_over_{location}",
                        f"carry {held_object} above {location} in Gazebo",
                    )
                )
            continue

        if action == "PLACE":
            obj = step["object"]
            location = step["location"]
            if held_object != obj:
                raise ValueError(f"Cannot place {obj}; currently holding {held_object}.")

            commands.extend(
                [
                    MotionCommand("arm", f"{location}_place", f"lower {obj} to {location}"),
                    MotionCommand("object", f"{obj}_at_{location}", f"place {obj} on {location} in Gazebo"),
                    MotionCommand("hand", "open", f"release {obj}"),
                    MotionCommand("arm", f"{location}_preplace", f"retreat from {location}"),
                ]
            )
            held_object = None

    if held_object is not None:
        raise ValueError(f"Plan ended while still holding {held_object}.")

    commands.append(MotionCommand("arm", "home", "return arm home"))
    return commands
