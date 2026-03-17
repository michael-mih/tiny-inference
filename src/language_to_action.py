import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_DEVICE = "cuda"
VALID_ACTIONS = {
    "PICK": {"object"},
    "MOVE_TO": {"location"},
    "PLACE": {"object", "location"},
}

_MODEL = None
_TOKENIZER = None
_MODEL_NAME = None
_DEVICE = None


def load_prompt(path="etc/transform_prompt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_model(model_name=DEFAULT_MODEL_NAME, device=DEFAULT_DEVICE, torch_dtype=torch.float16):
    global _MODEL, _TOKENIZER, _MODEL_NAME, _DEVICE

    if _MODEL is not None and _TOKENIZER is not None and _MODEL_NAME == model_name and _DEVICE == device:
        return _TOKENIZER, _MODEL

    _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    _MODEL = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    ).to(device)
    _MODEL_NAME = model_name
    _DEVICE = device
    return _TOKENIZER, _MODEL


def build_prompt(instruction, tokenizer):
    messages = [
        {"role": "system", "content": "You translate robot instructions into a strict JSON action plan."},
        {"role": "user", "content": instruction},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_json_object(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Model response does not contain a JSON object.")
    return text[start : end + 1]


def build_json_repair_prompt(invalid_json_text):
    return (
        "Repair the following malformed JSON so it becomes valid JSON.\n"
        "Return only valid JSON.\n"
        "Do not add markdown fences.\n"
        "Preserve the same schema with a top-level 'plan' array.\n\n"
        f"{invalid_json_text}"
    )


def validate_plan(plan):
    if not isinstance(plan, dict):
        raise ValueError("Plan must be a JSON object.")

    actions = plan.get("plan")
    if not isinstance(actions, list) or not actions:
        raise ValueError("Plan must contain a non-empty 'plan' list.")

    for index, step in enumerate(actions):
        if not isinstance(step, dict):
            raise ValueError(f"Step {index} must be an object.")

        action = step.get("action")
        if action not in VALID_ACTIONS:
            raise ValueError(f"Step {index} has invalid action: {action}")

        required_fields = VALID_ACTIONS[action]
        allowed_fields = required_fields | {"action"}
        missing_fields = [field for field in required_fields if not isinstance(step.get(field), str) or not step[field].strip()]
        extra_fields = sorted(set(step.keys()) - allowed_fields)

        if missing_fields:
            raise ValueError(f"Step {index} is missing required fields for {action}: {', '.join(missing_fields)}")
        if extra_fields:
            raise ValueError(f"Step {index} has unexpected fields: {', '.join(extra_fields)}")


def parse_and_validate_plan(text):
    json_text = extract_json_object(text)
    plan = json.loads(json_text)
    validate_plan(plan)
    return plan, json_text


def generate_raw_output(
    instruction,
    *,
    model_name=DEFAULT_MODEL_NAME,
    device=DEFAULT_DEVICE,
    torch_dtype=torch.float16,
    max_new_tokens=120,
    do_sample=False,
    temperature=None,
    top_p=None,
):
    tokenizer, model = load_model(model_name=model_name, device=device, torch_dtype=torch_dtype)
    prompt_text = build_prompt(instruction, tokenizer)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample and temperature is not None:
        generation_kwargs["temperature"] = temperature
    if do_sample and top_p is not None:
        generation_kwargs["top_p"] = top_p

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)

    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return {
        "prompt_text": prompt_text,
        "generated_text": generated_text,
        "input_token_count": int(inputs["input_ids"].shape[1]),
        "output_token_count": int(generated_ids.shape[0]),
    }


def generate_plan(
    instruction,
    *,
    model_name=DEFAULT_MODEL_NAME,
    device=DEFAULT_DEVICE,
    torch_dtype=torch.float16,
    max_new_tokens=120,
    do_sample=False,
    temperature=None,
    top_p=None,
    repair_attempts=1,
):
    result = generate_raw_output(
        instruction,
        model_name=model_name,
        device=device,
        torch_dtype=torch_dtype,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )

    try:
        plan, json_text = parse_and_validate_plan(result["generated_text"])
        result["plan"] = plan
        result["json_text"] = json_text
        return result
    except (ValueError, json.JSONDecodeError) as initial_error:
        last_error = initial_error

    for _ in range(repair_attempts):
        repair_result = generate_raw_output(
            build_json_repair_prompt(result["generated_text"]),
            model_name=model_name,
            device=device,
            torch_dtype=torch_dtype,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        try:
            plan, json_text = parse_and_validate_plan(repair_result["generated_text"])
            result["plan"] = plan
            result["json_text"] = json_text
            result["repair_generated_text"] = repair_result["generated_text"]
            return result
        except (ValueError, json.JSONDecodeError) as repair_error:
            last_error = repair_error

    raise ValueError(
        "Failed to produce valid plan JSON. "
        f"Last error: {last_error}. "
        f"Raw model output: {result['generated_text']}"
    ) from last_error
