import gc
import importlib.util
import json
from dataclasses import dataclass
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_DEVICE = "auto"
VALID_ACTIONS = {
    "PICK": {"object"},
    "MOVE_TO": {"location"},
    "PLACE": {"object", "location"},
}

SUPPORTED_BACKENDS = {"transformers", "vllm"}
SUPPORTED_QUANTIZATION = {"none", "bitsandbytes-8bit", "bitsandbytes-4bit"}
SUPPORTED_ATTENTION_IMPLEMENTATIONS = {"default", "sdpa", "flash_attention_2"}

_BACKEND_MODEL = None
_BACKEND_CACHE_KEY = None


def get_default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class InferenceConfig:
    """Inference settings for plan generation.

    Available options by field:
    - `model_name`: any Hugging Face model id/path compatible with the selected backend.
      Default: `Qwen/Qwen2.5-3B-Instruct`.
    - `device`: `"auto"` or a torch device string such as `"cpu"`, `"cuda"`, `"cuda:0"`.
      Other torch-supported device strings may also work if the backend supports them.
    - `backend`: `"transformers"` or `"vllm"`.
    - `precision`: `"auto"`, `"float32"`, `"float16"`, or `"bfloat16"`.
    - `quantization`: `"none"`, `"bitsandbytes-8bit"`, or `"bitsandbytes-4bit"`.
    - `attention_implementation`: `"default"`, `"sdpa"`, or `"flash_attention_2"`.
    - `enable_prefix_caching`: enables vLLM automatic prefix caching when `backend="vllm"`.
    - `vllm_max_num_seqs`: maximum concurrent sequences for vLLM engine warmup/scheduling.
    - `vllm_gpu_memory_utilization`: fraction of GPU memory vLLM may reserve for KV cache.
    - `use_torch_compile`: `True` or `False`.
    - `compile_mode`: forwarded to `torch.compile(..., mode=...)` when `use_torch_compile=True`.
      Common modes include `"default"`, `"reduce-overhead"`, and `"max-autotune"`.
    """
    model_name: str = DEFAULT_MODEL_NAME
    device: str = DEFAULT_DEVICE
    backend: str = "transformers"
    precision: str = "auto"
    quantization: str = "none"
    attention_implementation: str = "default"
    enable_prefix_caching: bool = False
    vllm_max_num_seqs: int = 1
    vllm_gpu_memory_utilization: float = 0.8
    use_torch_compile: bool = False
    compile_mode: str = "reduce-overhead"

    def resolved_device(self):
        if self.device == "auto":
            return get_default_device()
        return self.device

    def cache_key(self):
        return (
            self.model_name,
            self.resolved_device(),
            self.backend,
            self.precision,
            self.quantization,
            self.attention_implementation,
            self.enable_prefix_caching,
            self.vllm_max_num_seqs,
            self.vllm_gpu_memory_utilization,
            self.use_torch_compile,
            self.compile_mode,
        )


@dataclass(frozen=True)
class PreparedRequest:
    prompt_text: str
    request_payload: object
    input_token_count: int


@dataclass(frozen=True)
class InferenceSession:
    config: InferenceConfig
    tokenizer: object
    backend_model: object

    def prepare_prompt(self, prompt_text):
        request_payload, input_token_count = prepare_request(prompt_text, self.tokenizer, self.config)
        return PreparedRequest(
            prompt_text=prompt_text,
            request_payload=request_payload,
            input_token_count=input_token_count,
        )

    def prepare_instruction(self, instruction):
        return self.prepare_prompt(build_prompt(instruction, self.tokenizer))

    def generate_prepared_token_count(
        self,
        prepared_request,
        *,
        max_new_tokens=120,
        do_sample=False,
        temperature=None,
        top_p=None,
        stop_on_json=True,
    ):
        generation_result = generate_from_request(
            prepared_request.request_payload,
            tokenizer=self.tokenizer,
            backend_model=self.backend_model,
            config=self.config,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            return_text=False,
            stop_on_json=stop_on_json,
        )
        return generation_result["output_token_count"]

    def generate_prepared_output(
        self,
        prepared_request,
        *,
        max_new_tokens=120,
        do_sample=False,
        temperature=None,
        top_p=None,
        stop_on_json=True,
    ):
        generation_result = generate_from_request(
            prepared_request.request_payload,
            tokenizer=self.tokenizer,
            backend_model=self.backend_model,
            config=self.config,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            return_text=True,
            stop_on_json=stop_on_json,
        )
        return {
            "prompt_text": prepared_request.prompt_text,
            "generated_text": generation_result["generated_text"],
            "input_token_count": prepared_request.input_token_count,
            "output_token_count": generation_result["output_token_count"],
            "inference_config": describe_inference_config(self.config),
        }

    def generate_raw_output(
        self,
        instruction,
        *,
        max_new_tokens=120,
        do_sample=False,
        temperature=None,
        top_p=None,
        stop_on_json=True,
    ):
        prepared_request = self.prepare_instruction(instruction)
        return self.generate_prepared_output(
            prepared_request,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            stop_on_json=stop_on_json,
        )

    def warmup(self, prepared_request, *, runs=1, max_new_tokens=120):
        if runs <= 0:
            return

        for _ in range(runs):
            self.generate_prepared_token_count(
                prepared_request,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        synchronize_device(self.config.resolved_device())


def load_prompt(path="etc/transform_prompt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def torch_dtype_to_precision(torch_dtype):
    if torch_dtype in (None, "auto"):
        return "auto"

    mapping = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }
    return mapping.get(torch_dtype, "auto")


def resolve_torch_dtype(precision, device):
    normalized = precision.lower()
    if normalized == "auto":
        if device.startswith("cuda"):
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported precision: {precision}")

    resolved_dtype = mapping[normalized]
    if device == "cpu" and resolved_dtype == torch.float16:
        return torch.float32
    return resolved_dtype


def synchronize_device(device):
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device=device)


def preferred_mixed_precision(device=DEFAULT_DEVICE):
    resolved_device = get_default_device() if device == "auto" else device
    if not resolved_device.startswith("cuda") or not torch.cuda.is_available():
        return "float16"
    return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"


def build_inference_config(
    *,
    model_name=DEFAULT_MODEL_NAME,
    device=DEFAULT_DEVICE,
    torch_dtype=torch.float16,
    inference_config=None,
):
    if inference_config is not None:
        return inference_config

    return InferenceConfig(
        model_name=model_name,
        device=device,
        precision=torch_dtype_to_precision(torch_dtype),
    )


def describe_inference_config(config):
    parts = [
        f"backend={config.backend}",
        f"device={config.resolved_device()}",
        f"precision={config.precision}",
    ]
    if config.quantization != "none":
        parts.append(f"quantization={config.quantization}")
    if config.attention_implementation != "default":
        parts.append(f"attention={config.attention_implementation}")
    if config.enable_prefix_caching:
        parts.append("prefix_caching=true")
    if config.backend == "vllm":
        parts.append(f"max_num_seqs={config.vllm_max_num_seqs}")
        parts.append(f"gpu_memory_utilization={config.vllm_gpu_memory_utilization}")
    if config.use_torch_compile:
        parts.append(f"torch_compile={config.compile_mode}")
    return ", ".join(parts)


def get_inference_support_issue(config):
    if config.backend not in SUPPORTED_BACKENDS:
        return f"Unsupported backend: {config.backend}"

    if config.quantization not in SUPPORTED_QUANTIZATION:
        return f"Unsupported quantization mode: {config.quantization}"

    if config.attention_implementation not in SUPPORTED_ATTENTION_IMPLEMENTATIONS:
        return f"Unsupported attention implementation: {config.attention_implementation}"

    if config.vllm_max_num_seqs < 1:
        return "vLLM max_num_seqs must be at least 1."
    if not 0 < config.vllm_gpu_memory_utilization <= 1:
        return "vLLM gpu_memory_utilization must be between 0 and 1."

    resolved_device = config.resolved_device()
    resolved_dtype = resolve_torch_dtype(config.precision, resolved_device)

    if config.backend == "vllm":
        if not importlib.util.find_spec("vllm"):
            return "vLLM is not installed."
        if not resolved_device.startswith("cuda") or not torch.cuda.is_available():
            return "vLLM benchmarking requires a CUDA device."
        if config.use_torch_compile:
            return "torch.compile only applies to the Transformers backend."
        if config.quantization != "none":
            return "This starter harness wires quantization through Transformers only; benchmark vLLM independently."
        if config.attention_implementation != "default":
            return "Attention implementation selection applies to the Transformers backend."
        return None

    if config.enable_prefix_caching:
        return "Prefix caching applies to the vLLM backend."

    if config.attention_implementation == "flash_attention_2":
        if not resolved_device.startswith("cuda") or not torch.cuda.is_available():
            return "FlashAttention 2 requires a CUDA device."
        if resolved_dtype not in (torch.float16, torch.bfloat16):
            return "FlashAttention 2 requires float16 or bfloat16 precision."
        if not importlib.util.find_spec("flash_attn"):
            return "flash-attn is not installed."

    if config.quantization != "none":
        if not resolved_device.startswith("cuda") or not torch.cuda.is_available():
            return "bitsandbytes quantization requires a CUDA device."
        if not importlib.util.find_spec("bitsandbytes"):
            return "bitsandbytes is not installed."
        if not importlib.util.find_spec("accelerate"):
            return "accelerate is required for bitsandbytes quantization."

    if config.use_torch_compile and not hasattr(torch, "compile"):
        return "torch.compile is not available in this PyTorch build."

    return None


@lru_cache(maxsize=8)
def load_tokenizer(model_name=DEFAULT_MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name)


def _load_transformers_model(config):
    resolved_device = config.resolved_device()
    resolved_dtype = resolve_torch_dtype(config.precision, resolved_device)
    load_kwargs = {}

    if config.quantization == "none":
        load_kwargs["torch_dtype"] = resolved_dtype
    else:
        from transformers import BitsAndBytesConfig

        quantization_kwargs = {
            "load_in_8bit": config.quantization == "bitsandbytes-8bit",
            "load_in_4bit": config.quantization == "bitsandbytes-4bit",
        }
        load_kwargs["quantization_config"] = BitsAndBytesConfig(**quantization_kwargs)
        load_kwargs["device_map"] = "auto"
        load_kwargs["torch_dtype"] = resolved_dtype

    if config.attention_implementation != "default":
        load_kwargs["attn_implementation"] = config.attention_implementation

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **load_kwargs)
    if config.quantization == "none":
        model = model.to(resolved_device)

    model.eval()

    if config.use_torch_compile:
        model = torch.compile(model, mode=config.compile_mode, fullgraph=False)

    return model


def _load_vllm_engine(config):
    from vllm import LLM

    resolved_device = config.resolved_device()
    resolved_dtype = resolve_torch_dtype(config.precision, resolved_device)
    dtype_name = torch_dtype_to_precision(resolved_dtype)
    if dtype_name == "float32":
        dtype_name = "float16"

    return LLM(
        model=config.model_name,
        dtype=dtype_name,
        enable_prefix_caching=config.enable_prefix_caching,
        max_num_seqs=config.vllm_max_num_seqs,
        gpu_memory_utilization=config.vllm_gpu_memory_utilization,
    )


def load_backend(inference_config):
    global _BACKEND_MODEL, _BACKEND_CACHE_KEY

    config = build_inference_config(inference_config=inference_config)
    support_issue = get_inference_support_issue(config)
    if support_issue is not None:
        raise RuntimeError(support_issue)

    tokenizer = load_tokenizer(config.model_name)
    cache_key = config.cache_key()
    if _BACKEND_MODEL is not None and _BACKEND_CACHE_KEY == cache_key:
        return tokenizer, _BACKEND_MODEL

    if config.backend == "transformers":
        backend_model = _load_transformers_model(config)
    else:
        backend_model = _load_vllm_engine(config)

    _BACKEND_MODEL = backend_model
    _BACKEND_CACHE_KEY = cache_key
    return tokenizer, backend_model


def clear_backend_cache():
    global _BACKEND_MODEL, _BACKEND_CACHE_KEY

    _BACKEND_MODEL = None
    _BACKEND_CACHE_KEY = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_inference_session(
    model_name=DEFAULT_MODEL_NAME,
    device=DEFAULT_DEVICE,
    torch_dtype=torch.float16,
    inference_config=None,
):
    config = build_inference_config(
        model_name=model_name,
        device=device,
        torch_dtype=torch_dtype,
        inference_config=inference_config,
    )
    tokenizer, backend_model = load_backend(config)
    return InferenceSession(
        config=config,
        tokenizer=tokenizer,
        backend_model=backend_model,
    )


def load_model(
    model_name=DEFAULT_MODEL_NAME,
    device=DEFAULT_DEVICE,
    torch_dtype=torch.float16,
    inference_config=None,
):
    config = build_inference_config(
        model_name=model_name,
        device=device,
        torch_dtype=torch_dtype,
        inference_config=inference_config,
    )
    if config.backend != "transformers":
        raise RuntimeError("load_model only supports the Transformers backend. Use load_backend for other engines.")
    return load_backend(config)


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


def prepare_request(prompt_text, tokenizer, config):
    if config.backend == "transformers":
        inputs = tokenizer(prompt_text, return_tensors="pt").to(config.resolved_device())
        return inputs, int(inputs["input_ids"].shape[1])

    input_token_count = len(tokenizer.encode(prompt_text, add_special_tokens=False))
    return prompt_text, input_token_count


def has_complete_top_level_json_object(text):
    start = text.find("{")
    if start == -1:
        return False

    depth = 0
    in_string = False
    escape = False
    for character in text[start:]:
        if in_string:
            if escape:
                escape = False
            elif character == "\\":
                escape = True
            elif character == "\"":
                in_string = False
            continue

        if character == "\"":
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return True

    return False


class JsonObjectStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.last_token_count = -1
        self.last_decoded_text = ""

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[0] != 1:
            return torch.zeros(input_ids.shape[0], device=input_ids.device, dtype=torch.bool)

        generated_ids = input_ids[0, self.prompt_length :]
        token_count = int(generated_ids.shape[0])
        if token_count == self.last_token_count:
            generated_text = self.last_decoded_text
        else:
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            self.last_token_count = token_count
            self.last_decoded_text = generated_text

        should_stop = has_complete_top_level_json_object(generated_text)
        return torch.full((input_ids.shape[0],), should_stop, device=input_ids.device, dtype=torch.bool)


def json_stopping_criteria(tokenizer, request_payload):
    if "input_ids" not in request_payload:
        return None
    prompt_length = int(request_payload["input_ids"].shape[1])
    return StoppingCriteriaList([JsonObjectStoppingCriteria(tokenizer, prompt_length)])


def generate_from_request(
    request_payload,
    *,
    tokenizer,
    backend_model,
    config,
    max_new_tokens=120,
    do_sample=False,
    temperature=None,
    top_p=None,
    return_text=True,
    stop_on_json=True,
):
    if config.backend == "transformers":
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if stop_on_json:
            generation_kwargs["stopping_criteria"] = json_stopping_criteria(tokenizer, request_payload)
        if do_sample and temperature is not None:
            generation_kwargs["temperature"] = temperature
        if do_sample and top_p is not None:
            generation_kwargs["top_p"] = top_p

        with torch.inference_mode():
            outputs = backend_model.generate(**request_payload, **generation_kwargs)

        output_token_count = int(outputs.shape[1] - request_payload["input_ids"].shape[1])
        if not return_text:
            return {"output_token_count": output_token_count}

        generated_ids = outputs[0][request_payload["input_ids"].shape[1] :]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return {
            "generated_text": generated_text,
            "output_token_count": output_token_count,
        }

    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature if do_sample and temperature is not None else 0.0,
        top_p=top_p if do_sample and top_p is not None else 1.0,
    )
    request_output = backend_model.generate([request_payload], sampling_params, use_tqdm=False)[0]
    generated = request_output.outputs[0]
    output_token_count = len(generated.token_ids)
    if not return_text:
        return {"output_token_count": output_token_count}

    return {
        "generated_text": generated.text.strip(),
        "output_token_count": output_token_count,
    }


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
    inference_config=None,
    inference_session=None,
    max_new_tokens=120,
    do_sample=False,
    temperature=None,
    top_p=None,
    stop_on_json=True,
):
    session = inference_session
    if session is None:
        session = load_inference_session(
            model_name=model_name,
            device=device,
            torch_dtype=torch_dtype,
            inference_config=inference_config,
        )
    return session.generate_raw_output(
        instruction,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        stop_on_json=stop_on_json,
    )


def finalize_plan_result(
    raw_result,
    *,
    model_name=DEFAULT_MODEL_NAME,
    device=DEFAULT_DEVICE,
    torch_dtype=torch.float16,
    inference_config=None,
    inference_session=None,
    max_new_tokens=120,
    repair_attempts=1,
):
    result = dict(raw_result)

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
            inference_config=inference_config,
            inference_session=inference_session,
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


def generate_plan(
    instruction,
    *,
    model_name=DEFAULT_MODEL_NAME,
    device=DEFAULT_DEVICE,
    torch_dtype=torch.float16,
    inference_config=None,
    inference_session=None,
    max_new_tokens=120,
    do_sample=False,
    temperature=None,
    top_p=None,
    repair_attempts=1,
):
    raw_result = generate_raw_output(
        instruction,
        model_name=model_name,
        device=device,
        torch_dtype=torch_dtype,
        inference_config=inference_config,
        inference_session=inference_session,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    return finalize_plan_result(
        raw_result,
        model_name=model_name,
        device=device,
        torch_dtype=torch_dtype,
        inference_config=inference_config,
        inference_session=inference_session,
        max_new_tokens=max_new_tokens,
        repair_attempts=repair_attempts,
    )
