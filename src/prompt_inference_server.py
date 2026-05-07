import argparse
import json
import sys
import time
from pathlib import Path

import torch

from language_to_action import (
    DEFAULT_MODEL_NAME,
    InferenceConfig,
    describe_inference_config,
    finalize_plan_result,
    load_inference_session,
    load_prompt,
    preferred_mixed_precision,
    synchronize_device,
)


EXIT_COMMANDS = {"exit", "quit", "q"}
SCENARIOS = (
    "baseline",
    "mixed_precision",
    "mixed_precision_compile",
    "mixed_precision_flash_attention",
    "mixed_precision_sdpa",
    "quantization",
    "torch_compile",
    "all_compatible",
    "optimized",
    "vllm",
    "vllm_prefix_caching",
)


def resolve_mixed_precision(device, precision):
    if precision == "auto":
        return preferred_mixed_precision(device)
    return precision


def build_scenario_config(model_name, device, scenario, precision, compile_mode):
    resolved_precision = resolve_mixed_precision(device, precision)
    vllm_precision = "float16" if precision == "auto" else resolved_precision

    if scenario == "baseline":
        return InferenceConfig(
            model_name=model_name,
            device=device,
            backend="transformers",
            precision="float32",
        )

    if scenario == "mixed_precision":
        return InferenceConfig(
            model_name=model_name,
            device=device,
            backend="transformers",
            precision=resolved_precision,
        )

    if scenario == "quantization":
        return InferenceConfig(
            model_name=model_name,
            device=device,
            backend="transformers",
            precision=resolved_precision,
            quantization="bitsandbytes-8bit",
        )

    if scenario == "mixed_precision_compile":
        return InferenceConfig(
            model_name=model_name,
            device=device,
            backend="transformers",
            precision=resolved_precision,
            use_torch_compile=True,
            compile_mode=compile_mode,
        )

    if scenario == "mixed_precision_flash_attention":
        return InferenceConfig(
            model_name=model_name,
            device=device,
            backend="transformers",
            precision=resolved_precision,
            attention_implementation="flash_attention_2",
        )

    if scenario == "mixed_precision_sdpa":
        return InferenceConfig(
            model_name=model_name,
            device=device,
            backend="transformers",
            precision=resolved_precision,
            attention_implementation="sdpa",
        )

    if scenario == "torch_compile":
        return InferenceConfig(
            model_name=model_name,
            device=device,
            backend="transformers",
            precision="float32",
            use_torch_compile=True,
            compile_mode=compile_mode,
        )

    if scenario == "all_compatible":
        return InferenceConfig(
            model_name=model_name,
            device=device,
            backend="transformers",
            precision=resolved_precision,
            quantization="bitsandbytes-8bit",
            use_torch_compile=True,
            compile_mode=compile_mode,
        )

    if scenario == "optimized":
        return InferenceConfig(
            model_name=model_name,
            device=device,
            backend="vllm",
            precision=vllm_precision,
            enable_prefix_caching=True,
        )

    if scenario == "vllm":
        return InferenceConfig(
            model_name=model_name,
            device=device,
            backend="vllm",
            precision=vllm_precision,
        )

    if scenario == "vllm_prefix_caching":
        return InferenceConfig(
            model_name=model_name,
            device=device,
            backend="vllm",
            precision=vllm_precision,
            enable_prefix_caching=True,
        )

    raise ValueError(f"Unsupported scenario: {scenario}")


def read_instruction_file(raw_path):
    prompt_path = Path(raw_path).expanduser()
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file does not exist: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def log_status(message):
    print(message, file=sys.stderr, flush=True)


def emit_json(payload, *, pretty):
    if pretty:
        print(json.dumps(payload, indent=2), flush=True)
    else:
        print(json.dumps(payload, separators=(",", ":")), flush=True)


def require_requested_device(config):
    device = config.resolved_device()
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Device {device} was requested, but CUDA is not available to PyTorch. "
            "Check the NVIDIA driver and PyTorch CUDA wheel before starting the prompt server."
        )


def run_warmups(session, warmup_prompt_path, warmup_runs, max_new_tokens):
    if warmup_runs <= 0:
        return

    warmup_instruction = load_prompt(warmup_prompt_path)
    prepared_request = session.prepare_instruction(warmup_instruction)
    session.warmup(
        prepared_request,
        runs=warmup_runs,
        max_new_tokens=max_new_tokens,
    )


def generate_plan_from_file(session, prompt_path, max_new_tokens, repair_attempts):
    instruction = read_instruction_file(prompt_path)
    prepared_request = session.prepare_instruction(instruction)
    device = session.config.resolved_device()

    synchronize_device(device)
    start_time = time.perf_counter()
    raw_result = session.generate_prepared_output(
        prepared_request,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    result = finalize_plan_result(
        raw_result,
        inference_session=session,
        max_new_tokens=max_new_tokens,
        repair_attempts=repair_attempts,
    )
    synchronize_device(device)

    return result, (time.perf_counter() - start_time) * 1000.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load one inference scenario once, run warmups, then accept prompt "
            "file paths on stdin and print one JSON plan response per request."
        )
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        default="mixed_precision",
        help="Optimization scenario to keep loaded. Defaults to the observed fastest variant.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for inference. Defaults to cuda so GPU failures are explicit.",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help=(
            "Precision for mixed_precision, quantization, and all_compatible. "
            "auto chooses bf16 when supported by CUDA, otherwise fp16."
        ),
    )
    parser.add_argument(
        "--compile-mode",
        default="reduce-overhead",
        help="torch.compile mode for torch_compile and all_compatible scenarios.",
    )
    parser.add_argument(
        "--warmup-prompt",
        default="etc/transform_prompt",
        help="Prompt file used for warmup requests before accepting user prompt paths.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
        help="Number of untimed warmup requests to run before accepting prompt paths.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="Maximum number of tokens to generate per prompt.",
    )
    parser.add_argument(
        "--repair-attempts",
        type=int,
        default=0,
        help="Maximum number of repair attempts if the raw model output is not valid JSON.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON responses instead of emitting one compact JSON object per line.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs cannot be negative.")
    if args.repair_attempts < 0:
        raise ValueError("--repair-attempts cannot be negative.")

    config = build_scenario_config(
        args.model_name,
        args.device,
        args.scenario,
        args.precision,
        args.compile_mode,
    )
    require_requested_device(config)

    log_status(f"loading scenario={args.scenario}: {describe_inference_config(config)}")
    setup_start_time = time.perf_counter()
    session = load_inference_session(inference_config=config)
    synchronize_device(session.config.resolved_device())
    log_status(f"model ready after {(time.perf_counter() - setup_start_time) * 1000.0:.2f} ms")

    log_status(f"running {args.warmup_runs} warmup request(s)")
    warmup_start_time = time.perf_counter()
    run_warmups(session, args.warmup_prompt, args.warmup_runs, args.max_new_tokens)
    synchronize_device(session.config.resolved_device())
    log_status(f"warmups complete after {(time.perf_counter() - warmup_start_time) * 1000.0:.2f} ms")
    log_status("ready; enter a prompt file path, or 'quit' to exit")

    for raw_line in sys.stdin:
        prompt_path = raw_line.strip().strip("\"'")
        if not prompt_path:
            continue
        if prompt_path.lower() in EXIT_COMMANDS:
            break

        try:
            result, latency_ms = generate_plan_from_file(
                session,
                prompt_path,
                max_new_tokens=args.max_new_tokens,
                repair_attempts=args.repair_attempts,
            )
        except Exception as exc:
            log_status(json.dumps({"status": "error", "path": prompt_path, "error": str(exc)}))
            continue

        emit_json(result["plan"], pretty=args.pretty)
        log_status(
            "completed "
            f"path={prompt_path} "
            f"latency_ms={latency_ms:.2f} "
            f"input_tokens={result['input_token_count']} "
            f"output_tokens={result['output_token_count']}"
        )

    log_status("stopped")


if __name__ == "__main__":
    main()
