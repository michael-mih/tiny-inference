import argparse
import statistics
import time
from dataclasses import dataclass

import torch

from language_to_action import (
    DEFAULT_DEVICE,
    DEFAULT_MODEL_NAME,
    InferenceConfig,
    build_prompt,
    describe_inference_config,
    get_inference_support_issue,
    load_inference_session,
    load_prompt,
    preferred_mixed_precision,
    synchronize_device,
)

WARMUP_RUNS = 3
BENCHMARK_RUNS = 10
PROMPT_LENGTHS = [32, 64, 128, 256, 512]
OUTPUT_LENGTHS = [8, 16, 32, 64, 128]


@dataclass(frozen=True)
class BenchmarkScenario:
    name: str
    description: str
    config: InferenceConfig


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[index]


def sample_gpu_utilization(device):
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return None

    try:
        utilization = torch.cuda.utilization(device=device)
    except (AttributeError, RuntimeError):
        return None

    if isinstance(utilization, (int, float)):
        return float(utilization)
    if isinstance(utilization, torch.Tensor):
        return float(utilization.item())
    return None


def build_instruction(target_tokens, base_instruction, tokenizer):
    words = base_instruction.split()
    if not words:
        words = ["plan"]

    repeated_words = []
    index = 0
    while True:
        repeated_words.append(words[index % len(words)])
        candidate = " ".join(repeated_words)
        token_count = len(tokenizer.encode(candidate, add_special_tokens=False))
        if token_count >= target_tokens:
            return candidate
        index += 1


def benchmark_request(prompt_text, session, max_new_tokens, warmup_runs, benchmark_runs):
    prepared_request = session.prepare_prompt(prompt_text)
    device = session.config.resolved_device()

    session.warmup(
        prepared_request,
        runs=warmup_runs,
        max_new_tokens=max_new_tokens,
    )

    latencies_ms = []
    output_token_counts = []
    peak_memory_mb = []
    gpu_util_samples = []

    for _ in range(benchmark_runs):
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device=device)

        utilization_before = sample_gpu_utilization(device)
        synchronize_device(device)
        start_time = time.perf_counter()
        generated_tokens = session.generate_prepared_token_count(
            prepared_request,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        synchronize_device(device)
        end_time = time.perf_counter()
        utilization_after = sample_gpu_utilization(device)

        latency_ms = (end_time - start_time) * 1000.0

        latencies_ms.append(latency_ms)
        output_token_counts.append(generated_tokens)

        if torch.cuda.is_available() and str(device).startswith("cuda"):
            peak_memory_mb.append(torch.cuda.max_memory_allocated(device=device) / (1024 ** 2))

        run_util_samples = [value for value in (utilization_before, utilization_after) if value is not None]
        if run_util_samples:
            gpu_util_samples.append(sum(run_util_samples) / len(run_util_samples))

    avg_latency_ms = statistics.mean(latencies_ms)
    avg_output_tokens = statistics.mean(output_token_counts)

    return {
        "input_tokens": prepared_request.input_token_count,
        "avg_latency_ms": avg_latency_ms,
        "p50_latency_ms": percentile(latencies_ms, 50),
        "p95_latency_ms": percentile(latencies_ms, 95),
        "avg_output_tokens": avg_output_tokens,
        "avg_tokens_per_sec": avg_output_tokens / (avg_latency_ms / 1000),
        "peak_vram_mb": max(peak_memory_mb) if peak_memory_mb else None,
        "avg_gpu_utilization": statistics.mean(gpu_util_samples) if gpu_util_samples else None,
    }


def print_metrics(title, metrics):
    print(title)
    print(f"  input_tokens: {metrics['input_tokens']}")
    print(f"  avg_latency_ms: {metrics['avg_latency_ms']:.2f}")
    print(f"  p50_latency_ms: {metrics['p50_latency_ms']:.2f}")
    print(f"  p95_latency_ms: {metrics['p95_latency_ms']:.2f}")
    print(f"  avg_output_tokens: {metrics['avg_output_tokens']:.2f}")
    print(f"  avg_tokens_per_sec: {metrics['avg_tokens_per_sec']:.2f}")
    if metrics["peak_vram_mb"] is None:
        print("  peak_vram_mb: unavailable")
    else:
        print(f"  peak_vram_mb: {metrics['peak_vram_mb']:.2f}")
    if metrics["avg_gpu_utilization"] is None:
        print("  avg_gpu_utilization: unavailable")
    else:
        print(f"  avg_gpu_utilization: {metrics['avg_gpu_utilization']:.2f}%")
    print()


def build_scenarios(model_name, device):
    mixed_precision = preferred_mixed_precision(device)
    return [
        BenchmarkScenario(
            name="baseline",
            description="Transformers baseline in full precision.",
            config=InferenceConfig(
                model_name=model_name,
                device=device,
                backend="transformers",
                precision="float32",
            ),
        ),
        BenchmarkScenario(
            name="mixed_precision",
            description="Transformers with reduced precision weights for inference.",
            config=InferenceConfig(
                model_name=model_name,
                device=device,
                backend="transformers",
                precision=mixed_precision,
            ),
        ),
        BenchmarkScenario(
            name="quantization",
            description="Transformers with bitsandbytes 8-bit quantization.",
            config=InferenceConfig(
                model_name=model_name,
                device=device,
                backend="transformers",
                precision=mixed_precision,
                quantization="bitsandbytes-8bit",
            ),
        ),
        BenchmarkScenario(
            name="torch_compile",
            description="Transformers with torch.compile enabled.",
            config=InferenceConfig(
                model_name=model_name,
                device=device,
                backend="transformers",
                precision="float32",
                use_torch_compile=True,
            ),
        ),
        BenchmarkScenario(
            name="all_compatible",
            description="All compatible Transformers-side optimizations together.",
            config=InferenceConfig(
                model_name=model_name,
                device=device,
                backend="transformers",
                precision=mixed_precision,
                quantization="bitsandbytes-8bit",
                use_torch_compile=True,
            ),
        ),
        BenchmarkScenario(
            name="vllm",
            description="vLLM as an alternate inference backend.",
            config=InferenceConfig(
                model_name=model_name,
                device=device,
                backend="vllm",
                precision=mixed_precision,
            ),
        ),
    ]


def resolve_requested_scenarios(all_scenarios, requested_names):
    scenarios_by_name = {scenario.name: scenario for scenario in all_scenarios}
    if not requested_names or requested_names == ["all"]:
        return all_scenarios

    resolved = []
    for raw_name in requested_names:
        for name in [item.strip() for item in raw_name.split(",") if item.strip()]:
            if name == "all":
                return all_scenarios
            if name not in scenarios_by_name:
                valid_names = ", ".join(scenarios_by_name)
                raise ValueError(f"Unknown scenario '{name}'. Valid scenarios: {valid_names}, all")
            resolved.append(scenarios_by_name[name])
    return resolved


def list_scenarios(scenarios):
    print("Available scenarios")
    for scenario in scenarios:
        print(f"  {scenario.name}: {scenario.description}")
        print(f"    {describe_inference_config(scenario.config)}")


def get_benchmark_skip_reason(scenario):
    support_issue = get_inference_support_issue(scenario.config)
    if support_issue is not None:
        return support_issue

    if scenario.name == "mixed_precision":
        device = scenario.config.resolved_device()
        if not device.startswith("cuda") or not torch.cuda.is_available():
            return "Mixed-precision benchmarking requires a CUDA device in this harness."

    return None


def run_scenario(scenario, instruction, args):
    print(f"Scenario: {scenario.name}")
    print(f"  description: {scenario.description}")
    print(f"  config: {describe_inference_config(scenario.config)}")

    support_issue = get_benchmark_skip_reason(scenario)
    if support_issue is not None:
        print(f"  status: skipped ({support_issue})")
        print()
        return

    try:
        setup_start_time = time.perf_counter()
        session = load_inference_session(inference_config=scenario.config)
        synchronize_device(session.config.resolved_device())
        setup_latency_ms = (time.perf_counter() - setup_start_time) * 1000.0

        print(f"  setup_latency_ms: {setup_latency_ms:.2f}")

        full_prompt = build_prompt(instruction, session.tokenizer)

        base_metrics = benchmark_request(
            full_prompt,
            session,
            max_new_tokens=args.max_new_tokens,
            warmup_runs=args.warmup_runs,
            benchmark_runs=args.benchmark_runs,
        )
        print_metrics("Base benchmark", base_metrics)

        if args.base_only:
            return

        print("Prompt-length scaling")
        for target_length in PROMPT_LENGTHS:
            scaled_instruction = build_instruction(target_length, instruction, session.tokenizer)
            prompt_text = build_prompt(scaled_instruction, session.tokenizer)
            metrics = benchmark_request(
                prompt_text,
                session,
                max_new_tokens=args.max_new_tokens,
                warmup_runs=args.warmup_runs,
                benchmark_runs=args.benchmark_runs,
            )
            print_metrics(f"Prompt length target {target_length} tokens", metrics)

        print("Output-length scaling")
        for output_length in OUTPUT_LENGTHS:
            metrics = benchmark_request(
                full_prompt,
                session,
                max_new_tokens=output_length,
                warmup_runs=args.warmup_runs,
                benchmark_runs=args.benchmark_runs,
            )
            print_metrics(f"Output length target {output_length} tokens", metrics)
    except Exception as exc:
        print(f"  status: failed ({exc})")
        print()


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark inference optimizations across backends and techniques.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--scenario", action="append", dest="scenarios", default=[])
    parser.add_argument("--list-scenarios", action="store_true")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=WARMUP_RUNS)
    parser.add_argument("--benchmark-runs", type=int, default=BENCHMARK_RUNS)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    return parser.parse_args()


def main():
    args = parse_args()
    scenarios = build_scenarios(args.model_name, args.device)

    if args.list_scenarios:
        list_scenarios(scenarios)
        return

    selected_scenarios = resolve_requested_scenarios(scenarios, args.scenarios)
    instruction = load_prompt()

    for scenario in selected_scenarios:
        run_scenario(scenario, instruction, args)


if __name__ == "__main__":
    main()
