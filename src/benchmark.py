import argparse
import importlib.util
import statistics
import subprocess
import time
from dataclasses import dataclass

import torch

from language_to_action import (
    DEFAULT_DEVICE,
    DEFAULT_MODEL_NAME,
    InferenceConfig,
    build_prompt,
    clear_backend_cache,
    describe_inference_config,
    get_inference_support_issue,
    load_inference_session,
    load_prompt,
    synchronize_device,
)
from wandb_latency import add_wandb_args, init_wandb_latency_logger

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


def is_cuda_device(device):
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return False
    return True


def cuda_device_index(device):
    if not is_cuda_device(device):
        return None

    torch_device = torch.device(device)
    if torch_device.index is not None:
        return torch_device.index
    return torch.cuda.current_device()


def read_nvidia_smi_metric(device, query_field):
    device_index = cuda_device_index(device)
    if device_index is None:
        return None

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(device_index),
                f"--query-gpu={query_field}",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    first_line = next((line.strip() for line in output.splitlines() if line.strip()), "")
    if not first_line:
        return None

    try:
        return float(first_line.split()[0])
    except ValueError:
        return None


def sample_gpu_utilization(device):
    if not is_cuda_device(device):
        return None

    utilization = None
    if importlib.util.find_spec("pynvml"):
        try:
            utilization = torch.cuda.utilization(device=device)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            utilization = None

    if isinstance(utilization, (int, float)):
        return float(utilization)
    if isinstance(utilization, torch.Tensor):
        return float(utilization.item())
    return read_nvidia_smi_metric(device, "utilization.gpu")


def reset_peak_vram(device):
    if not is_cuda_device(device):
        return

    try:
        torch.cuda.reset_peak_memory_stats(device=device)
    except RuntimeError:
        return


def read_peak_vram_mb(device):
    if not is_cuda_device(device):
        return None

    measurements = []
    try:
        measurements.append(torch.cuda.max_memory_allocated(device=device) / (1024 ** 2))
    except RuntimeError:
        pass

    nvidia_smi_memory_mb = read_nvidia_smi_metric(device, "memory.used")
    if nvidia_smi_memory_mb is not None:
        measurements.append(nvidia_smi_memory_mb)

    return max(measurements) if measurements else None


def describe_gpu_telemetry_issue(device):
    if not str(device).startswith("cuda"):
        return f"device is {device}; GPU telemetry is only collected for CUDA runs"
    if not torch.cuda.is_available():
        return "CUDA is not available to PyTorch; check the NVIDIA driver and PyTorch CUDA build"
    return "CUDA is available, but PyTorch/NVML/nvidia-smi did not return GPU telemetry samples"


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
        reset_peak_vram(device)

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

        peak_vram_mb = read_peak_vram_mb(device)
        if peak_vram_mb is not None:
            peak_memory_mb.append(peak_vram_mb)

        run_util_samples = [value for value in (utilization_before, utilization_after) if value is not None]
        if run_util_samples:
            gpu_util_samples.append(sum(run_util_samples) / len(run_util_samples))

    avg_latency_ms = statistics.mean(latencies_ms)
    avg_output_tokens = statistics.mean(output_token_counts)
    gpu_telemetry_issue = None
    if not peak_memory_mb or not gpu_util_samples:
        gpu_telemetry_issue = describe_gpu_telemetry_issue(device)

    return {
        "input_tokens": prepared_request.input_token_count,
        "avg_latency_ms": avg_latency_ms,
        "p50_latency_ms": percentile(latencies_ms, 50),
        "p95_latency_ms": percentile(latencies_ms, 95),
        "avg_output_tokens": avg_output_tokens,
        "avg_tokens_per_sec": avg_output_tokens / (avg_latency_ms / 1000),
        "peak_vram_mb": max(peak_memory_mb) if peak_memory_mb else None,
        "avg_gpu_utilization": statistics.mean(gpu_util_samples) if gpu_util_samples else None,
        "gpu_telemetry_issue": gpu_telemetry_issue,
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
    if metrics["gpu_telemetry_issue"] is not None:
        print(f"  gpu_telemetry_issue: {metrics['gpu_telemetry_issue']}")
    print()


def scenario_metadata(scenario):
    config = scenario.config
    return {
        "scenario": scenario.name,
        "scenario_description": scenario.description,
        "model_name": config.model_name,
        "backend": config.backend,
        "requested_device": config.device,
        "device": config.resolved_device(),
        "precision": config.precision,
        "enable_prefix_caching": config.enable_prefix_caching,
        "vllm_max_num_seqs": config.vllm_max_num_seqs,
        "vllm_gpu_memory_utilization": config.vllm_gpu_memory_utilization,
        "inference_config": describe_inference_config(config),
    }


def log_benchmark_record(wandb_logger, scenario, benchmark_name, phase, metrics=None, **extra_fields):
    if not wandb_logger.enabled:
        return

    status = extra_fields.pop("status", "completed")
    record = {
        **scenario_metadata(scenario),
        "benchmark_name": benchmark_name,
        "phase": phase,
        "status": status,
    }
    if metrics is not None:
        record.update(metrics)
    record.update(extra_fields)
    wandb_logger.log(record)


def build_scenarios(model_name, device):
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
            name="optimized",
            description="Final low-latency path: vLLM, float16, and automatic prefix caching.",
            config=InferenceConfig(
                model_name=model_name,
                device=device,
                backend="vllm",
                precision="float16",
                enable_prefix_caching=True,
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

    return None


def run_scenario(scenario, instruction, args, wandb_logger):
    print(f"Scenario: {scenario.name}")
    print(f"  description: {scenario.description}")
    print(f"  config: {describe_inference_config(scenario.config)}")

    support_issue = get_benchmark_skip_reason(scenario)
    if support_issue is not None:
        print(f"  status: skipped ({support_issue})")
        print()
        log_benchmark_record(
            wandb_logger,
            scenario,
            "scenario_status",
            "skip",
            status="skipped",
            status_reason=support_issue,
        )
        return

    session = None
    try:
        setup_start_time = time.perf_counter()
        session = load_inference_session(inference_config=scenario.config)
        synchronize_device(session.config.resolved_device())
        setup_latency_ms = (time.perf_counter() - setup_start_time) * 1000.0

        print(f"  setup_latency_ms: {setup_latency_ms:.2f}")
        log_benchmark_record(
            wandb_logger,
            scenario,
            "setup",
            "setup",
            setup_latency_ms=setup_latency_ms,
        )

        full_prompt = build_prompt(instruction, session.tokenizer)

        base_metrics = benchmark_request(
            full_prompt,
            session,
            max_new_tokens=args.max_new_tokens,
            warmup_runs=args.warmup_runs,
            benchmark_runs=args.benchmark_runs,
        )
        print_metrics("Base benchmark", base_metrics)
        log_benchmark_record(
            wandb_logger,
            scenario,
            "base",
            "base",
            base_metrics,
            warmup_runs=args.warmup_runs,
            benchmark_runs=args.benchmark_runs,
            max_new_tokens=args.max_new_tokens,
        )
        wandb_logger.update_summary(
            {
                f"{scenario.name}_base_avg_latency_ms": base_metrics["avg_latency_ms"],
                f"{scenario.name}_base_p95_latency_ms": base_metrics["p95_latency_ms"],
                f"{scenario.name}_base_avg_tokens_per_sec": base_metrics["avg_tokens_per_sec"],
            }
        )

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
            log_benchmark_record(
                wandb_logger,
                scenario,
                f"prompt_length_{target_length}",
                "prompt_length",
                metrics,
                warmup_runs=args.warmup_runs,
                benchmark_runs=args.benchmark_runs,
                prompt_target_tokens=target_length,
                max_new_tokens=args.max_new_tokens,
            )

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
            log_benchmark_record(
                wandb_logger,
                scenario,
                f"output_length_{output_length}",
                "output_length",
                metrics,
                warmup_runs=args.warmup_runs,
                benchmark_runs=args.benchmark_runs,
                max_new_tokens=output_length,
                output_target_tokens=output_length,
            )
    except Exception as exc:
        print(f"  status: failed ({exc})")
        print()
        log_benchmark_record(
            wandb_logger,
            scenario,
            "scenario_status",
            "failure",
            status="failed",
            status_reason=str(exc),
        )
    finally:
        session = None
        clear_backend_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark the baseline and optimized inference tests.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--scenario", action="append", dest="scenarios", default=[])
    parser.add_argument("--list-scenarios", action="store_true")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=WARMUP_RUNS)
    parser.add_argument("--benchmark-runs", type=int, default=BENCHMARK_RUNS)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    add_wandb_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    scenarios = build_scenarios(args.model_name, args.device)

    if args.list_scenarios:
        list_scenarios(scenarios)
        return

    selected_scenarios = resolve_requested_scenarios(scenarios, args.scenarios)
    instruction = load_prompt()

    wandb_logger = init_wandb_latency_logger(
        args,
        config={
            "model_name": args.model_name,
            "device": args.device,
            "selected_scenarios": [scenario.name for scenario in selected_scenarios],
            "base_only": args.base_only,
            "warmup_runs": args.warmup_runs,
            "benchmark_runs": args.benchmark_runs,
            "max_new_tokens": args.max_new_tokens,
            "prompt_lengths": [] if args.base_only else PROMPT_LENGTHS,
            "output_lengths": [] if args.base_only else OUTPUT_LENGTHS,
        },
        job_type="latency-benchmark",
        table_key="latency_reports",
    )

    try:
        for scenario in selected_scenarios:
            run_scenario(scenario, instruction, args, wandb_logger)
    finally:
        wandb_logger.finish()


if __name__ == "__main__":
    main()
