import statistics

import torch

from language_to_action import (
    DEFAULT_DEVICE,
    DEFAULT_MODEL_NAME,
    build_prompt,
    load_model,
    load_prompt,
)

WARMUP_RUNS = 3
BENCHMARK_RUNS = 10
PROMPT_LENGTHS = [32, 64, 128, 256, 512]
OUTPUT_LENGTHS = [8, 16, 32, 64, 128]


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[index]


def sample_gpu_utilization(device):
    if not torch.cuda.is_available():
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


def benchmark_request(prompt_text, tokenizer, model, device, max_new_tokens):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    for _ in range(WARMUP_RUNS):
        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize()

    latencies_ms = []
    output_token_counts = []
    peak_memory_mb = []
    gpu_util_samples = []

    for _ in range(BENCHMARK_RUNS):
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        utilization_before = sample_gpu_utilization(device)
        start.record()
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        end.record()
        torch.cuda.synchronize()
        utilization_after = sample_gpu_utilization(device)

        latency_ms = start.elapsed_time(end)
        generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]

        latencies_ms.append(latency_ms)
        output_token_counts.append(generated_tokens)
        peak_memory_mb.append(torch.cuda.max_memory_allocated() / (1024 ** 2))

        run_util_samples = [value for value in (utilization_before, utilization_after) if value is not None]
        if run_util_samples:
            gpu_util_samples.append(sum(run_util_samples) / len(run_util_samples))

    avg_latency_ms = statistics.mean(latencies_ms)
    avg_output_tokens = statistics.mean(output_token_counts)

    return {
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "avg_latency_ms": avg_latency_ms,
        "p50_latency_ms": percentile(latencies_ms, 50),
        "p95_latency_ms": percentile(latencies_ms, 95),
        "avg_output_tokens": avg_output_tokens,
        "avg_tokens_per_sec": avg_output_tokens / (avg_latency_ms / 1000),
        "peak_vram_mb": max(peak_memory_mb),
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
    print(f"  peak_vram_mb: {metrics['peak_vram_mb']:.2f}")
    if metrics["avg_gpu_utilization"] is None:
        print("  avg_gpu_utilization: unavailable")
    else:
        print(f"  avg_gpu_utilization: {metrics['avg_gpu_utilization']:.2f}%")
    print()


def main():
    tokenizer, model = load_model(model_name=DEFAULT_MODEL_NAME, device=DEFAULT_DEVICE)
    instruction = load_prompt()
    full_prompt = build_prompt(instruction, tokenizer)

    base_metrics = benchmark_request(full_prompt, tokenizer, model, DEFAULT_DEVICE, max_new_tokens=80)
    print_metrics("Base benchmark", base_metrics)

    print("Prompt-length scaling")
    for target_length in PROMPT_LENGTHS:
        scaled_instruction = build_instruction(target_length, instruction, tokenizer)
        prompt_text = build_prompt(scaled_instruction, tokenizer)
        metrics = benchmark_request(prompt_text, tokenizer, model, DEFAULT_DEVICE, max_new_tokens=80)
        print_metrics(f"Prompt length target {target_length} tokens", metrics)

    print("Output-length scaling")
    for output_length in OUTPUT_LENGTHS:
        metrics = benchmark_request(full_prompt, tokenizer, model, DEFAULT_DEVICE, max_new_tokens=output_length)
        print_metrics(f"Output length target {output_length} tokens", metrics)


if __name__ == "__main__":
    main()
