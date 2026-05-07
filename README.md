# tiny-inference

## Recommended environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install \
  "transformers>=4.49,<5" \
  "accelerate>=1.4,<2" \
  "bitsandbytes>=0.45.1,<0.46" \
  "nvidia-ml-py>=12,<13" \
  "wandb>=0.19,<1"
```

Optional FlashAttention benchmark support:

```bash
python -m pip install flash-attn --no-build-isolation
```

Optional vLLM benchmark support:

```bash
python -m pip install "vllm>=0.7,<1"
```

Dependency roles:

- `torch==2.6.0+cu124`: CUDA 12.4 PyTorch runtime for GPU inference.
- `transformers>=4.49,<5`: loads Qwen/Qwen2.5 models through Hugging Face.
- `accelerate>=1.4,<2`: required by Transformers for quantized/device-mapped loading.
- `bitsandbytes>=0.45.1,<0.46`: required for the quantized optimized scenario.
- `flash-attn`: optional dependency for the `mixed_precision_flash_attention` benchmark scenario.
- `vllm>=0.7,<1`: optional dependency for the `optimized`, `vllm`, and `vllm_prefix_caching` benchmark scenarios.
- `nvidia-ml-py>=12,<13`: provides the `pynvml` module used for GPU telemetry.
- `wandb>=0.19,<1`: logs benchmark rows and tables to Weights & Biases.


## Weights & Biases latency dashboards

Install and authenticate W&B in the environment where you run inference:

```bash
wandb login
```

Benchmark runs can stream latency metrics and a final table to a W&B project:

```bash
python src/benchmark.py \
  --device cuda \
  --scenario baseline \
  --base-only \
  --wandb-project tiny-inference-latency \
  --wandb-run-name baseline-smoke
```

To compare baseline and the optimized Transformers paths with warmups and repeated measured runs:

```bash
python src/benchmark.py \
  --device cuda \
  --scenario baseline \
  --scenario mixed_precision \
  --scenario mixed_precision_compile \
  --scenario mixed_precision_flash_attention \
  --scenario mixed_precision_sdpa \
  --base-only \
  --warmup-runs 3 \
  --benchmark-runs 10 \
  --max-new-tokens 80 \
  --wandb-project tiny-inference-latency \
  --wandb-run-name baseline-vs-optimized
```

For full latency sweeps across selected scenarios:

```bash
python src/benchmark.py \
  --device cuda \
  --scenario baseline \
  --scenario mixed_precision \
  --wandb-project tiny-inference-latency \
  --wandb-tags gpu,latency
```

To compare plain mixed precision against FlashAttention:

```bash
python src/benchmark.py \
  --device cuda \
  --scenario mixed_precision \
  --scenario mixed_precision_flash_attention \
  --base-only \
  --warmup-runs 3 \
  --benchmark-runs 10 \
  --max-new-tokens 80
```

If `flash-attn` cannot be installed, compare against PyTorch SDPA instead:

```bash
python src/benchmark.py \
  --device cuda \
  --scenario mixed_precision \
  --scenario mixed_precision_sdpa \
  --base-only \
  --warmup-runs 3 \
  --benchmark-runs 10 \
  --max-new-tokens 80
```

To compare vLLM with and without automatic prefix caching:

```bash
python src/benchmark.py \
  --device cuda \
  --scenario vllm \
  --scenario vllm_prefix_caching \
  --base-only \
  --warmup-runs 3 \
  --benchmark-runs 10 \
  --max-new-tokens 80
```

The vLLM benchmark scenarios default to `float16`, which works on Tesla T4 and other pre-Ampere GPUs where `bfloat16` is unsupported.
They also default to `max_num_seqs=1` and `gpu_memory_utilization=0.8` because this project benchmarks one request at a time, and vLLM's larger default concurrency warmup can exceed memory on smaller GPUs.

The final optimized scenario is `optimized`, which uses vLLM, `float16`, automatic prefix caching, `max_num_seqs=1`, and `gpu_memory_utilization=0.8`:

```bash
python src/benchmark.py \
  --device cuda \
  --scenario mixed_precision \
  --scenario optimized \
  --base-only \
  --warmup-runs 3 \
  --benchmark-runs 10 \
  --max-new-tokens 80
```

The benchmark logs scalar history rows for charts and a `latency_reports` table with scenario, backend, precision, prompt/output length, latency percentiles, tokens/sec, VRAM, GPU utilization, and skipped/failed status rows.

Single inference telemetry can also be logged:

```bash
python src/output_inference.py \
  --profile optimized \
  --repeat 5 \
  --wandb-project tiny-inference-latency
```

Use `--wandb-mode offline` to save runs locally and sync later with `wandb sync`.

## Persistent prompt runner

For ROS-style integration, keep the model loaded in one process and send prompt file paths over stdin. The runner loads one selected Transformers scenario, runs warmups, then prints one JSON plan response to stdout for each prompt path.

Status messages, latency, and errors are written to stderr so stdout stays machine-readable.

```bash
python src/prompt_inference_server.py \
  --scenario optimized \
  --device cuda \
  --warmup-runs 3 \
  --max-new-tokens 80 \
  --repair-attempts 0
```

Available scenarios are `baseline`, `mixed_precision`, `mixed_precision_compile`, `mixed_precision_flash_attention`, `mixed_precision_sdpa`, `quantization`, `torch_compile`, `all_compatible`, `optimized`, `vllm`, and `vllm_prefix_caching`. The default is `mixed_precision`.

After it prints `ready` on stderr, enter prompt file paths:

```text
etc/transform_prompt
quit
```

For pipe-based integration:

```bash
printf '%s\n' etc/transform_prompt quit | python src/prompt_inference_server.py --scenario optimized --device cuda
```
