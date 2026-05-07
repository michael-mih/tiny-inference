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

Dependency roles:

- `torch==2.6.0+cu124`: CUDA 12.4 PyTorch runtime for GPU inference.
- `transformers>=4.49,<5`: loads Qwen/Qwen2.5 models through Hugging Face.
- `accelerate>=1.4,<2`: required by Transformers for quantized/device-mapped loading.
- `bitsandbytes>=0.45.1,<0.46`: required for the quantized optimized scenario.
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

To compare baseline and the optimized Transformers path with warmups and repeated measured runs:

```bash
python src/benchmark.py \
  --device cuda \
  --scenario baseline \
  --scenario all_compatible \
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
  --scenario mixed_precision \
  --device cuda \
  --warmup-runs 3 \
  --max-new-tokens 256
```

Available scenarios are `baseline`, `mixed_precision`, `quantization`, `torch_compile`, and `all_compatible`. The default is `mixed_precision`.

After it prints `ready` on stderr, enter prompt file paths:

```text
etc/transform_prompt
quit
```

For pipe-based integration:

```bash
printf '%s\n' etc/transform_prompt quit | python src/prompt_inference_server.py --scenario mixed_precision --device cuda
```
