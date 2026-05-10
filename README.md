# HPML Final Project: Tiny Robotic Inference

> **Course:** High Performance Machine Learning
> **Semester:** Spring 2026
> **Instructor:** Dr. Kaoutar El Maghraoui

## Team Information

- **Team Name:** Tiny Robotic Inference
- **Members:**
  - Michael Mih (mjm2442) - inference benchmarking, W&B logging, ROS demo integration

## Submission

- **GitHub repository:** [https://github.com/michael-mih/tiny-inference](https://github.com/michael-mih/tiny-inference)
- **Final report source:** [`deliverables/Tiny_Robotic_Inference_Paper.tex`](deliverables/Tiny_Robotic_Inference_Paper.tex)
- **Final report PDF target:** `deliverables/Tiny_Robotic_Inference_HPML_Final_Report.pdf` after exporting from Overleaf/IEEE.
- **Final presentation:** [`deliverables/Tiny_Robotic_Inference_Presentation.pdf`](deliverables/Tiny_Robotic_Inference_Presentation.pdf)
- **Experiment-tracking dashboard export:** [`results/dashboard/README.md`](results/dashboard/README.md)

The committed dashboard export mirrors the local W&B run `optimized-vs-base`. If the W&B project is made public, add that public URL here and keep the static export as a fallback.

## 1. Problem Statement

This project optimizes inference latency for a small language-to-action pipeline that converts natural-language robot instructions into strict JSON action plans. The target workload is inference for `Qwen/Qwen2.5-3B-Instruct`, with the bottleneck centered on single-request decode latency and GPU memory pressure. The final comparison is limited to two tests: a full-precision Transformers baseline and an optimized vLLM serving path.

## 2. Model/Application Description

- **Model architecture:** `Qwen/Qwen2.5-3B-Instruct`, a 3B-parameter instruction-tuned causal language model.
- **Framework:** PyTorch, Hugging Face Transformers, vLLM, and ROS 2 for the downstream demo bridge.
- **Dataset/workload:** a fixed prompt in [`etc/transform_prompt`](etc/transform_prompt) that requests a JSON robot action plan. No external dataset is committed.
- **Custom logic:** strict JSON action validation, optional JSON repair, reusable prompt server, W&B latency logging, and a ROS 2 symbolic pick/place demo adapter.
- **Hardware target:** measured on 1x NVIDIA Tesla T4 16 GB with CUDA 12.4 and Python 3.10.15.

## 3. Final Results Summary

Measured from the local W&B run `optimized-vs-base` on May 7, 2026 with 3 warmup runs, 10 measured runs, and `max_new_tokens=256`.

| Metric | Baseline | Optimized | Improvement |
| --- | ---: | ---: | ---: |
| Inference latency, p50 | 8598.23 ms | 2494.24 ms | 3.45x faster |
| Inference latency, p95 | 8691.40 ms | 2923.45 ms | 2.97x faster |
| Throughput | 16.63 tok/s | 28.66 tok/s | 1.72x higher |
| Peak GPU memory | 11969 MB | 12163 MB | 1.6% higher |
| Setup latency | 8269.03 ms | 34928.11 ms | slower startup |

**Baseline:** Transformers, CUDA, float32.  
**Optimized:** vLLM, CUDA, float16, automatic prefix caching, `max_num_seqs=1`, `gpu_memory_utilization=0.8`.

**Headline result:** vLLM plus float16 and prefix caching reduced steady-state p50 inference latency from 8.60 s to 2.49 s on a Tesla T4, a 3.45x speedup, while using roughly the same peak GPU memory.

## 4. Repository Structure

```text
.
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- configs/
|   |-- baseline.yaml
|   `-- optimized.yaml
|-- etc/
|   `-- transform_prompt
|-- deliverables/
|   |-- Tiny_Robotic_Inference_Paper.tex
|   `-- Tiny_Robotic_Inference_Presentation.pdf
|-- results/
|   `-- dashboard/
|-- scripts/
|   |-- run_baseline.sh
|   `-- run_optimized.sh
|-- src/
|   |-- benchmark.py
|   |-- language_to_action.py
|   |-- output_inference.py
|   |-- prompt_inference_server.py
|   `-- wandb_latency.py
`-- ros2_ws/
    `-- src/tiny_inference_ros/
```

The public test names are intentionally limited to:

- `baseline`: Transformers full-precision inference.
- `optimized`: vLLM float16 inference with automatic prefix caching.

## 5. Reproducibility Instructions

### A. Environment Setup

```bash
git clone https://github.com/michael-mih/tiny-inference.git
cd tiny-inference

python3.10 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

System requirements: Python 3.10+, CUDA 12.x, and a CUDA GPU with enough memory for Qwen2.5-3B inference. The reported run used the pinned package versions in [`requirements.txt`](requirements.txt). If PyTorch cannot resolve a CUDA wheel automatically on your system, install the matching CUDA wheel for your platform before installing the remaining packages.

### B. Experiment Tracking Dashboard

The benchmark can log scalar metrics and a final `latency_reports` table to W&B.

```bash
wandb login
```

Add these flags to any benchmark command:

```bash
--wandb-project tiny-inference-latency --wandb-run-name baseline-vs-optimized
```

The static export of the final comparison is committed under [`results/dashboard/`](results/dashboard/). The W&B link is [here](https://wandb.ai/mjm2442-columbia-university/tiny-inference-latency/runs/6enskkzu?nw=nwusermjm2442). 

### C. Dataset

No dataset download is required. The reproducible workload is the prompt stored in:

```text
etc/transform_prompt
```

### D. Baseline Test

```bash
bash scripts/run_baseline.sh
```

Equivalent direct command:

```bash
python src/benchmark.py \
  --device cuda \
  --scenario baseline \
  --base-only \
  --warmup-runs 3 \
  --benchmark-runs 10 \
  --max-new-tokens 256
```

### E. Optimized Test

```bash
bash scripts/run_optimized.sh
```

Equivalent direct command:

```bash
python src/benchmark.py \
  --device cuda \
  --scenario optimized \
  --base-only \
  --warmup-runs 3 \
  --benchmark-runs 10 \
  --max-new-tokens 256
```

### F. Quickstart: Reproduce the Headline Result

Run both named tests in one benchmark invocation:

```bash
python src/benchmark.py \
  --device cuda \
  --scenario baseline \
  --scenario optimized \
  --base-only \
  --warmup-runs 3 \
  --benchmark-runs 10 \
  --max-new-tokens 256 \
  --wandb-project tiny-inference-latency \
  --wandb-run-name baseline-vs-optimized
```

### G. Single Inference

```bash
python src/output_inference.py --profile baseline --repeat 5
python src/output_inference.py --profile optimized --repeat 5
```

### H. Persistent Prompt Runner

The prompt server keeps one selected test loaded and reads prompt file paths from stdin.

```bash
python src/prompt_inference_server.py \
  --scenario optimized \
  --device cuda \
  --warmup-runs 3 \
  --max-new-tokens 80 \
  --repair-attempts 0
```

Available scenarios are `baseline` and `optimized`. The default is `optimized`.

After the server prints `ready` on stderr:

```text
etc/transform_prompt
quit
```

For pipe-based integration:

```bash
printf '%s\n' etc/transform_prompt quit | python src/prompt_inference_server.py --scenario optimized --device cuda
```

### I. ROS 2 Demo

The ROS 2 package consumes generated JSON plans and runs a scripted pick/place demo. See [`ros2_ws/src/tiny_inference_ros/README.md`](ros2_ws/src/tiny_inference_ros/README.md).

## 6. Results and Observations

- The optimized path improves steady-state latency substantially because vLLM serves the same prompt format with lower per-request decode overhead and float16 weights.
- Startup is slower for the optimized path because vLLM engine initialization takes longer than loading the Transformers baseline.
- Peak GPU memory is similar across both tests on the measured T4 run; the optimized path used about 194 MB more peak memory.
https://wandb.ai/mjm2442-columbia-university/tiny-inference-latency/runs/6enskkzu?nw=nwusermjm2442
## 7. Notes

- Source files live under `src/`.
- Reproduction scripts live under `scripts/` and follow the required `run_baseline.sh` / `run_optimized.sh` naming.
- W&B local run files, Python caches, and virtual environments are ignored by git.
- Secrets such as W&B tokens should be provided through environment variables or `wandb login`; do not commit them.

### AI Tool Use Disclosure

**Did your team use any AI tool in completing this project?**

- [ ] No, we did not use any AI tool.
- [x] Yes, we used AI assistance as described below.

**Tool(s) used:** OpenAI Codex.

**Specific purpose:** Cleaned up the benchmark surface to expose only the final `baseline` and `optimized` tests; rewrote the README around the HPML template; added reproducibility scripts and dashboard export files; checked grammar on the final paper.

**Sections affected:** `README.md`, `src/benchmark.py`, `src/output_inference.py`, `src/prompt_inference_server.py`, `deliverables/Tiny_Robotic_Inference_Paper.tex`, `scripts/`, `configs/`, and `results/dashboard/`.

**How we verified correctness:** Re-ran static Python compilation checks and confirmed the public scenario list only contains `baseline` and `optimized`. Reported performance numbers come from the local W&B benchmark table and can be reproduced with the quickstart command above.

### License

This repository is released under the MIT License. See [`LICENSE`](LICENSE).

### Citation

```bibtex
@misc{tinyinference2026hpml,
  title  = {Tiny Robotic Inference},
  author = {Mih, Michael},
  year   = {2026},
  note   = {HPML Spring 2026 Final Project, Columbia University},
  url    = {https://github.com/michael-mih/tiny-inference}
}
```

### Contact

Open a GitHub issue at [https://github.com/michael-mih/tiny-inference/issues](https://github.com/michael-mih/tiny-inference/issues).
