# Static Experiment Dashboard Export

This directory provides the static dashboard export for the final HPML comparison viewable at 
https://wandb.ai/mjm2442-columbia-university/tiny-inference-latency/runs/6enskkzu?nw=nwusermjm2442

Source run: local W&B run `optimized-vs-base`, May 7, 2026.

| Metric | Baseline | Optimized | Improvement |
| --- | ---: | ---: | ---: |
| p50 latency | 8598.23 ms | 2494.24 ms | 3.45x faster |
| p95 latency | 8691.40 ms | 2923.45 ms | 2.97x faster |
| Throughput | 16.63 tok/s | 28.66 tok/s | 1.72x higher |
| Peak GPU memory | 11969 MB | 12163 MB | 1.6% higher |
| Setup latency | 8269.03 ms | 34928.11 ms | slower startup |

The raw exported table is [`latency_reports.csv`](latency_reports.csv).

Reproduce the comparison with:

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
