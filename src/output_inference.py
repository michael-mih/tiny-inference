import argparse
import json
import statistics
import time

import torch

from language_to_action import (
    InferenceConfig,
    describe_inference_config,
    finalize_plan_result,
    load_inference_session,
    load_prompt,
    synchronize_device,
)
from wandb_latency import add_wandb_args, init_wandb_latency_logger


def build_inference_config(profile):
    if profile == "optimized":
        return InferenceConfig(
            model_name="Qwen/Qwen2.5-3B-Instruct",
            device="cuda",
            backend="transformers",
            precision="bfloat16",
            quantization="bitsandbytes-4bit",
            use_torch_compile=True,
            compile_mode="reduce-overhead",
        )

    return InferenceConfig()


def read_vram_metrics(device):
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return None

    return {
        "allocated_vram_mb": torch.cuda.memory_allocated(device=device) / (1024 ** 2),
        "reserved_vram_mb": torch.cuda.memory_reserved(device=device) / (1024 ** 2),
    }


def inference_metadata(config, profile):
    return {
        "profile": profile,
        "model_name": config.model_name,
        "backend": config.backend,
        "requested_device": config.device,
        "device": config.resolved_device(),
        "precision": config.precision,
        "quantization": config.quantization,
        "use_torch_compile": config.use_torch_compile,
        "compile_mode": config.compile_mode if config.use_torch_compile else "none",
        "inference_config": describe_inference_config(config),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single inference request and print the generated plan.")
    parser.add_argument(
        "--profile",
        choices=("default", "optimized"),
        default="default",
        help="Choose the inference configuration to preload before measuring.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate per request.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
        help="Number of untimed warmup requests to run before measuring steady-state latency.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of timed sequential inferences to run after the model is loaded.",
    )
    parser.add_argument(
        "--repair-attempts",
        type=int,
        default=1,
        help="Maximum number of repair attempts if the raw output is not valid JSON.",
    )
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
        help="Disable latency telemetry in the output.",
    )
    add_wandb_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1.")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs cannot be negative.")
    if args.repair_attempts < 0:
        raise ValueError("--repair-attempts cannot be negative.")

    instruction = load_prompt()
    inference_config = build_inference_config(args.profile)
    wandb_logger = init_wandb_latency_logger(
        args,
        config={
            "profile": args.profile,
            "model_name": inference_config.model_name,
            "device": inference_config.device,
            "backend": inference_config.backend,
            "precision": inference_config.precision,
            "quantization": inference_config.quantization,
            "use_torch_compile": inference_config.use_torch_compile,
            "compile_mode": inference_config.compile_mode,
            "max_new_tokens": args.max_new_tokens,
            "warmup_runs": args.warmup_runs,
            "repeat": args.repeat,
            "repair_attempts": args.repair_attempts,
        },
        job_type="single-inference",
        table_key="inference_latency_report",
    )

    try:
        setup_start_time = time.perf_counter()
        session = load_inference_session(inference_config=inference_config)
        device = session.config.resolved_device()
        synchronize_device(device)
        setup_latency_ms = (time.perf_counter() - setup_start_time) * 1000.0
        post_load_vram = read_vram_metrics(device)

        prepared_request = session.prepare_instruction(instruction)
        session.warmup(
            prepared_request,
            runs=args.warmup_runs,
            max_new_tokens=args.max_new_tokens,
        )
        post_warmup_vram = read_vram_metrics(device)

        latencies_ms = []
        peak_vram_mb = []
        raw_result = None
        for _ in range(args.repeat):
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device=device)

            synchronize_device(device)
            start_time = time.perf_counter()
            raw_result = session.generate_prepared_output(
                prepared_request,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            synchronize_device(device)
            latencies_ms.append((time.perf_counter() - start_time) * 1000.0)

            if torch.cuda.is_available() and str(device).startswith("cuda"):
                peak_vram_mb.append(torch.cuda.max_memory_allocated(device=device) / (1024 ** 2))

        result = finalize_plan_result(
            raw_result,
            inference_session=session,
            max_new_tokens=args.max_new_tokens,
            repair_attempts=args.repair_attempts,
        )

        avg_model_latency_ms = statistics.mean(latencies_ms)
        min_model_latency_ms = min(latencies_ms)
        max_model_latency_ms = max(latencies_ms)
        peak_vram_during_inference_mb = max(peak_vram_mb) if peak_vram_mb else None

        wandb_record = {
            **inference_metadata(session.config, args.profile),
            "status": "completed",
            "setup_latency_ms": setup_latency_ms,
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.repeat,
            "avg_model_latency_ms": avg_model_latency_ms,
            "min_model_latency_ms": min_model_latency_ms,
            "max_model_latency_ms": max_model_latency_ms,
            "input_token_count": result["input_token_count"],
            "output_token_count": result["output_token_count"],
            "max_new_tokens": args.max_new_tokens,
            "repair_attempts": args.repair_attempts,
            "peak_vram_during_inference_mb": peak_vram_during_inference_mb,
        }
        if post_load_vram is not None:
            wandb_record.update(
                {
                    "allocated_vram_after_load_mb": post_load_vram["allocated_vram_mb"],
                    "reserved_vram_after_load_mb": post_load_vram["reserved_vram_mb"],
                    "allocated_vram_after_warmup_mb": post_warmup_vram["allocated_vram_mb"],
                    "reserved_vram_after_warmup_mb": post_warmup_vram["reserved_vram_mb"],
                }
            )
        wandb_logger.log(wandb_record)
        wandb_logger.update_summary(
            {
                "avg_model_latency_ms": avg_model_latency_ms,
                "max_model_latency_ms": max_model_latency_ms,
                "setup_latency_ms": setup_latency_ms,
                "output_token_count": result["output_token_count"],
            }
        )

        print(json.dumps(result["plan"], indent=2))
        if not args.no_telemetry:
            print(f"\nsetup_latency_ms: {setup_latency_ms:.2f}")
            print(f"warmup_runs: {args.warmup_runs}")
            print(f"measured_runs: {args.repeat}")
            print(f"avg_model_latency_ms: {avg_model_latency_ms:.2f}")
            print(f"min_model_latency_ms: {min_model_latency_ms:.2f}")
            print(f"max_model_latency_ms: {max_model_latency_ms:.2f}")
            print(f"input_token_count: {result['input_token_count']}")
            print(f"output_token_count: {result['output_token_count']}")
            if post_load_vram is None:
                print("allocated_vram_after_load_mb: unavailable")
                print("reserved_vram_after_load_mb: unavailable")
                print("allocated_vram_after_warmup_mb: unavailable")
                print("reserved_vram_after_warmup_mb: unavailable")
                print("peak_vram_during_inference_mb: unavailable")
            else:
                print(f"allocated_vram_after_load_mb: {post_load_vram['allocated_vram_mb']:.2f}")
                print(f"reserved_vram_after_load_mb: {post_load_vram['reserved_vram_mb']:.2f}")
                print(f"allocated_vram_after_warmup_mb: {post_warmup_vram['allocated_vram_mb']:.2f}")
                print(f"reserved_vram_after_warmup_mb: {post_warmup_vram['reserved_vram_mb']:.2f}")
                print(f"peak_vram_during_inference_mb: {peak_vram_during_inference_mb:.2f}")
    except Exception as exc:
        wandb_logger.log(
            {
                **inference_metadata(inference_config, args.profile),
                "status": "failed",
                "status_reason": str(exc),
                "max_new_tokens": args.max_new_tokens,
                "warmup_runs": args.warmup_runs,
                "measured_runs": args.repeat,
            }
        )
        raise
    finally:
        wandb_logger.finish()


if __name__ == "__main__":
    main()
