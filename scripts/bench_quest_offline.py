"""
Offline throughput benchmark: compare normal vs Quest sparse attention.

Uses sglang's Engine directly (no server needed), giving direct control
over batch size via --num-prompts. Measures prefill, decode, and overall
throughput separately, and computes speedup when both modes are run.

Single engine with Quest KV pool always active. Toggles the Quest decode
path via flag file between runs (same as bench_throughput.py server mode),
so both modes share identical prefill overhead.

Usage:
    python bench_quest_offline.py
    python bench_quest_offline.py --num-prompts 8 --batch-size 4
    python bench_quest_offline.py --only normal
    python bench_quest_offline.py --only quest
"""

import argparse
import dataclasses
import json
import logging
import os
import random
import time

import numpy as np

from sglang.bench_serving import sample_random_requests, get_tokenizer, set_ulimit
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.server_args import ServerArgs

# Flag file used to toggle Quest decode path at runtime (IPC with flashinfer_backend)
_QUEST_FLAG = "/tmp/.sglang_quest_enabled"

logger = logging.getLogger(__name__)


def set_quest_enabled(enabled: bool):
    """Toggle Quest sparse decode path via flag file."""
    if enabled:
        open(_QUEST_FLAG, "w").close()
    elif os.path.exists(_QUEST_FLAG):
        os.remove(_QUEST_FLAG)


def run_once(engine, prompts, output_len, ignore_eos=True, extra_request_body=None):
    """Run a batch through the engine, return timing stats with separate prefill/decode."""
    sampling_params = [
        {
            "temperature": 0,
            "max_new_tokens": output_len,
            "ignore_eos": ignore_eos,
            **(extra_request_body or {}),
        }
        for _ in prompts
    ]

    t0 = time.perf_counter()
    gen_out = engine.generate(prompt=prompts, sampling_params=sampling_params)
    wall = time.perf_counter() - t0

    total_input = sum(o["meta_info"]["prompt_tokens"] for o in gen_out)
    total_output = sum(o["meta_info"]["completion_tokens"] for o in gen_out)

    # Extract per-request prefill and decode times from engine metrics.
    # prefill_finished_ts = time.time() when first token was generated (TTFT)
    # decode_finished_ts  = time.time() when generation finished
    # request_received_ts = time.time() when request was created
    prefill_times = []
    decode_times = []
    for o in gen_out:
        mi = o["meta_info"]
        t_recv = mi.get("request_received_ts")
        t_prefill = mi.get("prefill_finished_ts")
        t_decode = mi.get("decode_finished_ts")
        if t_recv is not None and t_prefill is not None:
            prefill_times.append(t_prefill - t_recv)
        if t_prefill is not None and t_decode is not None:
            decode_times.append(t_decode - t_prefill)

    # For throughput = tokens / time, use the max (wall-clock span) of each phase.
    prefill_wall = max(prefill_times) if prefill_times else wall
    decode_wall = max(decode_times) if decode_times else wall

    return {
        "num_requests": len(gen_out),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "wall_sec": wall,
        "prefill_sec": prefill_wall,
        "decode_sec": decode_wall,
        "prefill_tok_per_sec": total_input / prefill_wall if prefill_wall > 0 else 0,
        "decode_tok_per_sec": total_output / decode_wall if decode_wall > 0 else 0,
        "total_tok_per_sec": (total_input + total_output) / wall if wall > 0 else 0,
    }


def print_result(label, r):
    print(f"\n  {label}:")
    print(f"    Requests:            {r['num_requests']}")
    print(f"    Input tokens:        {r['total_input_tokens']}")
    print(f"    Output tokens:       {r['total_output_tokens']}")
    print(f"    Wall time:           {r['wall_sec']:.2f}s")
    print(f"    Prefill time:        {r['prefill_sec']:.2f}s")
    print(f"    Decode time:         {r['decode_sec']:.2f}s")
    print(f"    Prefill throughput:  {r['prefill_tok_per_sec']:.1f} tok/s")
    print(f"    Decode throughput:   {r['decode_tok_per_sec']:.1f} tok/s")
    print(f"    Total throughput:    {r['total_tok_per_sec']:.1f} tok/s")


def print_speedup(normal, quest):
    def safe_div(a, b):
        return a / b if b > 0 else float("nan")

    print(f"\n  Quest vs Normal speedup:")
    print(f"    Prefill:  {safe_div(quest['prefill_tok_per_sec'], normal['prefill_tok_per_sec']):.2f}x")
    print(f"    Decode:   {safe_div(quest['decode_tok_per_sec'], normal['decode_tok_per_sec']):.2f}x")
    print(f"    Overall:  {safe_div(quest['total_tok_per_sec'], normal['total_tok_per_sec']):.2f}x")


def make_default_server_args(**overrides):
    """Construct ServerArgs with defaults matching start_sglang_qwen3_4b.sh."""
    defaults = dict(
        model_path="/home/xutingl/downloaded_models/Qwen3-VL-4B-Instruct",
        page_size=32,
        disable_cuda_graph=True,
        context_length=32768,
        mem_fraction_static=0.80,
        trust_remote_code=True,
        enable_metrics=True,  # needed for per-request prefill/decode timing
        enable_quest_attention=True,  # always use QuestMHATokenToKVPool
    )
    defaults.update(overrides)
    return ServerArgs(**defaults)


def main():
    parser = argparse.ArgumentParser(description="Offline Quest throughput benchmark")

    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Max concurrent requests")
    parser.add_argument("--input-len", type=int, default=24000,
                        help="Input token count per prompt (random tokens)")
    parser.add_argument("--output-len", type=int, default=512,
                        help="Max output tokens per prompt")
    parser.add_argument("--only", choices=["normal", "quest"],
                        help="Run only one mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-prompts", type=int, default=2,
                        help="Number of short warmup prompts (0 to skip)")

    args = parser.parse_args()
    server_args = make_default_server_args(max_running_requests=args.batch_size)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    set_ulimit()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Build tokenizer for random prompt generation
    tokenizer_id = server_args.tokenizer_path or server_args.model_path
    tokenizer = get_tokenizer(tokenizer_id)

    # Generate random prompts
    reqs = sample_random_requests(
        input_len=args.input_len,
        output_len=args.output_len,
        num_prompts=args.num_prompts,
        range_ratio=0.0,
        tokenizer=tokenizer,
        dataset_path="",
    )
    prompts = [r.prompt for r in reqs]

    # Warmup prompts (short)
    warmup_reqs = sample_random_requests(
        input_len=256,
        output_len=16,
        num_prompts=max(args.warmup_prompts, 1),
        range_ratio=0.0,
        tokenizer=tokenizer,
        dataset_path="",
    )
    warmup_prompts = [r.prompt for r in warmup_reqs]

    # Single engine with Quest KV pool always active.
    # Toggle Quest decode path via flag file between runs.
    print("\nStarting engine (Quest KV pool enabled)...")
    engine = Engine(**dataclasses.asdict(server_args))

    if args.warmup_prompts > 0:
        print("  Warmup...")
        set_quest_enabled(False)
        run_once(engine, warmup_prompts, output_len=16)

    results = {}

    # --- Normal mode (Quest decode disabled) ---
    if args.only != "quest":
        print(f"\n{'=' * 60}")
        print("Normal Attention (Quest decode disabled)")
        print("=" * 60)
        set_quest_enabled(False)
        time.sleep(0.5)
        print(f"  Benchmarking {args.num_prompts} prompts, input={args.input_len}, output={args.output_len}...")
        results["normal"] = run_once(engine, prompts, args.output_len)
        print_result("NORMAL", results["normal"])

    # --- Quest mode (Quest decode enabled) ---
    if args.only != "normal":
        print(f"\n{'=' * 60}")
        print("Quest Sparse Attention (Quest decode enabled)")
        print("=" * 60)
        set_quest_enabled(True)
        time.sleep(0.5)
        print(f"  Benchmarking {args.num_prompts} prompts, input={args.input_len}, output={args.output_len}...")
        results["quest"] = run_once(engine, prompts, args.output_len)
        print_result("QUEST", results["quest"])

    # Cleanup
    set_quest_enabled(False)
    engine.shutdown()

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print("=" * 60)
    for label, r in results.items():
        print_result(label.upper(), r)

    if "normal" in results and "quest" in results:
        print_speedup(results["normal"], results["quest"])

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
