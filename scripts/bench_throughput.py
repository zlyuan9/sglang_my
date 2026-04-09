"""
Simple throughput benchmark: compare normal vs sparse (Quest) attention.
Sends prompts to a running SGLang server and measures tokens/sec and latency.

Usage:
    python bench_throughput.py [--num-prompts 5] [--max-tokens 256] [--host localhost] [--port 8000]
"""

import argparse
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

PROMPTS = [
    (
        "Explain the theory of general relativity in detail, covering spacetime curvature, the equivalence principle, and gravitational waves. "
        "Start with the historical context: Newton's law of universal gravitation dominated physics for over two centuries before Einstein published his field equations in 1915. "
        "Discuss how the equivalence principle—the idea that gravitational and inertial mass are identical—led Einstein to reconceptualize gravity not as a force but as the curvature of spacetime caused by mass and energy. "
        "Explain the mathematical framework of the Einstein field equations, including the stress-energy tensor, the Ricci curvature tensor, and the metric tensor. "
        "Describe key experimental confirmations: the perihelion precession of Mercury, the deflection of light by the Sun observed during the 1919 solar eclipse by Eddington, gravitational redshift measured by Pound and Rebka in 1959, and the direct detection of gravitational waves by LIGO in 2015. "
        "Discuss the Schwarzschild solution, black holes, event horizons, and singularities. Explain the Kerr metric for rotating black holes. "
        "Cover modern applications such as GPS satellite corrections, gravitational lensing in astronomy, and the detection of binary neutron star mergers. "
        "Discuss open problems: the incompatibility of general relativity with quantum mechanics, attempts at quantum gravity including string theory and loop quantum gravity, and the cosmological constant problem. "
        "Explain how general relativity predicts the expansion of the universe, the Big Bang, and the role of dark energy in accelerating expansion. "
        "Describe the ADM formalism, numerical relativity, and how supercomputers simulate black hole mergers to produce gravitational wave templates used by LIGO and Virgo. "
        "Finally, discuss the philosophical implications of general relativity for our understanding of space, time, causality, and determinism."
    ),
    (
        "Write a comprehensive guide to building a web application using Python and FastAPI, including authentication, database integration, and deployment. "
        "Begin with an overview of FastAPI's key features: automatic OpenAPI documentation, dependency injection, async support via ASGI, and Pydantic-based request/response validation. "
        "Compare FastAPI with Flask and Django in terms of performance, developer experience, ecosystem maturity, and community support. "
        "Walk through setting up a project: virtual environment creation, installing FastAPI and Uvicorn, project directory structure following best practices with routers, models, schemas, and services. "
        "Explain how to define RESTful endpoints with path parameters, query parameters, request bodies, and response models. Cover status codes, error handling with HTTPException, and custom exception handlers. "
        "Detail database integration using SQLAlchemy ORM with async session support, Alembic for migrations, and connection pooling strategies. "
        "Explain authentication: implement JWT-based auth with OAuth2PasswordBearer, password hashing with bcrypt, refresh tokens, role-based access control, and protecting routes with dependency injection. "
        "Cover middleware: CORS configuration, request logging, rate limiting, and custom middleware for request ID tracking. "
        "Discuss testing strategies: unit tests with pytest, integration tests with TestClient, mocking external services, and test database fixtures. "
        "Explain deployment: containerization with Docker and multi-stage builds, docker-compose for local development with PostgreSQL and Redis, CI/CD pipelines with GitHub Actions, and production deployment on AWS ECS or Kubernetes. "
        "Cover monitoring and observability: structured logging with structlog, metrics with Prometheus, distributed tracing with OpenTelemetry, and health check endpoints. "
        "Finally, discuss performance optimization: connection pooling, caching with Redis, background tasks with Celery or FastAPI's BackgroundTasks, and horizontal scaling considerations."
    ),
    (
        "Describe the history of artificial intelligence from its origins in the 1950s to modern large language models, highlighting key breakthroughs. "
        "Start with the Dartmouth Conference of 1956 where John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon coined the term 'artificial intelligence' and laid out the ambitious research agenda. "
        "Cover the early symbolic AI era: the Logic Theorist, the General Problem Solver, ELIZA, and early work on theorem proving and game playing. "
        "Discuss the first AI winter in the 1970s caused by the Lighthill Report, limited computing power, and the failure of machine translation projects. "
        "Explain the rise of expert systems in the 1980s: MYCIN, DENDRAL, R1/XCON, and the commercial boom followed by the second AI winter when maintenance costs and brittleness became apparent. "
        "Cover the statistical revolution: the shift from symbolic to statistical methods, hidden Markov models for speech recognition, support vector machines, and the resurgence of neural networks. "
        "Detail the deep learning revolution starting with the 2012 ImageNet breakthrough by AlexNet, the roles of Hinton, LeCun, and Bengio, and the importance of GPUs and large datasets. "
        "Explain key architectures: convolutional neural networks for vision, recurrent neural networks and LSTMs for sequences, the attention mechanism introduced in Bahdanau et al. 2014, and the Transformer architecture from 'Attention Is All You Need' in 2017. "
        "Cover the emergence of large language models: GPT, GPT-2, GPT-3, BERT, T5, PaLM, LLaMA, and the scaling laws discovered by Kaplan et al. "
        "Discuss reinforcement learning milestones: Deep Q-Networks playing Atari, AlphaGo defeating Lee Sedol in 2016, AlphaFold solving protein structure prediction, and robotics applications. "
        "Explain the current landscape: ChatGPT and the commercialization of AI, multimodal models, AI safety research, alignment techniques like RLHF and constitutional AI, and the societal impact including job displacement, copyright concerns, and regulation efforts worldwide."
    ),
    (
        "Explain how modern CPUs work, covering pipelining, branch prediction, cache hierarchies, out-of-order execution, and speculative execution. "
        "Begin with the fetch-decode-execute cycle and how pipelining overlaps multiple instructions to increase throughput, explaining pipeline stages in a classic 5-stage RISC pipeline. "
        "Discuss pipeline hazards: data hazards resolved with forwarding and stalling, control hazards addressed by branch prediction, and structural hazards managed through resource duplication. "
        "Explain branch prediction in depth: static prediction heuristics, dynamic predictors like 2-bit saturating counters, correlating predictors, tournament predictors, and the TAGE predictor used in modern processors. Discuss the branch target buffer and return address stack. "
        "Cover superscalar execution: multiple issue, instruction-level parallelism, and the limits imposed by true data dependencies. Explain Tomasulo's algorithm and the reservation station approach. "
        "Detail out-of-order execution: the reorder buffer, register renaming to eliminate false dependencies (WAR and WAW hazards), the physical register file, and precise exceptions. "
        "Explain the memory hierarchy in detail: L1 instruction and data caches (typically 32-64KB, 4-5 cycle latency), L2 unified cache (256KB-1MB, 12-15 cycles), L3 shared cache (8-64MB, 30-50 cycles), and main memory (DRAM, 100-300 cycles). "
        "Discuss cache design: direct-mapped, set-associative, and fully associative caches. Explain cache line size, replacement policies (LRU, pseudo-LRU, random), write policies (write-back vs write-through), and inclusion policies. "
        "Cover TLBs and virtual memory: page tables, multi-level page tables, huge pages, and the role of the TLB in address translation. "
        "Explain speculative execution and its security implications: Spectre and Meltdown vulnerabilities, side-channel attacks via cache timing, and mitigations like retpoline and kernel page table isolation. "
        "Discuss modern CPU features: SIMD extensions (SSE, AVX, AVX-512), hardware prefetching, simultaneous multithreading (SMT/Hyper-Threading), and the trend toward heterogeneous architectures with performance and efficiency cores."
    ),
    (
        "Write a detailed comparison of different sorting algorithms including their time complexity, space complexity, stability, and best use cases. "
        "Begin with simple quadratic algorithms: bubble sort, selection sort, and insertion sort. Explain why insertion sort is preferred for small arrays and nearly-sorted data despite its O(n^2) worst case, and why it's used as the base case in hybrid algorithms. "
        "Cover divide-and-conquer algorithms: merge sort with its guaranteed O(n log n) time but O(n) space overhead, and quicksort with its O(n log n) average case but O(n^2) worst case. "
        "Explain quicksort optimizations: median-of-three pivot selection, Dutch national flag partitioning for handling duplicates, tail recursion elimination, and the introsort hybrid that switches to heapsort when recursion depth exceeds a threshold. "
        "Detail heapsort: the heap data structure, heapify operation, and why heapsort guarantees O(n log n) worst case but has poor cache performance compared to quicksort. "
        "Discuss non-comparison-based sorting: counting sort, radix sort (both LSD and MSD variants), and bucket sort. Explain when these achieve O(n) time and their space requirements. "
        "Cover Timsort: the hybrid algorithm used by Python and Java, combining merge sort and insertion sort with run detection, galloping mode for merging, and its O(n) best case on partially sorted data. "
        "Explain parallel sorting algorithms: parallel merge sort, bitonic sort for GPU implementations, sample sort for distributed systems, and the practical considerations of parallelizing sorting. "
        "Discuss external sorting for data that doesn't fit in memory: the external merge sort algorithm, the replacement selection technique for initial run generation, and optimization with buffer management. "
        "Cover specialized algorithms: shell sort with different gap sequences, radix exchange sort, and the theoretical lower bound of O(n log n) for comparison-based sorting proved via decision trees. "
        "Finally, discuss practical considerations: cache efficiency, branch prediction friendliness, stability requirements, adaptive behavior on partially sorted inputs, and how standard library implementations choose algorithms."
    ),
]


def pad_prompt_to_tokens(prompt: str, target_tokens: int, model_path: str) -> str:
    """Pad a prompt by repeating its content until it reaches target_tokens."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    current_tokens = len(tokenizer.encode(prompt))
    if current_tokens >= target_tokens:
        return prompt
    # Repeat the prompt text to fill up to the target
    filler = prompt
    while current_tokens < target_tokens:
        prompt = prompt + "\n\n" + filler
        current_tokens = len(tokenizer.encode(prompt))
    # Truncate to exact token count
    token_ids = tokenizer.encode(prompt)[:target_tokens]
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def set_quest_attention(base_url: str, enabled: bool):
    """Toggle Quest sparse attention on/off."""
    resp = requests.post(f"{base_url}/set_quest_attention", json={"enabled": enabled})
    resp.raise_for_status()
    print(f"  Quest attention set to: {enabled}")


def send_request(base_url: str, model: str, prompt: str, max_tokens: int):
    """Send a single completion request, return (output_tokens, latency_sec)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    t0 = time.perf_counter()
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload)
    latency = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    usage = data["usage"]
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "latency": latency,
    }


def run_benchmark(base_url: str, model: str, prompts: list[str], max_tokens: int, concurrency: int):
    """Run all prompts and collect stats."""
    results = []
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(send_request, base_url, model, p, max_tokens): i
            for i, p in enumerate(prompts)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                print(f"    prompt {idx}: {r['completion_tokens']} tokens in {r['latency']:.2f}s")
            except Exception as e:
                print(f"    prompt {idx}: FAILED - {e}")

    wall_time = time.perf_counter() - t_start
    total_prompt_tokens = sum(r["prompt_tokens"] for r in results)
    total_completion_tokens = sum(r["completion_tokens"] for r in results)
    total_tokens = total_prompt_tokens + total_completion_tokens
    avg_latency = sum(r["latency"] for r in results) / len(results) if results else 0

    return {
        "num_requests": len(results),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "wall_time_sec": wall_time,
        "avg_latency_sec": avg_latency,
        "throughput_tok_per_sec": total_tokens / wall_time if wall_time > 0 else 0,
        "prefill_tok_per_sec": total_prompt_tokens / wall_time if wall_time > 0 else 0,
        "decode_tok_per_sec": total_completion_tokens / wall_time if wall_time > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark sparse vs normal attention throughput")
    parser.add_argument("--num-prompts", type=int, default=5, help="Number of prompts to send")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max output tokens per request")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent requests")
    parser.add_argument("--only", choices=["normal", "sparse"], help="Run only one mode")
    parser.add_argument("--prompt-tokens", type=int, default=24000,
                        help="Pad each prompt to this many tokens. Need >9600 for Quest sparsity. (0 = no padding)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="HF model path for tokenizer (auto-detected from server if not set)")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    # Get model name
    models = requests.get(f"{base_url}/v1/models").json()
    model = models["data"][0]["id"]
    print(f"Model: {model}")

    prompts = (PROMPTS * ((args.num_prompts // len(PROMPTS)) + 1))[:args.num_prompts]

    if args.prompt_tokens > 0:
        model_path = args.model_path or f"/home/xutingl/downloaded_models/{model}"
        print(f"\nPadding prompts to ~{args.prompt_tokens} tokens using tokenizer from {model_path}...")
        prompts = [pad_prompt_to_tokens(p, args.prompt_tokens, model_path) for p in prompts]
        print(f"  Done. {len(prompts)} prompts padded.")

    # Warmup
    print("\nWarmup (1 request)...")
    send_request(base_url, model, "Hello", 16)

    results = {}

    if args.only != "sparse":
        print(f"\n=== Normal Attention ({args.num_prompts} prompts, concurrency={args.concurrency}) ===")
        set_quest_attention(base_url, False)
        time.sleep(1)
        results["normal"] = run_benchmark(base_url, model, prompts, args.max_tokens, args.concurrency)

    if args.only != "normal":
        print(f"\n=== Sparse (Quest) Attention ({args.num_prompts} prompts, concurrency={args.concurrency}) ===")
        set_quest_attention(base_url, True)
        time.sleep(1)
        results["sparse"] = run_benchmark(base_url, model, prompts, args.max_tokens, args.concurrency)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for mode, r in results.items():
        print(f"\n  {mode.upper()}:")
        print(f"    Requests:           {r['num_requests']}")
        print(f"    Prompt tokens:      {r['total_prompt_tokens']}")
        print(f"    Completion tokens:  {r['total_completion_tokens']}")
        print(f"    Wall time:          {r['wall_time_sec']:.2f}s")
        print(f"    Avg latency:        {r['avg_latency_sec']:.2f}s")
        print(f"    Total throughput:   {r['throughput_tok_per_sec']:.1f} tok/s")
        print(f"    Prefill throughput: {r['prefill_tok_per_sec']:.1f} tok/s")
        print(f"    Decode throughput:  {r['decode_tok_per_sec']:.1f} tok/s")

    if "normal" in results and "sparse" in results:
        n, s = results["normal"], results["sparse"]
        overall_speedup = s["throughput_tok_per_sec"] / n["throughput_tok_per_sec"] if n["throughput_tok_per_sec"] > 0 else 0
        prefill_speedup = s["prefill_tok_per_sec"] / n["prefill_tok_per_sec"] if n["prefill_tok_per_sec"] > 0 else 0
        decode_speedup = s["decode_tok_per_sec"] / n["decode_tok_per_sec"] if n["decode_tok_per_sec"] > 0 else 0
        print(f"\n  Sparse vs Normal speedup:")
        print(f"    Overall:  {overall_speedup:.2f}x")
        print(f"    Prefill:  {prefill_speedup:.2f}x")
        print(f"    Decode:   {decode_speedup:.2f}x")


if __name__ == "__main__":
    main()
