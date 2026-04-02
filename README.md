# Quest Attention for SGLang

This is a fork of [SGLang](https://github.com/sgl-project/sglang) with **Quest Attention** — a sparse-attention mechanism for the decode phase that scores every KV-cache page and attends only to the top-*k* most critical pages, reducing memory bandwidth and improving decode throughput for long-context inference.

Based on the paper: [*Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference*](https://arxiv.org/abs/2406.10774) (ICML 2024).

## How It Works

Standard paged-attention reads *all* KV-cache pages during decode. Quest adds a lightweight page-scoring step that selects only the most relevant pages per query, then runs FlashInfer attention on the sparse subset.

```
decode step
  │
  ├─ update_min_max()          # maintain per-page key bounding boxes
  │
  ├─ estimate_page_criticality()  # score pages via bounding-box upper bound
  │     for each dimension d:
  │       score += q[d] * max_key[d]   if q[d] >= 0
  │       score += q[d] * min_key[d]   if q[d] <  0
  │
  ├─ top-k selection           # keep k most critical + n recent pages
  │
  └─ FlashInfer paged decode   # attend only to selected pages
```

### Key idea

Each KV-cache page maintains a bounding box (`min_key`, `max_key`) per head and dimension. At decode time, the query is dot-producted against these bounds to get a tight upper bound on the maximum attention score for that page. Only the top-*k* highest-scoring pages (plus a window of recent pages for recency bias) are passed to the attention kernel.

## Changed Files

| File | What changed |
|------|-------------|
| `python/sglang/srt/layers/attention/quest_backend.py` | **New.** `QuestMHATokenToKVPool` (KV pool with per-page min/max tracking), `estimate_page_criticality()`, `quest_select_sparse_page_table()` |
| `python/sglang/srt/layers/attention/flashinfer_backend.py` | Quest sparse-attention decode path: builds sparse page table, calls FlashInfer with selected pages only |
| `python/sglang/srt/model_executor/model_runner.py` | Instantiates `QuestMHATokenToKVPool` when `--enable-quest-attention` is set; pre-computes per-request page indices each step |
| `python/sglang/srt/server_args.py` | `--enable-quest-attention` CLI flag |
| `python/sglang/srt/mem_cache/allocator.py` | Calls `invalidate_pages()` on page free so stale bounding boxes aren't reused |
| `python/sglang/srt/entrypoints/http_server.py` | `POST /set_quest_attention` endpoint for runtime toggle |
| `python/sglang/srt/kernels/quest_attention.cu` | Standalone CUDA kernel (page-level fused QKV attention with online softmax) |

## Usage

### Launch with Quest enabled

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-quest-attention \
    --page-size 16 \
    --disable-cuda-graph
```

- `--page-size 16` (or larger) is required — Quest operates at page granularity, so `page_size=1` is meaningless and would double memory.
- `--disable-cuda-graph` is required because the min/max tracking uses `torch.unique()` which triggers host-device syncs incompatible with CUDA graph capture.

### Runtime toggle

Quest can be enabled/disabled at runtime without restarting the server:

```bash
# Enable
curl -X POST http://localhost:30000/set_quest_attention \
     -H "Content-Type: application/json" \
     -d '{"enabled": true}'

# Disable (falls back to standard FlashInfer path)
curl -X POST http://localhost:30000/set_quest_attention \
     -H "Content-Type: application/json" \
     -d '{"enabled": false}'
```

### Configuration

`QuestMHATokenToKVPool` accepts two tuning parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_quest_pages` | 256 | Total page budget per request (top-k scored + recent) |
| `num_recent_pages` | 32 | Pages from the end of the sequence always included (recency window) |

## Architecture

```
QuestMHATokenToKVPool (extends MHATokenToKVPool)
├── k_buffer / v_buffer          # standard paged KV cache
├── min_k_buffer[layer]          # per-page key lower bounds  [num_pages, H, D]
├── max_k_buffer[layer]          # per-page key upper bounds  [num_pages, H, D]
└── page_valid[layer]            # tracks which pages have valid bounds

set_kv_buffer()
└── super().set_kv_buffer()      # write K,V to cache
└── update_min_max()             # update bounding boxes via scatter_reduce

PagedTokenToKVPoolAllocator.free()
└── invalidate_pages()           # reset bounds on freed pages

FlashInferAttnBackend.forward_decode()
├── quest_select_sparse_page_table()   # score + select top-k pages
│   ├── estimate_page_criticality()    # bounding-box scoring
│   └── torch.topk()                   # page selection
└── BatchDecodeWithPagedKVCacheWrapper # FlashInfer on sparse page table
```

## Building the CUDA Kernel (optional)

The standalone CUDA kernel in `python/sglang/srt/kernels/` is a reference implementation. The main integration uses FlashInfer's paged decode with a sparse page table, so building this kernel is not required for the default path.

```bash
cd python/sglang/srt/kernels
pip install -e .
python quest_kernel_test.py
```

## Upstream

This fork is based on [JhengLu/sglang_my](https://github.com/JhengLu/sglang_my), which tracks [sgl-project/sglang](https://github.com/sgl-project/sglang).
