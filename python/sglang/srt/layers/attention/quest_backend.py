import logging
import os
from typing import Optional

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

logger = logging.getLogger(__name__)

# Flag file path used for runtime IPC: presence = QUEST enabled.
_QUEST_FLAG = "/tmp/.sglang_quest_enabled"


def estimate_page_criticality(
    query,      # [batch, num_heads, head_dim]
    min_keys,   # [num_pages, num_heads, head_dim]
    max_keys,   # [num_pages, num_heads, head_dim]
) -> torch.Tensor:  # scores: [batch, num_heads, num_pages]
    """Quest bounding-box upper bound on attention scores per page.

    For each dimension d the tightest upper bound on q[d]*k[d] when
    k[d] in [min_keys[d], max_keys[d]] is:
        q[d] * max_keys[d]   if q[d] >= 0
        q[d] * min_keys[d]   if q[d] <  0

    Returns: [batch, num_heads, num_pages]
    """
    # [B, H, 1, D]
    q = query.unsqueeze(2)
    # [1, H, P, D]  (swap P and H dims so H aligns with query)
    k_min = min_keys.unsqueeze(0).transpose(1, 2)
    k_max = max_keys.unsqueeze(0).transpose(1, 2)

    criticality = torch.where(q >= 0, q * k_max, q * k_min)  # [B, H, P, D]
    return criticality.sum(dim=-1)  # [B, H, P]

def get_top_k(scores, k):
    top, _ = torch.topk(scores, k, dim=-1)
    return top

def get_top_k_indices(scores, k):
    """Return indices of top-k pages per (batch, head). Shape: [batch, num_heads, k]."""
    _, indices = torch.topk(scores, k, dim=-1)
    return indices

class QuestMHATokenToKVPool(MHATokenToKVPool):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
        num_quest_pages: int = 256,
        num_recent_pages: int = 32,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            head_num,
            head_dim,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
            enable_alt_stream,
            enable_kv_cache_copy,
        )
        self.num_quest_pages = num_quest_pages
        self.num_recent_pages = num_recent_pages

        # The allocator reserves page 0 for padding and hands out pages
        # 1 .. size//page_size.  We need size//page_size + 1 entries so
        # that every valid page index (including the last one) is in bounds.
        num_page_slots = size // page_size + 1

        self.min_k_buffer = [torch.full(
            (num_page_slots, head_num, head_dim),
            float('inf'),
            dtype=torch.float32,
            device=device,
        ) for _ in range(self.layer_num)]

        self.max_k_buffer = [torch.full(
            (num_page_slots, head_num, head_dim),
            float('-inf'),
            dtype=torch.float32,
            device=device,
        ) for _ in range(self.layer_num)]

        self.page_valid = [torch.zeros(
            num_page_slots, dtype=torch.bool, device=device,
        ) for _ in range(self.layer_num)]

    def update_min_max(self, layer_id, loc):
        k_buf = self.k_buffer[layer_id]
        keys = k_buf[loc].to(torch.float32)
        pages = (loc // self.page_size).to(torch.int64)

        # Reset pages that were freed and reallocated (page_valid=False) so
        # stale min/max from a previous allocation doesn't corrupt the new
        # bounding boxes.  torch.unique() requires a host-device sync which
        # is incompatible with CUDA graph capture; Quest currently requires
        # --disable-cuda-graph.
        unique_pages = pages.unique()
        stale_mask = ~self.page_valid[layer_id][unique_pages]
        stale_pages = unique_pages[stale_mask]
        if stale_pages.numel() > 0:
            self.min_k_buffer[layer_id][stale_pages] = float('inf')
            self.max_k_buffer[layer_id][stale_pages] = float('-inf')

        page_idx = pages.view(-1, 1, 1).expand_as(keys)
        self.min_k_buffer[layer_id].scatter_reduce_(
            0, page_idx, keys, reduce='amin', include_self=True
        )
        self.max_k_buffer[layer_id].scatter_reduce_(
            0, page_idx, keys, reduce='amax', include_self=True
        )
        self.page_valid[layer_id][unique_pages] = True

    def invalidate_pages(self, page_indices: torch.Tensor):
        """Reset page representations when pages are freed.

        Called by the allocator when pages are returned to the free pool so
        that stale bounding boxes from a previous allocation are not carried
        over when the page is reused.
        """
        for layer_id in range(self.layer_num):
            self.min_k_buffer[layer_id][page_indices] = float('inf')
            self.max_k_buffer[layer_id][page_indices] = float('-inf')
            self.page_valid[layer_id][page_indices] = False

    def set_kv_buffer(
        self,
        layer,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        super().set_kv_buffer(
            layer, 
            loc,
            cache_k,
            cache_v,
            k_scale,
            v_scale,
            layer_id_override
        ) 
        self.update_min_max(layer.layer_id, loc)  # Quest logic


# ---------------------------------------------------------------------------
# Main integration point: ties quest_backend scoring to the CUDA kernel
# ---------------------------------------------------------------------------

def quest_select_sparse_page_table(
    q: torch.Tensor,               # [B, Hq, D]
    layer_id: int,
    kv_pool: "QuestMHATokenToKVPool",
    page_indices_per_req: list,    # List[Tensor[P_b]] pool page indices per request
    seq_lens_cpu: list,            # List[int] sequence lengths (CPU)
) -> tuple:                        # (kv_indptr[B+1], kv_indices[total], kv_last_page_len[B]) — int32 CUDA
    """
    QUEST sparse page selection using FlashInfer-compatible output.

    For each request:
      1. Score pages via estimate_page_criticality (max-pooled over KV heads for shared mask)
      2. Select top-k pages sorted by position for causal correctness
      3. Compute kv_last_page_len accounting for partially-filled last pages

    Returns a sparse paged-KV table (kv_indptr, kv_indices, kv_last_page_len) that can be
    passed directly to BatchDecodeWithPagedKVCacheWrapper.begin_forward() + forward() with
    the KV pool viewed as [total_pages, page_size, Hkv, D].
    """
    B, Hq, D = q.shape
    Hkv = kv_pool.min_k_buffer[layer_id].shape[1]
    k_budget = kv_pool.num_quest_pages
    n_recent = kv_pool.num_recent_pages
    ps = kv_pool.page_size

    # GQA: reduce query to KV head count via mean over groups (matches
    # the reference Quest algorithm — averaging is a better approximation
    # for MQA/GQA than max-pooling).
    if Hq > Hkv:
        q_kv = q.view(B, Hkv, Hq // Hkv, D).mean(dim=2)  # [B, Hkv, D]
    else:
        q_kv = q  # [B, Hkv, D]

    kv_indices_parts = []
    kv_last_page_len_list = []

    for b in range(B):
        page_idxs = page_indices_per_req[b]
        P = page_idxs.shape[0]
        slen = int(seq_lens_cpu[b])

        if P <= k_budget:
            selected = page_idxs
        else:
            n_recent_clamped = min(n_recent, k_budget)
            scored_budget = k_budget - n_recent_clamped

            recent_page_idxs = page_idxs[P - n_recent_clamped:]
            older_page_idxs  = page_idxs[:P - n_recent_clamped]

            if scored_budget == 0 or older_page_idxs.shape[0] == 0:
                selected = recent_page_idxs
            else:
                clamped_idxs = older_page_idxs.clamp(
                    0, kv_pool.min_k_buffer[layer_id].shape[0] - 1
                )
                min_k = kv_pool.min_k_buffer[layer_id][clamped_idxs]    # [P-n, Hkv, D]
                max_k = kv_pool.max_k_buffer[layer_id][clamped_idxs]    # [P-n, Hkv, D]
                valid = kv_pool.page_valid[layer_id][clamped_idxs]      # [P-n]

                scores = estimate_page_criticality(
                    q_kv[b:b+1].to(min_k.dtype), min_k, max_k
                )  # [1, Hkv, P-n]

                # Sum over KV heads (reference aggregation — a page is critical
                # if its total contribution across all heads is high).
                page_scores = scores[0].sum(dim=0)                       # [P-n]
                page_scores = torch.where(
                    valid, page_scores,
                    torch.full_like(page_scores, float("-inf")),
                )

                n_top = min(scored_budget, older_page_idxs.shape[0])
                _, top_local = torch.topk(page_scores, n_top)
                scored_sorted = older_page_idxs[top_local.sort().values]
                selected = torch.cat([scored_sorted, recent_page_idxs])

        kv_indices_parts.append(selected.to(dtype=torch.int32))

        # kv_last_page_len: the final entry in selected[] may be partially filled.
        # The actual last allocated page of the sequence always has
        #   (slen - 1) % ps + 1  valid tokens; all earlier pages are full.
        actual_last_pool_idx = page_idxs[-1].item()
        last_selected_pool_idx = selected[-1].item()
        if last_selected_pool_idx == actual_last_pool_idx:
            kv_last_page_len_list.append((slen - 1) % ps + 1)
        else:
            kv_last_page_len_list.append(ps)

    # Build kv_indptr [B+1] on the same device as q
    page_counts = [x.shape[0] for x in kv_indices_parts]
    kv_indptr = torch.zeros(B + 1, dtype=torch.int32, device=q.device)
    cumsum = 0
    for i, c in enumerate(page_counts):
        cumsum += c
        kv_indptr[i + 1] = cumsum

    kv_indices = torch.cat(kv_indices_parts, dim=0)  # [total_selected_pages] int32 CUDA
    kv_last_page_len = torch.tensor(
        kv_last_page_len_list, dtype=torch.int32, device=q.device
    )

    return kv_indptr, kv_indices, kv_last_page_len


# ---------------------------------------------------------------------------
# test cases
# ---------------------------------------------------------------------------

#test cases
def test_estimate_page_criticality():
    # Define dimensions
    batch = 2
    num_heads = 4
    num_pages = 8
    head_dim = 16

    # Create random input tensors
    query = torch.randn(batch, num_heads, head_dim)
    min_keys = torch.randn(num_pages, num_heads, head_dim)
    max_keys = torch.randn(num_pages, num_heads, head_dim)

    # Run your function
    scores = estimate_page_criticality(query, min_keys, max_keys)

    # Check output shape is correct
    assert scores.shape == (batch, num_heads, num_pages), f"Wrong shape: {scores.shape}"

    print("PASSED!")

def test_estimate_page_criticality_values():
    # Tiny example: 1 batch, 1 head, 1 page, 2 dimensions
    query = torch.tensor([[[2.0, -1.0]]])        # [1, 1, 2]
    min_keys = torch.tensor([[[1.0, -3.0]]])     # [1, 1, 2]
    max_keys = torch.tensor([[[4.0, 1.0]]])      # [1, 1, 2]

    scores = estimate_page_criticality(query, min_keys, max_keys)

    # dim 0: q=2 >= 0  → 2 * max(4) = 8
    # dim 1: q=-1 < 0  → -1 * min(-3) = 3
    # score = 8 + 3 = 11

    expected = torch.tensor([[[11.0]]])
    assert torch.allclose(scores, expected), f"Wrong values: {scores}"
    print("VALUES TEST PASSED!")

def test_estimate_page_criticality_inverted_bounds():
    """When min > max (e.g. uninitialized page), the score must be -inf or
    very negative, not +inf.  This was the main bug in the old max() formula."""
    query = torch.tensor([[[1.0, 1.0]]])
    min_keys = torch.tensor([[[float('inf'), float('inf')]]])
    max_keys = torch.tensor([[[float('-inf'), float('-inf')]]])

    scores = estimate_page_criticality(query, min_keys, max_keys)
    assert (scores <= 0).all(), f"Inverted bounds should give non-positive scores, got {scores}"
    print("INVERTED BOUNDS TEST PASSED!")

def test_top_k():
    # Define dimensions
    batch = 2
    num_heads = 4
    pages = 16
    k = 3

    scores = torch.randn(batch, num_heads, pages)
    topk = get_top_k(scores, k)

    assert topk.shape == (batch, num_heads, k)

    print("PASSED!")

def test_questMHA():
    pool = QuestMHATokenToKVPool(
        size=4,      # 4 total token slots
        page_size=2, # 2 tokens per page → 2 pages total
        dtype=torch.float32,
        head_num=1,
        head_dim=1,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False
    )

    assert not pool.page_valid[0][0], "Page 0 should start invalid"

    pool.k_buffer[0][0] = torch.tensor([[3.0]])
    pool.k_buffer[0][1] = torch.tensor([[7.0]])

    loc = torch.tensor([0, 1])
    pool.update_min_max(layer_id=0, loc=loc)

    assert pool.min_k_buffer[0][0] == 3.0, f"Wrong min: {pool.min_k_buffer[0][0]}"
    assert pool.max_k_buffer[0][0] == 7.0, f"Wrong max: {pool.max_k_buffer[0][0]}"
    assert pool.page_valid[0][0], "Page 0 should be valid after update"
    print("test_QuestMHA passed!")

def test_invalidate_pages():
    pool = QuestMHATokenToKVPool(
        size=4, page_size=2, dtype=torch.float32,
        head_num=1, head_dim=1, layer_num=1,
        device="cpu", enable_memory_saver=False
    )

    pool.k_buffer[0][0] = torch.tensor([[3.0]])
    pool.k_buffer[0][1] = torch.tensor([[7.0]])
    pool.update_min_max(layer_id=0, loc=torch.tensor([0, 1]))
    assert pool.page_valid[0][0], "Page should be valid after write"

    pool.invalidate_pages(torch.tensor([0]))
    assert not pool.page_valid[0][0], "Page should be invalid after invalidation"
    assert pool.min_k_buffer[0][0] == float('inf'), "min should reset to inf"
    assert pool.max_k_buffer[0][0] == float('-inf'), "max should reset to -inf"

    # Simulate reallocation: new data written to the same page
    pool.k_buffer[0][0] = torch.tensor([[10.0]])
    pool.k_buffer[0][1] = torch.tensor([[20.0]])
    pool.update_min_max(layer_id=0, loc=torch.tensor([0, 1]))

    assert pool.min_k_buffer[0][0] == 10.0, f"Wrong min after realloc: {pool.min_k_buffer[0][0]}"
    assert pool.max_k_buffer[0][0] == 20.0, f"Wrong max after realloc: {pool.max_k_buffer[0][0]}"
    assert pool.page_valid[0][0], "Page should be valid after reallocation write"
    print("test_invalidate_pages passed!")


if __name__ == "__main__":
    test_estimate_page_criticality()
    test_estimate_page_criticality_values()
    test_estimate_page_criticality_inverted_bounds()
    test_top_k()
    test_questMHA()
    test_invalidate_pages()
