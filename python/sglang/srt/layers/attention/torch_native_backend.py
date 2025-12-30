from __future__ import annotations

from typing import TYPE_CHECKING
import os
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


# Global state for saving attention weights
_ATTN_SAVE_COUNTER = 0  # Request counter
_ATTN_TOKEN_COUNTER = 0  # Token counter within current request (cumulative across steps)
_ATTN_LAYER_COUNTER = -1  # Current layer index (-1 means not initialized)
# Use path relative to this file's location
_DEFAULT_ATTN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../../../sglang_log/attention_weights")
_ATTN_SAVE_DIR = os.getenv("SGLANG_ATTN_SAVE_DIR", _DEFAULT_ATTN_DIR)
_ATTN_SAVE_ENABLED = os.getenv("SGLANG_SAVE_ATTN")
# Layer and head ranges to save (format: "start-end" or "all", e.g., "0-5" or "10" or "all")
_ATTN_SAVE_LAYERS = os.getenv("SGLANG_SAVE_ATTN_LAYERS", "0,1")  # Default: only layer 0,1
_ATTN_SAVE_HEADS = os.getenv("SGLANG_SAVE_ATTN_HEADS", "0,1")    # Default: only head 0,1
_ATTN_CURRENT_FILES = {}  # Dict mapping (layer, head) -> (file, writer, csv_path)
_IS_PREFILL_PHASE = False  # Track if we're in prefill phase
_ATTN_MAX_KEYS = {}  # Dict mapping (layer, head) -> max_keys


def _parse_range(range_str):
    """Parse range string like '0-5', '10', '2,3,5', or 'all' into a set of indices or None for all."""
    if not range_str or range_str.lower() == 'all':
        return None  # None means all

    result = set()
    # Split by comma to handle multiple ranges/values
    for part in range_str.split(','):
        part = part.strip()
        if '-' in part:
            # Handle range like "0-5"
            start, end = part.split('-')
            result.update(range(int(start), int(end) + 1))
        else:
            # Handle single value like "10"
            result.add(int(part))

    return result


def _should_save_layer_head(layer_id, head_id):
    """Check if we should save this layer/head combination."""
    layer_set = _parse_range(_ATTN_SAVE_LAYERS)
    head_set = _parse_range(_ATTN_SAVE_HEADS)

    layer_match = layer_set is None or layer_id in layer_set
    head_match = head_set is None or head_id in head_set

    return layer_match and head_match


def manual_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, is_prefill=False, layer_id=None, input_token_ids=None
):
    """
    Manual implementation of scaled dot product attention that saves weights to CSV.

    This replaces torch.nn.functional.scaled_dot_product_attention to expose
    attention weights for analysis.

    Args:
        query: [batch, num_q_heads, seq_len_q, head_dim]
        key:   [batch, num_kv_heads, seq_len_k, head_dim]
        value: [batch, num_kv_heads, seq_len_v, head_dim]
        enable_gqa: If True, handles Grouped Query Attention where num_q_heads > num_kv_heads
        is_prefill: If True, this is a prefill phase (start of new request)
        layer_id: Layer index (for saving to separate files)
        input_token_ids: Actual token IDs being processed (optional, for CSV labels)
    """
    global _ATTN_SAVE_COUNTER, _ATTN_TOKEN_COUNTER, _ATTN_LAYER_COUNTER, _ATTN_CURRENT_FILES, _IS_PREFILL_PHASE, _ATTN_MAX_KEYS, _ATTN_ROWS_DATA

    batch_size = query.size(0)
    num_q_heads = query.size(1)
    num_kv_heads = key.size(1)
    seq_len_q = query.size(2)
    seq_len_k = key.size(2)
    head_dim = query.size(3)

    # Handle GQA: replicate key/value heads to match query heads
    if enable_gqa and num_q_heads != num_kv_heads:
        assert num_q_heads % num_kv_heads == 0, f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        num_groups = num_q_heads // num_kv_heads
        # Repeat each KV head num_groups times
        # [batch, num_kv_heads, seq_len, head_dim] -> [batch, num_q_heads, seq_len, head_dim]
        key = key.repeat_interleave(num_groups, dim=1)
        value = value.repeat_interleave(num_groups, dim=1)

    L, S = seq_len_q, seq_len_k
    scale_factor = 1 / (head_dim ** 0.5) if scale is None else scale

    # Compute attention scores: Q @ K^T
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    # [batch, num_heads, seq_len_q, head_dim] @ [batch, num_heads, head_dim, seq_len_k]
    # -> [batch, num_heads, seq_len_q, seq_len_k]
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    # Save attention weights to CSV (only if enabled via environment variable)
    if _ATTN_SAVE_ENABLED and layer_id is not None:
        try:
            # If this is a new prefill, start a new request (close all old files)
            if is_prefill and not _IS_PREFILL_PHASE:
                # Close all previous files
                for (l_id, h_id), (f, w, path) in _ATTN_CURRENT_FILES.items():
                    f.close()
                _ATTN_CURRENT_FILES.clear()
                _ATTN_MAX_KEYS.clear()

                _IS_PREFILL_PHASE = True
                _ATTN_TOKEN_COUNTER = 0
                _ATTN_SAVE_COUNTER += 1
                print(f"✅ Started new request [{_ATTN_SAVE_COUNTER-1}]", flush=True)

                # 🐛 DEBUG: Log INSIDE manual_scaled_dot_product_attention
                if layer_id == 0 and input_token_ids is not None:
                    print(f"🐛 DEBUG INSIDE manual_sdpa (layer {layer_id}):", flush=True)
                    print(f"   input_token_ids length: {len(input_token_ids)}", flush=True)
                    print(f"   First 10 token IDs: {input_token_ids[:min(10, len(input_token_ids))]}", flush=True)
                    if len(input_token_ids) > 0:
                        print(f"   First token ID: {input_token_ids[0]} (expected 151644 for <|im_start|>)", flush=True)

            # If we're in decode phase (not prefill), update the phase flag
            if not is_prefill and _IS_PREFILL_PHASE:
                _IS_PREFILL_PHASE = False
                print(f"   Switched to decode phase for request [{_ATTN_SAVE_COUNTER-1}]", flush=True)

            # Save attention weights for each head that matches the filter
            for head_idx in range(num_q_heads):
                if not _should_save_layer_head(layer_id, head_idx):
                    continue  # Skip this layer/head combination

                file_key = (layer_id, head_idx)

                # Create file if it doesn't exist for this layer/head
                if file_key not in _ATTN_CURRENT_FILES:
                    Path(_ATTN_SAVE_DIR).mkdir(parents=True, exist_ok=True)
                    csv_path = Path(_ATTN_SAVE_DIR) / f"attn_weights_request_{_ATTN_SAVE_COUNTER-1:06d}_layer_{layer_id:02d}_head_{head_idx:02d}.csv"
                    f = open(csv_path, 'w', newline='')
                    writer = csv.writer(f)

                    # Write simple header (just query_token_id)
                    # Rows will have variable length (valid CSV format)
                    writer.writerow(["query_token_id"])

                    # Initialize tracking
                    weights_np = attn_weight[0, head_idx].detach().cpu().float().numpy()
                    num_keys = weights_np.shape[1]

                    _ATTN_CURRENT_FILES[file_key] = (f, writer, csv_path)
                    _ATTN_MAX_KEYS[file_key] = num_keys
                    print(f"   Created CSV for layer {layer_id}, head {head_idx}: {csv_path.name}", flush=True)

                # Get file, writer, and path for this layer/head
                f, writer, csv_path = _ATTN_CURRENT_FILES[file_key]
                max_keys = _ATTN_MAX_KEYS[file_key]

                # Extract attention weights for this head
                weights_np = attn_weight[0, head_idx].detach().cpu().float().numpy()
                num_queries = weights_np.shape[0]
                num_keys = weights_np.shape[1]

                # Track max keys seen (just for logging)
                if num_keys > max_keys:
                    print(f"   ⚠️  Layer {layer_id} Head {head_idx}: Sequence grew from {max_keys} to {num_keys} keys", flush=True)
                    _ATTN_MAX_KEYS[file_key] = num_keys

                # Append new rows only (variable-length rows are valid CSV)
                for row_idx, row in enumerate(weights_np):
                    if input_token_ids is not None and row_idx < len(input_token_ids):
                        # Use actual vocabulary token ID
                        token_id_value = input_token_ids[row_idx]
                    else:
                        # Fall back to sequential counter
                        token_id_value = _ATTN_TOKEN_COUNTER + row_idx

                    row_data = row.tolist()
                    # Write row with token ID + all attention weights
                    # CSV allows variable-length rows - early rows will have fewer columns
                    writer.writerow([token_id_value] + row_data)

                f.flush()  # Ensure data is written

            # Update token counter (only once per call, not per head)
            if num_q_heads > 0:
                weights_np = attn_weight[0, 0].detach().cpu().float().numpy()
                _ATTN_TOKEN_COUNTER += weights_np.shape[0]

        except Exception as e:
            print(f"⚠️  Failed to save attention weights: {e}", flush=True)

    if dropout_p > 0.0:
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    # Apply attention to values
    # [batch, num_heads, seq_len_q, seq_len_k] @ [batch, num_heads, seq_len_k, head_dim]
    # -> [batch, num_heads, seq_len_q, head_dim]
    return attn_weight @ value


class TorchNativeAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device
        print("=" * 80, flush=True)
        print("🔍 VERIFICATION: Using TorchNativeAttnBackend for attention!", flush=True)
        print("=" * 80, flush=True)

    def __del__(self):
        """Close CSV files when backend is destroyed."""
        global _ATTN_CURRENT_FILES
        for (l_id, h_id), (f, w, path) in _ATTN_CURRENT_FILES.items():
            f.close()
        _ATTN_CURRENT_FILES.clear()

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        pass

    def _run_sdpa_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
        layer_id=None,
        input_token_ids=None,
    ):
        """Run the extend forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            extend_prefix_lens: [num_seqs]
            extend_seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.

            extend_seq_len_q = extend_seq_lens[seq_idx]
            prefill_seq_len_q = extend_prefix_lens[seq_idx]

            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + extend_seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]
            per_req_query_redudant = torch.empty(
                (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
                dtype=per_req_query.dtype,
                device=per_req_query.device,
            )

            per_req_query_redudant[:, prefill_seq_len_q:, :] = per_req_query

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)
            # Original PyTorch SDPA (for verification)
            per_req_out_redudant_original = (
                scaled_dot_product_attention(
                    per_req_query_redudant.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )

            # Extract token IDs for this request if available
            per_req_token_ids = None
            if input_token_ids is not None:
                per_req_token_ids = input_token_ids[start_q:end_q].cpu().numpy().tolist()

                # 🐛 DEBUG: Log BEFORE calling manual_scaled_dot_product_attention
                if layer_id == 0 and seq_idx == 0 and start_q == 0:
                    print(f"🐛 DEBUG BEFORE manual_sdpa (layer {layer_id}, seq {seq_idx}):", flush=True)
                    print(f"   Total input_token_ids length: {len(input_token_ids)}", flush=True)
                    print(f"   First 10 token IDs: {input_token_ids[:10].cpu().numpy().tolist()}", flush=True)
                    print(f"   per_req_token_ids length: {len(per_req_token_ids)}", flush=True)
                    print(f"   First 10 per_req IDs: {per_req_token_ids[:10]}", flush=True)
                    print(f"   start_q={start_q}, end_q={end_q}", flush=True)

            # Manual SDPA (saves attention weights)
            per_req_out_redudant_manual = (
                manual_scaled_dot_product_attention(
                    per_req_query_redudant.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    scale=scaling,
                    is_causal=causal,
                    enable_gqa=enable_gqa,
                    is_prefill=True,  # This is the extend/prefill phase
                    layer_id=layer_id,
                    input_token_ids=per_req_token_ids,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )

            # Verify they match
            # 🐛 DEBUG: Log shapes of attention outputs for all sequences
            if layer_id == 0:
                print(f"🐛 DEBUG Attention output shapes (layer {layer_id}, seq {seq_idx}):", flush=True)
                print(f"   per_req_out_redudant_original shape: {per_req_out_redudant_original.shape}", flush=True)
                print(f"   per_req_out_redudant_manual shape: {per_req_out_redudant_manual.shape}", flush=True)
                print(f"   per_req_query_redudant shape: {per_req_query_redudant.shape}", flush=True)
                print(f"   per_req_key shape: {per_req_key.shape}", flush=True)
                print(f"   per_req_value shape: {per_req_value.shape}", flush=True)
                print(f"   prefill_seq_len_q: {prefill_seq_len_q.item()}", flush=True)
                print(f"   extend_seq_len_q: {extend_seq_len_q.item()}", flush=True)
                print(f"   seq_len_kv: {seq_len_kv.item()}", flush=True)

            diff = per_req_out_redudant_original - per_req_out_redudant_manual
            max_diff = diff.abs().max().item()
            mse = (diff ** 2).mean().item()

            if max_diff > 1e-5 or mse > 1e-10:
                print(f"⚠️  WARNING: Extend attention outputs differ! Max diff: {max_diff:.2e}, MSE: {mse:.2e}", flush=True)

            # Use manual output when saving attention weights (to match the saved weights),
            # otherwise use original output for correctness
            if _ATTN_SAVE_ENABLED:
                per_req_out_redudant = per_req_out_redudant_manual
            else:
                per_req_out_redudant = per_req_out_redudant_original
            output[start_q:end_q, :, :] = per_req_out_redudant[prefill_seq_len_q:, :, :]
            start_q, start_kv = end_q, end_kv
        return output

    def _run_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
        layer_id=None,
        input_token_ids=None,
    ):
        """Run the decode forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.

            seq_len_q = 1
            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            # Original PyTorch SDPA (for verification)
            per_req_out_original = (
                scaled_dot_product_attention(
                    per_req_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                ) 
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )

            # Extract token ID for this request if available (decode = single token)
            per_req_token_id = None
            if input_token_ids is not None and seq_idx < len(input_token_ids):
                per_req_token_id = [input_token_ids[seq_idx].item()]

            # Manual SDPA (saves attention weights)
            per_req_out_manual = (
                manual_scaled_dot_product_attention(
                    per_req_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    scale=scaling,
                    is_causal=causal,
                    enable_gqa=enable_gqa,
                    is_prefill=False,  # This is the decode phase
                    layer_id=layer_id,
                    input_token_ids=per_req_token_id,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )

            # Verify they match
            diff = per_req_out_original - per_req_out_manual
            max_diff = diff.abs().max().item()
            mse = (diff ** 2).mean().item()

            if max_diff > 1e-5 or mse > 1e-10:
                print(f"⚠️  WARNING: Decode attention outputs differ! Max diff: {max_diff:.2e}, MSE: {mse:.2e}", flush=True)

            # Use manual output when saving attention weights (to match the saved weights),
            # otherwise use original output for correctness
            if _ATTN_SAVE_ENABLED:
                per_req_out = per_req_out_manual
            else:
                per_req_out = per_req_out_original
            output[start_q:end_q, :, :] = per_req_out
            start_q, start_kv = end_q, end_kv

        return output

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # print(f"🔍 TorchNativeAttnBackend.forward_extend called! layer_id={layer.layer_id}", flush=True)
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

        self._run_sdpa_forward_extend(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_prefix_lens,
            forward_batch.extend_seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=causal,
            layer_id=layer.layer_id,
            input_token_ids=forward_batch.input_ids,
        )
        return o

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # print(f"🔍 TorchNativeAttnBackend.forward_decode called! layer_id={layer.layer_id}", flush=True)
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        self._run_sdpa_forward_decode(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=False,
            layer_id=layer.layer_id,
            input_token_ids=forward_batch.input_ids,
        )

        return o

    def support_triton(self):
        return False
