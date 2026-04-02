import torch
import quest_attention

def test_quest_attention():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Define dimensions
    batch_size = 2
    num_heads = 4
    head_dim = 64
    num_tokens = 1024
    num_pages = 32
    page_size = 32  # num_tokens / num_pages
    num_selected = 3  # Select 3 pages per query
    
    # Create random tensors
    Q = torch.randn(batch_size, num_heads, head_dim, device=device)
    K_cache = torch.randn(num_tokens, num_heads, head_dim, device=device)
    V_cache = torch.randn(num_tokens, num_heads, head_dim, device=device)
    
    # Random page selection (values should be in range [0, num_pages-1])
    selected_pages = torch.randint(0, num_pages, (batch_size, num_heads, num_selected), 
                               dtype=torch.int32, device=device)  # Add dtype=torch.int32
    
    print(f"Q shape: {Q.shape}")
    print(f"K_cache shape: {K_cache.shape}")
    print(f"V_cache shape: {V_cache.shape}")
    print(f"selected_pages shape: {selected_pages.shape}")
    print(f"Selected pages: {selected_pages[0, 0]}")  # Show first batch, first head
    
    # Run QUEST attention
    print("\nRunning QUEST attention...")
    output = quest_attention.quest_attention_forward(Q, K_cache, V_cache, selected_pages, page_size)
    
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output device: {output.device}")
    
    # Basic sanity checks
    assert output.shape == (batch_size, num_heads, head_dim), f"Expected shape {(batch_size, num_heads, head_dim)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values!"
    assert not torch.isinf(output).any(), "Output contains Inf values!"
    
    print("\n✅ Basic checks passed!")
    
    # Print some statistics
    print(f"\nOutput statistics:")
    print(f"  Mean: {output.mean().item():.4f}")
    print(f"  Std: {output.std().item():.4f}")
    print(f"  Min: {output.min().item():.4f}")
    print(f"  Max: {output.max().item():.4f}")
    
    # Compare with naive attention on selected tokens (for validation)
    print("\n--- Validation against naive implementation ---")
    # Gather ALL selected tokens for this (batch, head)
    b, h = 0, 0
    all_k = []
    all_v = []
    for i in range(num_selected):
        page_idx = selected_pages[b, h, i].item()
        token_start = page_idx * page_size
        token_end = min(token_start + page_size, num_tokens)
        all_k.append(K_cache[token_start:token_end, h])
        all_v.append(V_cache[token_start:token_end, h])

    all_k = torch.cat(all_k, dim=0)  # [num_selected * page_size, head_dim]
    all_v = torch.cat(all_v, dim=0)

    q = Q[b, h]
    scores = torch.matmul(all_k, q)
    attn_weights = torch.softmax(scores, dim=0)
    expected = torch.matmul(attn_weights.unsqueeze(0), all_v).squeeze(0)

    print(f"Max difference: {torch.abs(expected - output[b, h]).max().item():.6f}")

if __name__ == "__main__":
    test_quest_attention()
