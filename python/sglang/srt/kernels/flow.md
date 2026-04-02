```
User generates token
    ↓
forward_decode() called
    ↓
Is this QuestMHATokenToKVPool?
    ↓ YES
estimate_page_criticality()  ← Python (quest_backend.py)
    ↓
get_top_k()                  ← Python (quest_backend.py)
    ↓
quest_attention_forward()    ← CUDA kernel (compiled from quest_kernel.cu)
    ↓
    └─ quest_attention_kernel<<<>>>() runs on GPU
           └─ Only loads selected pages
           └─ Computes attention
           └─ Returns output
    ↓
Return output to model
```