```
ModelKVzip.prefill()
    ↓
ModelKVzip.__call__()
    ↓
self.model.model(...)
    ↓
Transformer Attention
    ↓
KV Cache Update
    ↓
FlashAttention

```

添加测试脚本:
prefill/fastkvzip_toy_trace.py
