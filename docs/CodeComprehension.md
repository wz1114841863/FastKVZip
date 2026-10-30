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


## 验证 Gate 可以流式计算
阶段 A1:验证 Gate 可以流式计算

使用官方训练好的 Gate,输入完全相同的 Hidden State,分别进行整批计算和逐 Token 计算.
比较输出的分数张量,确认两者在允许的浮点误差范围内一致.
这一阶段不需要运行完整模型,也不需要修改 KV Cache.

阶段 A2:验证裁剪策略能否映射到 Route-A 生命周期

保留 Fast KVzip 原本的 Gate 分数/预算与窗口策略,记录每个 Attention 时刻实际使用的 KV 集合.
然后检查当前 R6 能否通过一次性成熟决策表达同样的集合.如果不能,再确定需要的是策略适配,还是额外的旧记录驱逐能力.
