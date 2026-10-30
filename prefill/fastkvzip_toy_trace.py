"""Trace the real FastKVzip cache path with a tiny, randomly initialized Qwen3.

Copy this file into FastKVzip/prefill and run it from that directory:

    python -B fastkvzip_toy_trace.py
    python -B fastkvzip_toy_trace.py --kv-type retain
    python -B fastkvzip_toy_trace.py --break-on-prune

This deliberately skips ModelKVzip.__init__: that constructor downloads a real
model/tokenizer and pretrained gates. Everything inside ModelKVzip.prefill,
including the repository's gate, cache eviction, custom CUDA cache update and
FlashAttention-varlen paths, stays unchanged.

Requires the project's Python/CUDA dependencies and an NVIDIA CUDA GPU.
No Hugging Face network access, model weights, or tokenizer is needed.
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from attention.gate import Weight
from attention.kvcache import EvictCache, RetainCache
from model.monkeypatch import replace_attn
from model.wrapper import ModelKVzip


def cache_mib(cache: EvictCache | RetainCache) -> float:
    """Actual K/V tensor payload in MiB; excludes allocator fragmentation."""
    nbytes = sum(
        k.numel() * k.element_size() + v.numel() * v.element_size()
        for k, v in zip(cache.key_cache, cache.value_cache)
    )
    return nbytes / (1024**2)


def first_head_lengths(cache: EvictCache | RetainCache) -> list[int]:
    if not cache.key_cache:
        return []
    if isinstance(cache, EvictCache):
        return cache.info["len_k"][0].tolist()
    # RetainCache stores *all* tokens, even after pruning the attention view.
    seq_len = cache.key_cache[0].shape[-2]
    return [seq_len] * cache.n_heads_kv


def install_debug_traces(cache_type: type, *, break_on_prune: bool) -> None:
    """Wrap methods just to print; calls all original project implementations."""
    original_update = cache_type.update
    original_prepare = cache_type.prepare
    original_prune_chunk = cache_type.prune_chunk

    def traced_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        n_new = int(key_states.shape[-2])
        result = original_update(
            self, key_states, value_states, layer_idx,
            {} if cache_kwargs is None else cache_kwargs,
        )
        if layer_idx == 0:
            print(
                f"  [KV.update] layer=0, new={n_new}, "
                f"head_lengths={first_head_lengths(self)}, "
                f"stored_K_shape={tuple(self.key_cache[0].shape)}"
            )
        return result

    def traced_prepare(self, query_states, key_states, value_states, layer_idx):
        result = original_prepare(
            self, query_states, key_states, value_states, layer_idx
        )
        if layer_idx == 0:
            q, k, v, info = result
            print(
                "  [KV.prepare -> FlashAttn] "
                f"Q={tuple(q.shape)}, K={tuple(k.shape)}, "
                f"V={tuple(v.shape)}; "
                f"cu_len_k={info['cu_len_k'].tolist()}"
            )
        return result

    def traced_prune_chunk(self, ratio, evict_range=tuple, level="pair"):
        before = first_head_lengths(self)
        before_mib = cache_mib(self)
        print(
            f"\n  [KV.prune_chunk] ratio={ratio:.3f}, "
            f"score_range={evict_range}, level={level}"
        )
        print(f"    BEFORE: lengths={before}, memory={before_mib:.4f} MiB")
        if break_on_prune:
            print(
                "    [PDB] Paused just before original prune_chunk(). "
                "Use 's' to step in, 'n' to step over, 'c' to continue."
            )
            breakpoint()
        result = original_prune_chunk(self, ratio, evict_range, level)
        after = first_head_lengths(self)
        after_mib = cache_mib(self)
        print(f"    AFTER:  lengths={after}, memory={after_mib:.4f} MiB")
        if isinstance(self, EvictCache):
            offsets = self.info["cu_len_k"][0].tolist()
            print(f"    PACKED HEAD OFFSETS: {offsets}")
            for layer_idx, key in enumerate(self.key_cache):
                physical_len = int(self.info["len_k"][layer_idx].sum().item())
                assert physical_len == key.shape[0], (
                    f"Layer {layer_idx}: metadata {physical_len} != "
                    f"stored length {key.shape[0]}"
                )
                assert self.key_cache[layer_idx].shape == self.value_cache[layer_idx].shape
        else:
            print("    RETAIN mode: stored KV deliberately stays uncompressed.")
        return result

    cache_type.update = traced_update
    cache_type.prepare = traced_prepare
    cache_type.prune_chunk = traced_prune_chunk


def build_random_model_and_gate(device: torch.device, kv_type: str) -> ModelKVzip:
    # The same *architecture classes* as the real pipeline; tiny random weights.
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=2048,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        attention_dropout=0.0,
    )
    replace_attn("qwen3")  # Official monkeypatch: custom gate + varlen attention.
    net = Qwen3ForCausalLM(config).to(device=device, dtype=torch.float16).eval()

    # Gate architecture is the real FastKVzip Weight module, but no trained
    # checkpoint is loaded: scores/quality have no semantic meaning here.
    group_size = config.num_attention_heads // config.num_key_value_heads
    gates = [
        Weight(
            index=i,
            input_dim=config.hidden_size,
            output_dim=16,
            nhead=config.num_key_value_heads,
            ngroup=group_size,
            dtype=torch.float16,
            sink=16,
        ).to(device).eval()
        for i in range(config.num_hidden_layers)
    ]

    # Normally __init__ downloads the 4B+ pretrained checkpoint and tokenizer.
    # We bypass ONLY that loading step and retain the real prefill() method.
    toy = object.__new__(ModelKVzip)
    toy.model = net
    toy.tokenizer = None  # integer tokens are generated below; no tokenizer.
    toy.name = "random-tiny-qwen3"
    toy.dtype = net.dtype
    toy.device = net.device
    toy.config = net.config
    toy.gates = gates
    toy.kv_type = kv_type
    toy.sys_prompt_ids = torch.tensor([[1]], device=device, dtype=torch.long)
    toy.postfix_ids = torch.empty((1, 0), device=device, dtype=torch.long)

    def show_first_gate(_module, inputs, output):
        print(
            f"  [Gate.forward] hidden={tuple(inputs[0].shape)}, "
            f"scores={tuple(output.shape)}, "
            f"range=[{output.min().item():.4f}, {output.max().item():.4f}]"
        )

    gates[0].register_forward_hook(show_first_gate)
    return toy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kv-type", choices=("evict", "retain"), default="evict")
    parser.add_argument("--tokens", type=int, default=127, help="random context tokens")
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--decode", type=int, default=3, help="test token-by-token appends")
    parser.add_argument(
        "--break-on-prune", action="store_true", help="enter pdb before every prune"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("A CUDA GPU is required: flash_attn_varlen_func uses CUDA.")
    if args.tokens < 1 or args.chunk < 1 or args.window < 0:
        parser.error("tokens and chunk must be positive; window must be >= 0.")
    if not (0 < args.ratio < 1):
        parser.error("Use 0 < --ratio < 1 to exercise actual eviction.")

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    device = torch.device("cuda:0")
    print(f"Torch={torch.__version__}, CUDA={torch.version.cuda}, GPU={torch.cuda.get_device_name(0)}")
    print("All weights and token IDs in this demo are RANDOM (no downloads).")

    cache_type = EvictCache if args.kv_type == "evict" else RetainCache
    install_debug_traces(cache_type, break_on_prune=args.break_on_prune)
    toy = build_random_model_and_gate(device, args.kv_type)

    context_ids = torch.randint(3, 255, (1, args.tokens), device=device)
    print(
        f"\n=== PREFILL: {args.tokens} context tokens; "
        f"chunk={args.chunk}, ratio={args.ratio}, window={args.window}, "
        f"cache={args.kv_type} ==="
    )
    torch.cuda.synchronize()
    begin = time.perf_counter()
    kv = toy.prefill(
        context_ids,
        prefill_chunk_size=args.chunk,
        window_size=args.window,
        chunk_ratio=args.ratio,
        level="pair",
    )
    torch.cuda.synchronize()
    print(
        f"\nPREFILL DONE: {time.perf_counter() - begin:.3f}s, "
        f"KV payload={cache_mib(kv):.4f} MiB; "
        f"layer0 head lengths={first_head_lengths(kv)}"
    )

    if args.decode:
        print(f"\n=== DECODE: append {args.decode} generated token(s) ===")
        token = torch.tensor([[5]], device=device, dtype=torch.long)
        for step in range(args.decode):
            with torch.inference_mode():
                outputs = toy.model(
                    input_ids=token, past_key_values=kv, use_cache=True
                )
                token = outputs.logits[:, -1:].argmax(dim=-1)
            print(
                f"  [Decode {step+1}] random token={token.item()}, "
                f"layer0 head lengths={first_head_lengths(kv)}, "
                f"KV payload={cache_mib(kv):.4f} MiB"
            )

    print("\nPASS: tiny Qwen3 -> random Gate -> real KV cache -> varlen FlashAttention")
    print("Do not interpret random tokens, attention quality, or timings as paper results.")


if __name__ == "__main__":
    main()
