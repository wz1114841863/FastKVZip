#!/usr/bin/env python3
"""Fast KVzip A1: official gated scorer, same hidden states, different batch organizations.

Run from FastKVzip/math after copying this file there:
    python a1_gate_equivalence.py --model Qwen/Qwen3-8B --layers 0 17 35

This does NOT evaluate KV eviction, latency, accuracy or Route-A lifecycle compatibility.
"""

import argparse
import json
import sys
from pathlib import Path

import torch


def score_in_chunks(gate, hidden_states, chunk_size):
    """Keep the token order unchanged; call the SAME official gate repeatedly."""
    parts = []
    for start in range(0, hidden_states.shape[1], chunk_size):
        stop = min(start + chunk_size, hidden_states.shape[1])
        parts.append(gate(hidden_states[:, start:stop, :]))
    return torch.cat(parts, dim=-1)


def score_in_irregular_chunks(gate, hidden_states):
    """Check that artificial chunk boundaries do not affect scores."""
    widths = [7, 1, 31, 128, 3, 17, 2, 68]
    parts = []
    start = 0
    j = 0
    while start < hidden_states.shape[1]:
        stop = min(start + widths[j % len(widths)], hidden_states.shape[1])
        parts.append(gate(hidden_states[:, start:stop, :]))
        start = stop
        j += 1
    return torch.cat(parts, dim=-1)


def tolerance_for(dtype):
    # These are *screening* tolerances, not a guarantee of algorithm-level equivalence.
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    if dtype == torch.float16:
        return 5e-3, 5e-3
    return 1e-5, 1e-5


def compare_scores(reference, candidate, atol, rtol):
    assert reference.shape == candidate.shape, (
        f"Shape mismatch: reference={tuple(reference.shape)} "
        f"candidate={tuple(candidate.shape)}"
    )
    a, b = reference.float(), candidate.float()
    diff = (a - b).abs()
    allowed = atol + rtol * a.abs()
    bad = (~torch.isfinite(a)) | (~torch.isfinite(b)) | (diff > allowed)
    return {
        "pass": int(bad.sum().item()) == 0,
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "bad_elements": int(bad.sum().item()),
        "total_elements": int(a.numel()),
    }


def run_layer(gate, layer_idx, hidden_states, atol, rtol):
    with torch.inference_mode():
        scores_full = gate(hidden_states)
        assert scores_full.ndim == 3
        assert scores_full.shape[0] == 1
        assert scores_full.shape[-1] == hidden_states.shape[1]
        expected_heads = gate.nhead
        assert scores_full.shape[1] == expected_heads

        modes = {
            "chunks_128": score_in_chunks(gate, hidden_states, 128),
            "chunks_16": score_in_chunks(gate, hidden_states, 16),
            "token_by_token": score_in_chunks(gate, hidden_states, 1),
            "irregular_chunks": score_in_irregular_chunks(gate, hidden_states),
        }

        comparisons = {
            mode: compare_scores(scores_full, candidate, atol, rtol)
            for mode, candidate in modes.items()
        }

        # Hold all tokens fixed except one, then check that other scores are unchanged.
        pivot = min(80, hidden_states.shape[1] - 1)
        modified_h = hidden_states.clone()
        modified_h[:, pivot, :] += torch.as_tensor(
            0.5, dtype=hidden_states.dtype, device=hidden_states.device
        )
        modified_scores = gate(modified_h)
        unaffected = torch.ones(hidden_states.shape[1], dtype=torch.bool,
                                device=hidden_states.device)
        unaffected[pivot] = False
        isolation = compare_scores(
            scores_full[..., unaffected],
            modified_scores[..., unaffected],
            atol,
            rtol,
        )
        pivot_change = (modified_scores[..., pivot].float() -
                        scores_full[..., pivot].float()).abs().max().item()

    return {
        "layer": layer_idx,
        "score_shape": list(scores_full.shape),
        "input_shape": list(hidden_states.shape),
        "score_min": float(scores_full.min().item()),
        "score_max": float(scores_full.max().item()),
        "score_std": float(scores_full.float().std().item()),
        "comparisons": comparisons,
        "token_isolation": {"changed_token": pivot,
                            "other_token_scores": isolation,
                            "pivot_max_score_change": float(pivot_change)},
        "pass": all(c["pass"] for c in comparisons.values()) and isolation["pass"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-8B",
                        help="Model ID used to choose the matching official Gate checkpoint")
    parser.add_argument("--seq-len", type=int, default=257)
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layer indices; default: first, middle, last")
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--out", default="a1_gate_report.json")
    args = parser.parse_args()
    if args.seq_len < 2:
        parser.error("--seq-len must be at least 2")
    if args.all_layers and args.layers is not None:
        parser.error("Use --all-layers or --layers, not both")

    try:
        # Official repository import. No AutoModelForCausalLM is loaded.
        from method.load_gate import load_gate
    except ImportError as exc:
        raise SystemExit(
            "Import failed. Copy this script into the repository's math/ directory "
            "and install the existing math/requirements.txt first.\n" + str(exc)
        ) from exc

    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is unavailable; choose --device cpu if desired")
        torch.backends.cuda.matmul.allow_tf32 = False
    gates = load_gate(args.model, device=args.device)
    if not gates:
        raise SystemExit("Official loader returned no gates")
    for g in gates:
        g.eval()
        for param in g.parameters():
            param.requires_grad_(False)

    layer_ids = (list(range(len(gates))) if args.all_layers
                 else args.layers if args.layers is not None
                 else sorted(set([0, len(gates) // 2, len(gates) - 1])))
    if any(l < 0 or l >= len(gates) for l in layer_ids):
        parser.error(f"--layers must be in [0, {len(gates)-1}]")

    gate_dtype = gates[0].q_proj.weight.dtype
    input_dim = gates[0].q_proj.in_features
    atol_default, rtol_default = tolerance_for(gate_dtype)
    atol = args.atol if args.atol is not None else atol_default
    rtol = args.rtol if args.rtol is not None else rtol_default

    print(f"Loaded {len(gates)} official gates; hidden_size={input_dim}; "
          f"KV_heads={gates[0].nhead}; dtype={gate_dtype}; "
          f"device={args.device}; sequence={args.seq_len}")
    print(f"Comparison tolerances: atol={atol}, rtol={rtol}")

    # Same *constructed* hidden state tensor is passed to every organization.
    # Reuse one tensor across layers: this tests call organization, not real model accuracy.
    hidden_states = torch.randn(
        (1, args.seq_len, input_dim),
        device=args.device, dtype=torch.float32,
    ).to(gate_dtype)

    results = []
    for layer_idx in layer_ids:
        result = run_layer(gates[layer_idx], layer_idx, hidden_states, atol, rtol)
        results.append(result)
        for mode, stats in result["comparisons"].items():
            print(f"layer={layer_idx:02d} mode={mode:17s} "
                  f"pass={stats['pass']} max_abs={stats['max_abs_error']:.6g} "
                  f"bad={stats['bad_elements']}/{stats['total_elements']}")
        print(f"layer={layer_idx:02d} token_isolation="
              f"{result['token_isolation']['other_token_scores']['pass']} "
              f"changed_token_delta={result['token_isolation']['pivot_max_score_change']:.6g}")

    report = {
        "purpose": "A1 score batching equivalence only (not eviction or Route-A lifecycle)",
        "model": args.model,
        "seed": args.seed,
        "device": args.device,
        "torch_version": torch.__version__,
        "dtype": str(gate_dtype),
        "sequence_length": args.seq_len,
        "hidden_size": input_dim,
        "kv_heads": gates[0].nhead,
        "gate_layers": len(gates),
        "atol": atol,
        "rtol": rtol,
        "results": results,
        "all_pass": all(x["pass"] for x in results),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report: {args.out}")
    print("A1_RESULT:", "PASS" if report["all_pass"] else "FAIL")
    return 0 if report["all_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
