"""
First-order expert pruning baselines.

Computes REAP, EAN, MAN, and frequency-based pruning sets from the same HDF5
observations collected by the HOPE calibration step.
"""

import os
import json
import numpy as np
import h5py


def solve_baselines(
    obs_path, out_path, method, prune_frac=None, prune_num=None, task_id=None
):
    """
    Computes a first-order baseline pruning set.
    Scoring criteria:
        - freq: Number of tokens routed to expert k.
        - ean: Sum of ||f_k(x)|| over active tokens (Expert Activation Norm).
        - man: Mean of ||f_k(x)|| over active tokens.
        - reap: Mean of g_k(x) * ||f_k(x)|| over active tokens
            (Router-weighted Expert Activation Pruning).
    Experts with the lowest scores are pruned.
    Arguments:
        `obs_path`: path to HDF5 observations from calibration
        `out_path`: path to save the output JSON pruning set
        `method`: one of 'reap', 'ean', 'man', 'freq'
        `prune_frac`: fraction of experts to prune per layer (0 < frac < 1);
            conflicts with `prune_num`
        `prune_num`: number of experts to prune per layer (positive int);
            conflicts with `prune_frac`
        `task_id`: task ID in the HDF5; if None, uses the first available
    """
    assert method in ("reap", "ean", "man", "freq")

    if (prune_frac is None) == (prune_num is None):
        raise ValueError(
            "Exactly one of `prune_frac` or `prune_num` must be provided"
        )

    with h5py.File(obs_path, "r") as f:
        if task_id is None:
            task_id = list(f.keys())[0]
        task_group = f[task_id]

        layer_keys = sorted(
            [k for k in task_group.keys()
             if k.startswith("layer_") and "-" not in k],
            key=lambda k: int(k.split("_")[1]),
        )

        pruneset = {}
        for layer_key in layer_keys:
            layer_i = int(layer_key.split("_")[1])
            layer_group = task_group[layer_key]

            # Per-expert frequency (diagonal of coselect_counts)
            freq = np.diag(
                layer_group["coselect_counts"][:]
            ).astype(np.float64)

            if method == "freq":
                scores = freq
            elif method == "ean":
                scores = layer_group["norm_sums"][:]
            elif method == "man":
                scores = layer_group["norm_sums"][:] / np.maximum(freq, 1)
            elif method == "reap":
                scores = (
                    layer_group["gate_norm_sums"][:] / np.maximum(freq, 1)
                )

            num_experts = len(scores)
            if prune_frac is not None:
                assert 0 < prune_frac < 1
                num_prune = int(num_experts * prune_frac)
            else:
                num_prune = prune_num
            assert 0 < num_prune < num_experts

            indices = np.argsort(scores)
            pruneset[str(layer_i)] = sorted(indices[:num_prune].tolist())

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(pruneset, f, indent=2)

    total = sum(len(v) for v in pruneset.values())
    print(
        "Wrote pruning set to %s (%s). Pruned %d experts over %d layers."
        % (out_path, method, total, len(pruneset))
    )
