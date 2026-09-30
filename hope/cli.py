"""
CLI entry point for HOPE expert pruning.

Usage:
    hope calibrate --model-path <p> --prompts <f> --out-path <p>
    hope solve --obs-path <p> --prune-frac 0.25 --out-path <p>
    hope baselines --obs-path <p> --prune-frac 0.25 --method reap --out-path <p>
    hope prune --model-path <p> --pruneset-path <p> --out-path <p>
"""

import click


@click.group()
def cli():
    """
    HOPE: Higher-Order Pruning of Experts for MoE LLMs.
    """
    pass


@cli.command()
@click.option(
    "--model-path", required=True,
    type=click.Path(exists=True, file_okay=False),
    help="path to the HuggingFace MoE model"
)
@click.option(
    "--prompts", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="path to prompts file; can be .json (list of strings or token ID lists) or .txt (one string per line)"
)
@click.option(
    "--out-path", required=True,
    help="output HDF5 path for observations"
)
@click.option(
    "--limit", type=float, default=None,
    help="subsample prompts; can be given as a fraction if < 1, or count if >= 1"
)
@click.option(
    "--seed", type=int, default=20260423,
    help="random seed for subsampling prompts"
)
@click.option(
    "--stats-on-cpu", is_flag=True,
    help="if specified, keep accumulators on CPU instead of GPU"
)
@click.option(
    "--max-prompt-length", type=int, default=None,
    help="truncate prompts longer than this (in tokens)"
)
def calibrate(
    model_path, prompts, out_path, limit, seed,
    stats_on_cpu, max_prompt_length,
):
    """
    Collect expert interaction statistics (F-matrix).
    """
    import json

    if prompts.endswith(".json"):
        with open(prompts) as f:
            prompt_data = json.load(f)
    else:
        with open(prompts) as f:
            prompt_data = [
                line.strip() for line in f if line.strip()
            ]

    from hope.calibrate import calibrate as _calibrate
    _calibrate(
        model_path, prompt_data, out_path,
        limit=limit, seed=seed, stats_on_cpu=stats_on_cpu,
        max_prompt_length=max_prompt_length,
    )


@cli.command()
@click.option(
    "--obs-path", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="path to HDF5 observations from calibration"
)
@click.option(
    "--prune-frac", type=float, default=None,
    help="fraction of experts to prune per layer; conflicts with --prune-num"
)
@click.option(
    "--prune-num", type=int, default=None,
    help="number of experts to prune per layer; conflicts with --prune-frac"
)
@click.option(
    "--out-path", required=True,
    help="output JSON path for the prune-set"
)
@click.option(
    "--task-id", default=None,
    help="task ID in the HDF5; default: first available"
)
def solve(obs_path, prune_frac, prune_num, out_path, task_id):
    """
    Solve the HOPE QP for the optimal pruning set.
    """
    from hope.solve import solve as _solve
    _solve(
        obs_path, out_path, prune_frac=prune_frac, prune_num=prune_num,
        task_id=task_id,
    )


@cli.command()
@click.option(
    "--obs-path", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="path to HDF5 observations from calibration"
)
@click.option(
    "--prune-frac", type=float, default=None,
    help="fraction of experts to prune per layer; conflicts with --prune-num"
)
@click.option(
    "--prune-num", type=int, default=None,
    help="number of experts to prune per layer; conflicts with --prune-frac"
)
@click.option(
    "--out-path", required=True,
    help="output JSON path for the prune-set"
)
@click.option(
    "--method", required=True,
    type=click.Choice(["reap", "ean", "man", "freq"]),
    help="method of first-order scoring"
)
@click.option(
    "--task-id", default=None,
    help="task ID in the HDF5; default: first available"
)
def baselines(
    obs_path, prune_frac, prune_num, out_path, method, task_id,
):
    """
    Compute a first-order baseline pruning set.
    """
    from hope.baselines import solve_baselines
    solve_baselines(
        obs_path, out_path, method, prune_frac=prune_frac, prune_num=prune_num,
        task_id=task_id,
    )


@cli.command()
@click.option(
    "--model-path", required=True,
    type=click.Path(exists=True, file_okay=False),
    help="path to the original HuggingFace model"
)
@click.option(
    "--pruneset-path", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="path to JSON pruning set"
)
@click.option(
    "--out-path", required=True,
    help="path to save the pruned model"
)
def prune(model_path, pruneset_path, out_path):
    """
    Apply a pruning set and save the pruned model.
    """
    from hope.prune import prune_model
    prune_model(model_path, pruneset_path, out_path)
