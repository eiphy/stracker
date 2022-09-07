from __future__ import annotations

from collections import defaultdict
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from aim import Run
from naapc import NConfig
from scipy.stats import spearmanr

from stracker.ana import get_metric_value_of_aim_run


def get_hp_ndict(run, path):
    return NConfig(run["hparams"])[path]


def check_seeds(runs, seeds, seed_path, check_repeat_ops, info):
    seed_path = seed_path.replace(";", ".")
    seed_run_map = defaultdict(list)
    for run in runs:
        s = get_hp_ndict(run, seed_path)
        seed_run_map[s].append(run)

def get_aim_metrics(runs: List[Run]):
    metrics = np.concatenate(
        [np.array(get_metric_value_of_aim_run(run, "test_batch_acc")).reshape(1, -1) for run in runs], axis=0
    )
    mean_metrics = metrics.mean(axis=0)

def get_metrics(exps: List[LocalCometExperiment], seeds, check_seeds=True, B=10):
    if check_seeds:
        _seeds = {exp.configs["task;seed"] for exp in exps}
        assert _seeds == set(seeds), f"{_seeds} / {set(seeds)}"
    metrics = np.concatenate(
        [exp.get_metric_array("test_batch_acc").reshape(1, -1) for exp in exps], axis=0
    )
    if check_seeds:
        assert metrics.shape[0] == len(
            seeds
        ), f"Has {metrics.shape[0]} experiments. Expected {len(seeds)} experiments."
    mean_metrics = metrics.mean(axis=0)
    mean_duration = sum(exp.duration for exp in exps) / len(exps)
    # effective_times = np.concatenate(
    #     [
    #         exp.get_metric_array("train_epoch_effective_time").reshape(1, -1)
    #         for exp in exps
    #     ],
    #     axis=0,
    # )
    # mean_effective_times = effective_times.mean(axis=0)
    # mean_effective_times = [
    #     sum(ts) for ts in split_list(mean_effective_times.tolist(), B)
    # ]
    # batch_times = np.concatenate(
    #     [exp.get_metric_array("batch_time").reshape(1, -1) for exp in exps],
    #     axis=0,
    # )
    # mean_batch_times = batch_times.mean(axis=0)
    return {
        "batch": mean_metrics.tolist(),
        "inc": mean_metrics.mean().item(),
        "fin": mean_metrics[-1].item(),
        "spearmanc": spearmanr(
            mean_metrics, list(range(mean_metrics.size))
        ).correlation.item(),
        "spearmanp": spearmanr(
            mean_metrics, list(range(mean_metrics.size))
        ).pvalue.item(),
        "duration": mean_duration,
        "average_batch_time": mean_duration / mean_metrics.size,
        # "effective_time": mean_effective_times,
        # "total_effective_time": sum(mean_effective_times),
        # "average_batch_effective_time": sum(mean_effective_times) / B,
        # "batch_time": mean_batch_times.tolist(),
    }


def plot_metrics(metrics: dict, title: str, xlabel: str, ylabel: str):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for k, ms in metrics.items():
        step = list(range(1, len(ms) + 1))
        plt.plot(step, ms, label=k)
    plt.legend()
    plt.grid()


def get_strong_baseline(metrics: dict, baselines: List[str]):
    for tr in baselines:
        assert tr in metrics, f"No info of {tr} in {metrics.keys()}"
    ms = {tr: m for tr, m in metrics.items() if tr in baselines}
    trs = list(ms.keys())
    fins = [m["fin"] for m in ms.values()]
    incs = [m["inc"] for m in ms.values()]
    fin_max = max(fins)
    fin_tr = trs[fins.index(fin_max)]
    inc_max = max(incs)
    inc_tr = trs[incs.index(inc_max)]

    return fin_max, fin_tr, inc_max, inc_tr


def deal_seeds(
    pro: LocalCometProject,
    config,
    prefix_info,
    seeds,
    check_repeat_ops,
    verbose=False,
    filt_uneeded_seeds=True,
):
    keys = pro.query(config, return_experiments=False)
    keys, repeat = pro.check_repeat_seeds(
        keys, prefix_info=prefix_info, verbose=verbose, **check_repeat_ops
    )
    keys = {k for s, k in keys.items() if s in seeds}
    missing = pro.check_missing_seeds(keys, seeds, verbose, prefix_info)
    return keys, repeat, missing
