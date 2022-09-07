import json
from collections import defaultdict
from typing import List, Union

import numpy as np
from aim import Repo, Run
from aim.storage.context import Context
from naapc import NDict

from .utils import get_aim_hp_ndict


def get_repetitions(d: dict):
    """Check repetition in a dictionary.
    :param d: {hashable: value (hashable)}. Will check the repetition in values.
    """
    reverse_d = defaultdict(list)
    for k, v in d.items():
        reverse_d[v].append(k)
    return {v: k for v, k in reverse_d.items() if len(k) > 1}


def get_hparams_difference(key_hparams: dict, verbose=False, prefix=""):
    """Compare the hyperparameters of the first run with subsequent runs accordingly."""
    difference = []
    for i, (key, hparams) in enumerate(key_hparams.items()):
        if i == 0:
            key0 = key
            hparams0 = hparams
            continue

        diff = hparams0.compare_dict(hparams)
        difference.append(diff)
        if verbose:
            print(
                f"{prefix}0 ({key0}) - {i} ({key}):\n{json.dumps(diff, indent=2, sort_keys=False)}"
            )
    return difference


def get_missing_seeds(runs, seeds, seed_path="task;seed", verbose=True, prefix=""):
    missing = sorted({get_aim_hp_ndict(run, seed_path) for run in runs} - set(seeds))
    if missing and verbose:
        print(f"{prefix}miss: {missing}.")
    return missing


def get_repeated_missing_seeds(
    key_hparams: dict,
    seed_path: str,
    seeds: List[int] = None,
    check_repeated_hparams_difference=True,
    verbose=False,
    prefix="",
):
    """
    It gets repeated and missing seeds according to the input hyperparameters.

    :param key_hparams: the hyperparameters dictionary {key: hparams}
    :type key_hparams: dict
    :param seed_path: the path to the seed
    :type seed_path: str
    :param seeds: target seeds.
    :type seeds: List[int]
    :param check_repeated_hparams_difference: if True, check if the hyperparameters
    of the repeated runs are the same, defaults to True (optional)
    :param verbose: whether to print out the repetitions, defaults to False
    (optional)
    :param prefix: str
    :return a repetition dict of {seed: hash} and a list of [missing seeds].
    """
    key_seed = {k: hparams[seed_path] for k, hparams in key_hparams.items()}
    repetitions = get_repetitions(key_seed)
    if repetitions and verbose:
        print(f"{prefix}has following repetitions: {repetitions}.")

    # Check hyperparameters' difference of repetitions.
    if repetitions and check_repeated_hparams_difference:
        if verbose:
            print("Checking hparams difference.")
        for s, keys in repetitions.items():
            get_hparams_difference(
                {k: key_hparams[k] for k in keys}, True, f"seed {s} | "
            )

    if seeds is None:
        seeds = set()
    missing = sorted(
        set(seeds) - {hparams[seed_path] for hparams in key_hparams.values()}
    )
    if missing and verbose:
        print(f"{prefix}miss: {missing}")

    return repetitions, missing


def query(key_hparams: dict, target: dict, missing_path_action: str = "ignore") -> dict:
    """
    It takes a dictionary of hyperparameters, a dictionary of target hyperparameters, and a
    missing path action, and returns a list of keys that match the target hyperparameters.

    :param key_hparams: the hyperparameters dictionary {key: hparams}
    :type key_hparams: dict
    :param target: a dictionary of paths and values to match {path: v | query expression}. Query expression can be any python expressions with v stands for hparams[path] and hparams for hyperparameters.
    :type target: dict
    :param missing_path_action: What to do ([ignore | error]) if a path is missing from the hparams
    defaults to ignore
    :type missing_path_action: str (optional)
    :return: a dict of {key: hparams} the match the query.
    """
    assert missing_path_action in {
        "ignore",
        "error",
    }, f"Unexpected action {missing_path_action}."

    def _check(path, target, hparams, missing_path_action):
        try:
            v = hparams[path]
            if isinstance(target, str) and target.startswith("QUERY"):
                return eval(f"{target[6:]}")
            else:
                return v == target
        except KeyError:
            if missing_path_action == "ignore":
                return True
            elif missing_path_action == "error":
                raise KeyError

    output = {}
    for key, hparams in key_hparams.items():
        success = all(
            _check(path, v, hparams, missing_path_action) for path, v in target.items()
        )

        if success:
            output[key] = hparams
    return output


#### aim helpers ####
def get_metric_names_of_aim_run(run: Run) -> list:
    """
    > This function returns a list of metric names for a given run

    :param run: The run object that you want to get the metric names from
    :type run: Run
    """
    return [m[0] for m in run.iter_metrics_info()]


def get_metric_value_of_aim_run(
    run: Run, name: str, context: Union[dict, Context] = None
) -> np.array:
    """
    > Get a metric array from a run

    :param run: The run object that contains the metric you want to retrieve
    :type run: Run
    :param name: The name of the metric you want to retrieve
    :type name: str
    :param context: A dictionary of key-value pairs that describe a given run that
    you want to query
    :type context: Union[dict, Context]
    """
    if context is None or isinstance(context, dict):
        context = Context(context)
    assert isinstance(context, Context), f"Unexpected type: {type(context)}."
    return run.get_metric(name, context).dataframe()["value"].values.tolist()


def filt_aim_repeated_runs(
    runs: List[Run], repetitions: dict, option: str = "latest"
) -> List[Run]:
    """
    > This function takes a list of runs and a dictionary of repetitions and returns
    a list of runs with the repeated runs filtered out

    :param runs: a list of Run objects
    :type runs: List[Run]
    :param repetitions: a dictionary {seed : Run}
    :type repetitions: dict
    :param option: ["latest", "oldest"], defaults to latest
    :type option: str (optional)
    :return List[Run]
    """
    if not repetitions:
        return runs

    available_options = ["latest", "oldest"]
    assert (
        option in available_options
    ), f"Unexpected option: {option} / ({available_options})"

    def _get_end_epoch_millis(run: Run) -> float:
        if "end_time_millis" in get_metric_names_of_aim_run():
            return get_metric_value_of_aim_run(run, "end_time_millis")
        else:
            return run.end_time

    def _get_removing_hashes(runs: List[Run], option: str):
        if option == "latest":
            sorted_runs = sorted(
                [(run, _get_end_epoch_millis(run)) for run in runs], key=lambda x: x[1]
            )
            return [run[0].hash for run in sorted_runs[1:]]
        elif option == "oldest":
            sorted_runs = sorted(
                [(run, _get_end_epoch_millis(run)) for run in runs], key=lambda x: x[1]
            )
            return [run[0].hash for run in sorted_runs[:-1]]

    removing_hashes = []
    for runs in repetitions.items():
        removing_hashes.extend(_get_removing_hashes(runs, option))

    return [run for run in runs if run.hash not in removing_hashes]
