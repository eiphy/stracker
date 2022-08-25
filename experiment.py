from __future__ import annotations

import contextlib
import datetime
import json
import math
import os
import pickle
import shutil
import threading
import time
import zipfile
from pathlib import Path

import comet_ml
import matplotlib.pyplot as plt
import numpy as np
import yaml
from genericpath import exists
from tqdm import tqdm

from utils.configuration import Config
from utils.util import join_sublist, safe_del_dir, safe_load_obj, split_list_even

KEY = "u8zzPhVTZTFmGL3QZF4eTmjqE"
TRAINERS = ["fine_tune", "lwf", "eeil", "eeol", "bic", "lucir", "icarl"]
REHERSAL_TRAINERS = ["eeil", "eeol", "bic", "lucir"]
CEL_DIST_TRAINERS = ["eeil", "eeol", "bic", "lwf"]
TEXT_SUFFIX = [".py", ".yaml", ".txt", ".md", ".cc", ".h", ".cpp", ".json"]

TRINAER_COLOR = {
    "fine_tune": "k",
    "lwf": "b",
    "eeil": "g",
    "eeol": "r",
    "bic": "cyan",
    "lucir": "slategray",
    "icarl": "moccasin",
}

API = comet_ml.API(cache=False)


def get_online_config(exp):
    assets = exp.get_asset_list()
    configs = next(
        (
            yaml.safe_load(exp.get_asset(asset["assetId"]))
            for asset in assets
            if asset["fileName"] == "configs.yaml"
        ),
        {},
    )

    return Config(config_dict=configs)


def get_online_para(exp):
    paras = exp.get_parameters_summary()
    para_dict = {}
    for p in paras:
        name = p["name"]
        if not name.endswith("cfg"):
            value = p["valueCurrent"]
            with contextlib.suppress(Exception):
                value = float(value)
            para_dict[name] = value
    return para_dict


def get_online_metrics(exp, metric=None):
    meta = exp.get_metrics_summary(metric=metric)
    if not meta:
        return {metric: {"value": [], "max_step": 0, "min_step": float("inf")}}
    meta = meta if isinstance(meta, list) else [meta]
    metrics = {
        m["name"]: {"value": [], "max_step": 0, "min_step": float("inf")} for m in meta
    }
    for name, m in tqdm(metrics.items(), disable=True):
        raw = exp.get_metrics(metric=name)
        for m_r in raw:
            step = m_r["step"]
            assert step > 0
            if step > m["max_step"]:
                diff = step - m["max_step"]
                m["max_step"] = step
                m["value"].extend([float("nan")] * diff)
            if step < m["min_step"]:
                m["min_step"] = step
            v = m_r["metricValue"]
            if v is not None:
                m["value"][step - 1] = float(v)

    return metrics


def get_online_assets(exp, asset_name=None):
    if asset_name is None:
        return {asset["fileName"]: asset["assetId"] for asset in exp.get_asset_list()}
    else:
        return {
            asset["fileName"]: asset["assetId"]
            for asset in exp.get_asset_list()
            if asset["fileName"] == asset_name
        }


def download_asset(exp, assets, save_path, verbose=False):
    """
    :param assets: {filename: assetid}
    """
    root_path = Path(save_path)
    root_path.mkdir(exist_ok=True, parents=True)
    for filename, i in tqdm(assets.items(), disable=not verbose):
        asset = exp.get_asset(i)
        path = root_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(root_path / filename, "wb") as f:
            f.write(asset)


def archive_exp(key, verbose=True):
    api = API.get_experiment_by_key(key)
    if api is None:
        if verbose:
            print(f"Experiment {key} not exist in remote database.")
        return
    meta = api.get_metadata()
    ws = meta["workspaceName"]
    pro = meta["projectName"]
    if pro.endswith("-archive") and verbose:
        print(f"Experiment {key} has already been archived in {pro}.")
    archive_pro = f"{pro}-archive"
    if API.get_project(ws, archive_pro) is None:
        if verbose:
            print(f"Creating {archive_pro} project in {ws}.")
        API.create_project(ws, archive_pro, "Store archived exps from {pro}.")
    if verbose:
        print(f"Archiving {key} to {archive_pro}")
    API.move_experiments([key], ws, archive_pro)


def unarchive_exp(key):
    api = API.get_experiment_by_key(key)
    if api is None:
        print(f"Experiment {key} not exist in remote database.")
        return
    meta = api.get_metadata()
    ws = meta["workspaceName"]
    pro = meta["projectName"]
    if not pro.endswith("-archive"):
        print(f"Experiment {key} has not been archived in {pro}.")
    orig_pro = pro[:-8]
    if API.get_project(ws, orig_pro) is None:
        print(f"Creating {orig_pro} project in {ws}.")
        API.create_project(ws, "0archive", "Store archived exps from {pro}.")
    print(f"Unarchiving {key} to {orig_pro}")
    API.move_experiments([key], ws, orig_pro)


def _if_archive_op(
    config,
    finished,
    m,
    name,
    path,
    url,
    archive_unfinished,
    archive_no_configs,
    archive_incomplete_metric,
    metric,
    metric_num,
    metric_num_fun,
):
    hint = f"Exp {name} in {path if url is None else url} should be archived as"
    if archive_unfinished and finished == False:
        print(f"{hint} it is not finished.")
        return True
    if archive_no_configs and len(config) == 0:
        print(f"{hint} it is has no config.")
        return True
    if archive_incomplete_metric:
        if metric_num is None:
            metric_num = metric_num_fun(config)
        if m is None:
            print(f"{hint} the metric is None.")
            return True
        elif len(m) != metric_num:
            print(f"{hint} the length {len(m)} of {metric} != {metric_num}.")
            return True
        elif any(math.isnan(x) for x in m):
            print(f"{hint} there are nan in the {metric}: {m}.")
            return True
        elif any(math.isinf(x) for x in m):
            print(f"{hint} there are inf in the {metric}: {m}.")
            return True
    return False


def _if_archive_config(config, exp_config, name, path, url):
    hint = f"Exp {name} in {path if url is None else url} should be archived as"
    for p, v in config.items():
        if exp_config[p] != v:
            print(f"{hint} path {p}: {v}/{exp_config[p]}")
            return False
    return True


def archive_exps_online(exps, archive_ops, config=None, no_progress=False):
    cnt = 0
    orig_len = len(exps)
    output = []
    if archive_ops is None and config is None:
        return

    for exp in tqdm(exps, disable=no_progress):
        finished = bool(get_online_metrics(exp, "total_time"))
        exp_config = get_online_config(exp)
        try:
            m = get_online_metrics(exp, archive_ops["metric"])[archive_ops["metric"]][
                "value"
            ]
        except Exception:
            m = None

        if (
            config is None
            and _if_archive_op(
                exp_config, finished, m, exp.name, None, exp.url, **archive_ops
            )
        ) or (
            config is not None
            and _if_archive_config(
                config, get_online_config(exp), exp.name, None, exp.url
            )
        ):
            cnt += 1
            archive_exp(exp.key)
        else:
            output.append(exp)

    assert (
        len(output) == orig_len - cnt
    ), f"Current: {len(output)}, expected: {orig_len - cnt}, orig: {orig_len}, cnt: {cnt}."

    if not no_progress:
        print(f"Current: {len(output)}, archived: {cnt} from {orig_len}")
    return output


# Analysis helpers.
def filt_status(exps, num_threads=1, verbose=True, no_progress=False):
    def _func(exps, out, no_progress):
        tmp = [
            exp
            for exp in tqdm(exps, disable=no_progress)
            if not exp.get_metadata()["running"]
        ]
        for exp in tmp:
            out.append(exp)

    exp_ts = split_list_even(exps, num_threads)
    num_threads = min(num_threads, len(exp_ts))
    print(f"Filting runing exps with {num_threads} threads.")
    if num_threads == 1:
        assert len(exp_ts) == 1
        out = []
        _func(exp_ts[0], out, no_progress=no_progress)
    else:
        ts = []
        outs = [[] for _ in range(num_threads)]
        for i in range(num_threads):
            t = threading.Thread(
                target=_func,
                kwargs={"exps": exp_ts[i], "out": outs[i], "no_progress": no_progress},
            )
            t.start()
            ts.append(t)
        for t in ts:
            t.join()
        out = join_sublist(outs)
    if verbose:
        print(f"Filted {len(exps) - len(out)} running exps.")
    return out


def filt_online_archive(exps, archive_ops, num_threads=1, verbose=True):
    def _func(exps, out, archive_ops, no_progress):
        tmp = archive_exps_online(exps, archive_ops, no_progress=no_progress)
        for exp in tmp:
            out.append(exp)

    exp_ts = split_list_even(exps, num_threads)
    num_threads = min(num_threads, len(exp_ts))
    print(f"Archiving online exps with {num_threads} threads.")
    if num_threads == 1:
        assert len(exp_ts) == 1
        out = []
        _func(exp_ts[0], out, archive_ops, not verbose)
    else:
        ts = []
        outs = [[] for _ in range(num_threads)]
        for i in range(num_threads):
            no_progress_t = not verbose
            t = threading.Thread(
                target=_func,
                kwargs={
                    "exps": exp_ts[i],
                    "out": outs[i],
                    "archive_ops": archive_ops,
                    "no_progress": no_progress_t,
                },
            )
            t.start()
            ts.append(t)
        for t in ts:
            t.join()
        out = join_sublist(outs)
    if verbose:
        print(f"Archived {len(exps) - len(out)} exps.")
    return out


class LocalCometProjectDatabase:
    reserved_file_names = ["key_config.pkl", "pulled_exp_keys.pkl"]

    def __init__(self, path, pro):
        self.path = path
        self.pro = pro
        self._retrieve_key_config()
        self._retrieve_pulled_keys()

    #### Getter ####
    @property
    def key_config_mapping_file(self):
        return self.path / "key_config.pkl"

    @property
    def pulled_exp_keys_file(self):
        return self.path / "pulled_exp_keys.pkl"

    @property
    def keys(self):
        return list(self.key_config.keys())

    @property
    def exp_files_from_scanning(self):
        return [
            f
            for f in self.path.glob("./*.pkl")
            if f.is_file() and f.name not in self.reserved_file_names
        ]

    @property
    def keys_from_scanning(self):
        return [
            f.stem
            for f in self.path.glob("./*.pkl")
            if f.is_file() and f.name not in self.reserved_file_names
        ]

    #### DB Updation ####
    def _update_key_config(self, verbose=True):
        mapping_keys = set(self.key_config.keys())
        existed_keys = set(self.keys_from_scanning)
        keys_to_delete = mapping_keys - existed_keys
        keys_to_add = existed_keys - mapping_keys

        if len(keys_to_delete) > 0:
            print(
                f"Deleting {len(keys_to_delete)} entries from {self.key_config_mapping_file}."
            )
            for k in tqdm(keys_to_delete, disable=not verbose):
                del self.key_config[k]
        if len(keys_to_add) > 0:
            print(
                f"Adding {len(keys_to_add)} entries to {self.key_config_mapping_file}."
            )
            for k in tqdm(keys_to_add, disable=not verbose):
                exp = self.pro.get(path=self.path / f"{k}.pkl")
                if exp is not None:
                    self.key_config[k] = exp.configs

        assert set(self.key_config.keys()) == existed_keys
        if len(keys_to_add) or len(keys_to_delete):
            self.save_key_config()

    def update_key_config(self, strict=False, verbose=True):
        if strict:
            self.key_config = {}
        self._update_key_config(verbose=verbose)

    def update_pulled_keys(self, num_threads=16, strict=False):
        def _func(ks, pro, out):
            for k in tqdm(ks):
                exp = pro.get(k)
                if exp is not None and exp.if_metrics_assets_outfile_downloaded():
                    out.append(k)

        if strict:
            self.pulled_keys = []
        keys = [k for k in self.keys if k not in self.pulled_keys]
        key_ts = split_list_even(keys, num_threads)
        num_threads = min(num_threads, len(key_ts))
        print(
            f"Checking integrity of {len(keys)} experiments with {num_threads} threads."
        )
        if num_threads == 1:
            pulled_keys = []
            _func(key_ts[0], self.pro, pulled_keys)
        else:
            ts = []
            pulled_keys_ts = [[] for _ in range(num_threads)]
            for i in range(num_threads):
                t = threading.Thread(
                    target=_func,
                    kwargs={"ks": key_ts[i], "pro": self.pro, "out": pulled_keys_ts[i]},
                )
                t.start()
                ts.append(t)
            for t in ts:
                t.join()
            pulled_keys = join_sublist(pulled_keys_ts)
        self.pulled_keys += pulled_keys
        self.save_pulled_keys()

    #### DB Retrive ####
    def _retrieve_key_config(self):
        """db: {key: configs}"""
        if self.key_config_mapping_file.is_file():
            key_config = safe_load_obj(self.key_config_mapping_file)
            if key_config is None:
                key_config = {}
            else:
                key_config = {k: Config(config_dict=v) for k, v in key_config.items()}
        else:
            key_config = {}
        self.key_config = key_config
        self._update_key_config()

    def _retrieve_pulled_keys(self):
        if self.pulled_exp_keys_file.is_file():
            keys = safe_load_obj(self.pulled_exp_keys_file) or []
        else:
            keys = []
        self.pulled_keys = keys

    #### DB Save ####
    def save_key_config(self):
        key_config = {k: v.str_configs for k, v in self.key_config.items()}
        with open(self.key_config_mapping_file, "wb") as f:
            pickle.dump(key_config, f)

    def save_pulled_keys(self):
        with open(self.pulled_exp_keys_file, "wb") as f:
            pickle.dump(self.pulled_keys, f)

    #### DB Chek ####
    def _check(self):
        # check key_config mapping.
        assert len(self.exp_files_from_scanning) == len(self.key_config)
        for f in self.exp_files_from_scanning:
            assert f.stem in self.key_config


# TODO archive folder.
class LocalCometProject:
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.front_dir.mkdir(parents=True, exist_ok=True)
        self.db = LocalCometProjectDatabase(self.path, self)
        self.archive_db = LocalCometProjectDatabase(self.archive_dir, self)
        self._check()

    #### Getters ####
    def get(self, key=None, path=None) -> LocalCometExperiment:
        try:
            if key is not None:
                if key in self.db.keys:
                    return LocalCometExperiment(path=self.path / f"{key}.pkl")
                elif key in self.archive_db.keys:
                    return LocalCometExperiment(path=self.archive_dir / f"{key}.pkl")
                else:
                    print(f"Unknown key {key}")
                    return
            return LocalCometExperiment(path=path)
        except Exception:
            # pass
            # raise ValueError
            print(f"File of experiment {key} / {path} is broken. Deleting it.")
            path = path or self.path / f"{key}.pkl"
            key = key or path.stem
            path.unlink(missing_ok=True)
            (self.path / f"{key}.txt").unlink(missing_ok=True)
            safe_del_dir(self.path / key)
            self.update_dbs()

    def get_config(self, key):
        if key in self.db.keys:
            return self.db.key_config[key]
        elif key in self.archive_db.keys:
            return self.archive_db.key_config[key]
        else:
            print(f"Unknown key {key}")
            return Config(config_dict={})

    def query(
        self,
        query_config=None,
        exps=None,
        return_experiments=True,
        ignore_missing_path=True,
        archive=False,
    ):
        def _check(path, target, config, ignore_missing_path):
            try:
                v = config[path]
                if isinstance(target, str) and target.startswith("QUERY"):
                    return eval(target[6:])
                else:
                    return v == target
            except KeyError:
                return bool(ignore_missing_path)

        output = []
        if exps is None:
            key_config = self.archive_db.key_config if archive else self.db.key_config
        else:
            exps = exps if isinstance(list) else [exps]
            key_config = {exp.key: exp.configs for exp in exps}

        for k, config in key_config.items():
            success = all(
                _check(path, v, config, ignore_missing_path)
                for path, v in query_config.items()
            )
            if success:
                if exps is None:
                    output.append(self.get(k) if return_experiments else k)
                else:
                    exp = list(filter(lambda x: x.key == k, exps))
                    assert len(exp) == 1
                    exp = exp[0]
                    output.append(exp)

        return output

    def query_configs(self, config, query_configs):
        keys = self.query(config, return_experiments=False)
        out = {p: [] for p in query_configs}
        for p in query_configs:
            for k in keys:
                with contextlib.suppress(Exception):
                    out[p].append(self.db.key_config[k][p])
        return {p: sorted(list(set(v))) for p, v in out.items()}

    @property
    def name(self):
        return self.path.name

    @property
    def front_dir(self):
        return self.path.parent / f"{self.name}_front"

    @property
    def archive_dir(self):
        return self.path / "archive"

    @property
    def key_config(self):
        return self.db.key_config

    @property
    def keys(self):
        return self.db.keys

    @property
    def exp_file_from_scanning(self):
        return self.db.exp_files_from_scanning

    @property
    def keys_from_scanning(self):
        return self.db.keys_from_scanning

    @property
    def pulled_keys(self):
        return self.db.pulled_keys

    @property
    def archive_key_config(self):
        return self.archive_db.key_config

    @property
    def archive_keys(self):
        return self.archive_db.keys

    @property
    def archive_exp_file_from_scanning(self):
        return self.archive_db.exp_files_from_scanning

    @property
    def archive_keys_from_scanning(self):
        return self.archive_db.keys_from_scanning

    @property
    def archive_pulled_keys(self):
        return self.archive_db.pulled_keys

    #### Analysis ####
    def check_missing_seeds(
        self, keys, required_seeds=None, verbose=True, prefix_info=""
    ):
        if not required_seeds:
            return []
        cur_seeds = [self.get_config(k)["task;seed"] for k in keys]
        missing = sorted(list(set(required_seeds) - set(cur_seeds)))
        if len(missing) > 0 and verbose:
            print(f"{prefix_info} miss: {missing}")
        return missing

    def deal_with_repeat(
        self,
        output,
        repeat,
        archive_repeat=False,
        delete_repeat=False,
        preserve_latest=True,
        verbose=True,
        prefix_info="",
    ):
        for s, ks in repeat.items():
            if len(ks) > 0:
                if verbose:
                    for k in ks:
                        orig_config = self.get_config(output[s])
                        diff = orig_config.diff_configs(self.get_config(k))
                        print(
                            f"{prefix_info}: Seed {s} has following configuration differences:\n {json.dumps(diff, sort_keys=False, indent=2)}"
                        )
                if delete_repeat:
                    for k in ks:
                        exp = self.get(k)
                        if exp is not None:
                            exp.delete(archive_online=True)
                    self.update_dbs()
                elif archive_repeat:
                    for k in ks:
                        exp = self.get(k)
                        if exp is not None:
                            exp.archive(archive_online=True)
                    self.update_dbs()

    def check_repeat_seeds(
        self,
        keys,
        archive_repeat=False,
        delete_repeat=False,
        preserve_latest=True,
        verbose=True,
        prefix_info="",
    ):
        seeds = sorted([self.get_config(k)["task;seed"] for k in keys])
        appeared_seeds = {s: [] for s in seeds}
        for k in keys:
            appeared_seeds[self.get_config(k)["task;seed"]].append(k)

        output = {}
        repeat = {}
        for s, ks in appeared_seeds.items():
            if len(ks) > 1 and preserve_latest:
                times = [self.get(k).meta["endTimeMillis"] for k in ks]
                i = times.index(max(times))
            else:
                i = 0
            output[s] = ks[i]
        repeat = {
            s: [k for k in ks if k != output[s]] for s, ks in appeared_seeds.items()
        }
        self.deal_with_repeat(
            output,
            repeat,
            archive_repeat,
            delete_repeat,
            preserve_latest,
            verbose,
            prefix_info,
        )

        return output, repeat

    #### Setters ####

    #### Manage Experiments ####
    def pull(
        self,
        ws,
        pro,
        archive_ops=None,
        no_progress=False,
        num_threads=1,
    ):
        """Sync with remote database."""
        assert num_threads >= 1
        path = self.path

        def _download(apis, path, no_progress):
            for api in tqdm(apis, disable=no_progress):
                obj = LocalCometExperiment(api, path)
                obj.pull(True, True, False, True, api)
                obj.save()
            print(f"finished: {len(apis)}")

        print(f"Pull epxeriments from {ws}/{pro} to {path}.")

        exps = API.get_experiments(workspace=ws, project_name=pro)
        print(f"Remote database has {len(exps)} experiments.")
        saved_keys = list(self.keys) + list(self.archive_keys)
        tmp = [exp for exp in exps if exp.key not in saved_keys]
        print(f"Ignore {len(exps) - len(tmp)} saved experiments.")
        exps = tmp

        exps = filt_status(exps, num_threads * 2, no_progress=True)
        exps = filt_online_archive(exps, archive_ops, num_threads * 2, False)

        if len(exps) == 0:
            print("No experiments need pulling.")
            return

        exp_ts = split_list_even(exps, num_threads)
        num_threads = min(num_threads, len(exp_ts))
        print(
            f"Pulling {len(exps)} online experiments to {self.path} with {num_threads} threads!"
        )

        if num_threads == 1:
            _download(exp_ts[0], self.path, no_progress)
        else:
            ts = []
            for i in range(num_threads):
                no_progress_t = no_progress
                t = threading.Thread(
                    target=_download,
                    kwargs={
                        "apis": exp_ts[i],
                        "path": self.path,
                        "no_progress": no_progress_t,
                    },
                )
                t.start()
                ts.append(t)
            for t in ts:
                t.join()

        print(
            f"saved: {len(saved_keys)}, online: {len(exps)}, total: {len(saved_keys) + len(exps)}."
        )

        self.update_dbs()

    def archive_exps(self, exps=None, config=None, archive_online=True):
        exps = exps or self.query(config)
        exps = exps if isinstance(exps, list) else [exps]
        for exp in exps:
            exp.archive(archive_online=archive_online)
        self.update_dbs()

    def unarchive_exps(self, exps=None, config=None):
        exps = exps or self.query(config, archive=True)
        exps = exps if isinstance(exps, list) else [exps]
        for exp in exps:
            exp.unarchive()
        self.update_dbs()

    def bring_front(self, exps=None, keys=None):
        exps = exps or [self.get(k) for k in keys]
        for exp in exps:
            exp.copy(self.front_dir)

    def clear_front(self):
        for f in self.front_dir.glob("*"):
            if f.is_file():
                f.unlink()
            else:
                shutil.rmtree(f)

    #### Updaters ####
    def update_dbs(self, strict=False, verbose=False):
        if not hasattr(self, "db") or not hasattr(self, "archive_db"):
            print("Missing database file.")
            return
        self.db.update_key_config(strict, verbose=verbose)
        self.archive_db.update_key_config(strict, verbose=verbose)
        self._check()

    def update_pulled_keys(self, **kwargs):
        self.db.update_pulled_keys(**kwargs)
        self.archive_db.update_pulled_keys(**kwargs)

    def archive_by_ops(self, archive_ops, no_progress=False):
        cnt = 0
        if archive_ops is None:
            return

        for key in tqdm(self.db.key_config.keys(), disable=no_progress):
            exp = self.get(key)
            finished = exp.finished
            exp_config = exp.configs
            m = exp.get_metric_array(archive_ops["metric"])

            if _if_archive_op(
                exp_config, finished, m, exp.name, exp.path, None, **archive_ops
            ):
                cnt += 1
                exp.delete()
                self.update_dbs()

    #### Checks ####
    def _check(self):
        # check integrity
        set(self.keys).isdisjoint(set(self.archive_keys))


# TODO copy output file to current folder.
# TODO copy asset file to current folder.
# TODO put experiments inside a folder.
# TODO put metrics in seperate file.
# TODO delete missing assets.
class LocalCometExperiment:
    """
    Metrics and meta information is stored with handler in pkl format.
    Assets and terminal output are stored in a directory and text file respectively.
    """

    def __init__(self, api=None, path=None):
        if api is not None:
            self.key = api.key
            self.path = Path(path) / f"{self.key}.pkl"
            self.asset_dir.mkdir(parents=True, exist_ok=True)
            self._update_asset_path_mapping()
            self._download(api)
        else:
            assert path is not None
            path = Path(path)
            self.key = path.stem
            self.path = path.parent / f"{path.stem}.pkl"
            if path.suffix == ".zip":
                with zipfile.ZipFile(path, "r") as archive:
                    for f in archive.namelist():
                        archive.extract(f, self.path.parent)
            else:
                assert path.suffix == ".pkl"
            self._load_data_dict()
            self._update_asset_path_mapping()

    #### Setters ####
    def _download_output_file(self, api=None):
        api = api or API.get_experiment_by_key(self.key)
        output = api.get_output()
        with open(self.output_file, "w") as f:
            f.write(output)

    def _download_assets(self, api=None, only_text=True):
        api = api or API.get_experiment_by_key(self.key)
        if len(self.asset_to_path) > 0:
            asset_list = {
                a: i
                for a, i in self.asset_to_id.items()
                if a not in self.asset_to_path or not self.asset_to_path[a].is_file()
            }
        else:
            asset_list = self.asset_to_id
        if only_text:
            asset_list = {
                a: i for a, i in asset_list.items() if Path(a).suffix in TEXT_SUFFIX
            }
        if asset_list:
            download_asset(api, asset_list, self.asset_dir, verbose=False)
        self._update_asset_path_mapping()

    def _download(self, api):
        api = api or API.get_experiment_by_key(self.key)
        self.name = api.get_name()
        self.meta = api.get_metadata()
        self.configs = get_online_config(api)
        self.paras = get_online_para(api)
        self.tags = api.get_tags()

        self.metrics = get_online_metrics(api)
        metrics_summary = api.get_metrics_summary()
        self.metrics_summary = (
            metrics_summary if isinstance(metrics_summary, list) else [metrics_summary]
        )
        self.finished = "total_time" in self.metrics

        self._download_output_file()

        self.asset_to_id = get_online_assets(api)

        self.save()

    def _load_data_dict(self):
        with open(self.path, "rb") as f:
            data_dict = pickle.load(f)

        saving = False
        self.key = data_dict["key"]
        assert self.key == self.path.stem
        self.name = data_dict["name"]
        self.meta = data_dict["meta"]
        self.tags = data_dict["tags"]
        self.paras = data_dict["paras"]
        configs = data_dict["configs"]
        self.configs = (
            Config(config_dict=configs) if isinstance(configs, dict) else configs
        )
        self.metrics = data_dict["metrics"]
        if "metrics_summary" in data_dict:
            self.metrics_summary = data_dict["metrics_summary"]
        else:
            api = API.get_experiment_by_key(self.key)
            if api is None:
                self.metrics_summary = {}
            else:
                metrics_summary = api.get_metrics_summary()
                self.metrics_summary = (
                    metrics_summary
                    if isinstance(metrics_summary, list)
                    else [metrics_summary]
                )
                saving = True
        self.finished = data_dict["finished"]

        if not self.output_file.is_file():
            self._download_output_file()

        self.asset_to_id = data_dict.get("asset_to_id", {})

        if saving:
            self.save()

    #### Updaters ####
    def _update_asset_path_mapping(self):
        self.asset_to_path = {}
        for f in self.asset_dir.glob("**/*"):
            if f.is_file():
                name = str(f.relative_to(self.asset_dir))
                self.asset_to_path[name] = f.absolute()

    def pull(
        self,
        metrics=True,
        output_file=True,
        assets=False,
        only_text_assets=True,
        api=None,
        verbose=False,
        strict=False,
    ):
        self._update_asset_path_mapping()
        saving = False
        if metrics and not self.is_metrics_downloaded(strict=strict):
            api = api or API.get_experiment_by_key(self.key)
            if api is None:
                print(f"No existed remote experiment of {self.name} ({self.key})!")
                return
            self.metrics = get_online_metrics(api)
            saving = True

        if output_file and not self.is_output_file_downloaded():
            api = api or API.get_experiment_by_key(self.key)
            if api is None:
                print(f"No existed remote experiment of {self.name} ({self.key})!")
                return
            self._download_output_file()

        if assets and not self.is_assets_downloaded(strict=strict):
            api = api or API.get_experiment_by_key(self.key)
            if api is None:
                print(f"No existed remote experiment of {self.name} ({self.key})!")
                return
            self._download_assets(api, False)

        if (
            not assets
            and only_text_assets
            and not self.is_assets_downloaded(strict=strict, only_text=only_text_assets)
        ):
            api = api or API.get_experiment_by_key(self.key)
            if api is None:
                print(f"No existed remote experiment of {self.name} ({self.key})!")
                return
            self._download_assets(api, True)

        if saving:
            self.save()

    #### Getters ####
    @property
    def output_file(self):
        return self.path.parent / f"{self.path.stem}.txt"

    @property
    def asset_dir(self):
        return self.path.parent / f"{self.path.stem}"

    @property
    def status(self):
        return "archive" if self.path.parent.name == "archive" else "active"

    @property
    def start_date(self):
        return datetime.datetime.fromtimestamp(
            self.meta["startTimeMillis"] / 1000
        ).strftime("%c")

    @property
    def end_date(self):
        return datetime.datetime.fromtimestamp(
            self.meta["endTimeMillis"] / 1000
        ).strftime("%c")

    @property
    def duration(self):
        return self.meta["durationMillis"] / 1000

    @property
    def has_assets(self):
        return len(self.asset_to_path) > 0

    @property
    def data_dict(self):
        return {
            "key": self.key,
            "name": self.name,
            "meta": self.meta,
            "tags": self.tags,
            "paras": self.paras,
            "configs": self.configs,
            "metrics": self.metrics,
            "metrics_summary": self.metrics_summary,
            "finished": self.finished,
            "asset_to_id": self.asset_to_id,
        }

    def get_metric_array(self, name):
        if name not in self.metrics:
            return None
        else:
            return np.array(self.metrics[name]["value"], dtype=float)

    #### Magics & miscs ####
    def plot_metric(self, metric, label=None):
        label = label or self.name
        m = self.get_metric_array(metric).reshape(-1)
        steps = np.arange(1, m.size + 1)
        plt.plot(steps, m, label=label)

    def print_para(self):
        print(json.dumps(self.paras, indent=4))

    def print_config(self):
        self.configs.print_config()

    def move(self, dest):
        path = Path(dest) / self.path.name
        path_output_file = Path(dest) / self.output_file.name
        path_asset_dir = Path(dest) / self.asset_dir.name
        self.path.rename(path)
        self.output_file.rename(path_output_file)
        if self.asset_dir.is_dir():
            self.asset_dir.rename(path_asset_dir)
        self.path = path

    def copy(self, dest):
        path = Path(dest) / self.path.name
        path_output_file = Path(dest) / self.output_file.name
        path_asset_dir = Path(dest) / self.asset_dir.name
        shutil.copy(self.path, path)
        shutil.copy(self.output_file, path_output_file)
        if self.asset_dir.is_dir():
            shutil.copytree(self.asset_dir, path_asset_dir)

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.data_dict, f)

    def delete(self, archive_online=False, verbose=True):
        if archive_online:
            archive_exp(self.key, verbose=verbose)
        if verbose:
            print(f"Deleting experiments, output, and assets from {self.path}.")
        self.path.unlink(missing_ok=True)
        self.output_file.unlink(missing_ok=True)
        safe_del_dir(self.asset_dir)

    def archive(self, archive_online=False, verbose=True):
        if archive_online:
            archive_exp(self.key, verbose=verbose)
        if self.status == "active":
            if verbose:
                print(f"Archiving experiment, output, and assets from {self.path}.")
            dest_path = self.path.parent / "archive"
            self.move(dest_path)

    def unarchive(self, unarchive_online=False):
        if unarchive_online:
            unarchive_exp(self.key)
        if self.status == "archive":
            print(f"Unarchiving experiment, output, and assets from {self.path}.")
            dest_path = self.path.parent.parent
            self.move(dest_path)

    @property
    def comet_ws(self):
        return self.meta["workspaceName"]

    @property
    def comet_pro(self):
        return self.meta["projectName"]

    def __repr__(self):
        return f"Exp name: {self.name}; Status {'Completed' if self.finished else 'Aborted'}"

    #### Checkers ####
    def is_metrics_downloaded(self, api=None, strict=False):
        if strict:
            api = api or API.get_experiment_by_key(self.key)
            if api is None:
                return True
            metrics_summary = api.get_metrics_summary()
            self.metrics_summary = (
                metrics_summary
                if isinstance(metrics_summary, list)
                else [metrics_summary]
            )

            self.save()
        metrics_remote = {m["name"] for m in self.metrics_summary}
        metrics_local = set(self.metrics.keys())
        return metrics_remote == metrics_local

    def is_assets_downloaded(self, api=None, strict=False, only_text=False):
        if strict:
            api = api or API.get_experiment_by_key(self.key)
            if api is None:
                return True
            self.asset_to_id = get_online_assets(api)
            self._update_asset_path_mapping()
        if len(self.asset_to_path) > 0:
            asset_list = {
                a: i
                for a, i in self.asset_to_id.items()
                if a not in self.asset_to_path or not self.asset_to_path[a].is_file()
            }
            if only_text:
                asset_list = {
                    a: i for a, i in asset_list.items() if Path(a).suffix in TEXT_SUFFIX
                }
            return not asset_list
        elif len(self.asset_to_id) > 0:
            return False
        else:
            return True

    def is_output_file_downloaded(self):
        return self.output_file.is_file()
