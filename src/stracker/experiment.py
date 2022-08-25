import datetime
import shutil
import zipfile
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from naapc import NConfig

from .metric import Metric

# from pyinstrument import Profiler

class SExperiment:
    empty_meta = {"key": "", "name": "", "tags": [], "hyperparameters": {}, "duration_millis": 0, "start_time_millis": 0, "end_time_millis": 0}
    def __init__(self, path, ignore_not_exist=True):
        path = Path(path)
        self.is_compressed = False
        self.is_empty = False
        if path.is_dir():
            self.path = path
        elif (path.parent / f"{path.stem}.zip").is_file():
            self.path = path.parent / f"{path.stem}.zip"
            self.is_compressed = True
        elif ignore_not_exist:
            self.meta = deepcopy(self.empty_meta)
            self.is_empty = True
        else:
            raise RuntimeError(f"{path} not exist!")

        self._update()

    ### getters ###
    def plot_metric(self, name, label=None):
        label = label or self.name
        metric = self.get_metric(name)
        if metric is not None:
            m = metric.array
            m = np.nan_to_num(m)
            steps = metric.step_array
            plt.plot(steps, m, label=label)

    def get_metric(self, name, default=None):
        return self._metrics.get(name, default)

    ### file ###
    def move(self, dest):
        dest = Path(dest)
        self.path.rename(dest / self.key)
        self.path = dest / self.key

    def copy(self, dest):
        dest = Path(dest)
        shutil.copytree(self.path, dest / self.key)

    def delete(self):
        if self.path.is_dir():
            shutil.rmtree(self.path)
        else:
            print("Experiment not exist.")

    def save(self):
        meta = {
            "key": str(self.key),
            "name": str(self.name),
            "tags": [str(t) for t in self.tags],
            "hyperparameters": self.hyperparameters.flatten_dict,
            "duration_millis": self.duration_millis,
            "start_time_millis": self.meta["start_time_millis"],
            "end_time_millis": self.meta["end_time_millis"],
        }
        with open(self.path / "meta.yaml", "w") as f:
            yaml.dump(meta, f, sort_keys=False)

    ### updaters ###
    def _update_metric_mapping(self):
        self._metrics = {f.stem: Metric(f) for f in self.metric_dir.glob("*.txt")}

    def _update_asset_path_mapping(self):
        self._asset_to_path = {}
        for f in self.asset_dir.glob("**/*"):
            if f.is_file():
                name = str(f.relative_to(self.asset_dir))
                self._asset_to_path[name] = f.absolute()

    def _update(self):
        if not self.is_compressed and not self.is_empty:
            with open(self.path / "meta.yaml", "r") as f:
                meta = yaml.safe_load(f)
            self._meta = meta
            self.hyperparameters = NConfig.from_flatten_dict(self._meta["hyperparameters"])
            assert self.key == self.path.stem

            self._update_metric_mapping()
            self._update_asset_path_mapping()

    ### properties ###
    @property
    def key(self):
        return self._meta["key"]

    @property
    def name(self):
        return self._meta["name"]

    @property
    def tags(self):
        return self._meta["tags"]

    @property
    def duration_millis(self):
        return self._meta["duration_millis"]

    @property
    def start_time(self):
        return datetime.datetime.fromtimestamp(
            self._meta["start_time_millis"] / 1000
        ).strftime("%c")

    @property
    def end_time(self):
        return datetime.datetime.fromtimestamp(
            self._meta["end_time_millis"] / 1000
        ).strftime("%c")

    @property
    def output_file(self):
        return self.path / "output.txt"

    @property
    def asset_dir(self):
        return self.path / "assets"

    @property
    def metric_dir(self):
        return self.path / "metrics"

    @property
    def metric_names(self):
        return list(self._metrics.keys())

    ### magics ###
    def __repr__(self):
        return f"{self.name} ({self.key}) start {self.start_time} end {self.end_time} duration {self.duration_millis / 1000}s"

    def __eq__(self, other):
        """Only compare the hyperparameters."""
        assert isinstance(other, SExperiment), f"'==' operator is not implemented for {type(other)}"
        return self.hyperparameters == other.hyperparameters

    def equal(self, other, strict=False):
        assert isinstance(other, SExperiment), f"'==' operator is not implemented for {type(other)}"
        if strict:
            if self.meta != other.meta:
                return False
            if self.hyperparameters != other.hyperparameters:
                return False
            if set(self.metric_names) != set(other.metric_names):
                return False
            for m in self.metric_names:
                mv = self.get_sorted_metrics(m)
                mv_other = other.get_sorted_metrics(m)
                assert mv["min_step"] == mv_other["min_step"]
                assert mv["max_step"] == mv_other["max_step"]

            return True


    ### miscs ###
    def compress(self, delete_old=False):
        path = self.path.parent / f"{self.key}.zip"
        with zipfile.ZipFile(path, mode="w") as archive:
            for f in self.path.glob("**/*"):
                archive.write(f, arcname=f.relative_to(self.path))
        if delete_old:
            shutil.rmtree(self.path)
            self.path = path
            self.is_compressed = True

    def uncompress(self, delete_old=False):
        with zipfile.ZipFile(self.path, "r") as archive:
            for f in archive.namelist():
                archive.extrac(f, self.path.parent)
        if delete_old:
            self.path.unlink()
            self.path = self.path.parent / self.key
            self.is_compressed = False
            self._update()
