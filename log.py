import logging
import logging.config
import shutil
import sys
import time
import uuid
import zipfile
from pathlib import Path

from utils.util import save_obj


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.error("Exception", exc_info=(exc_type, exc_value, exc_traceback))


def setup_logging(logging_file):
    logging_file = Path(logging_file)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(str(logging_file.absolute()), mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info(f"Saving outputs to {logging_file}")
    sys.excepthook = handle_exception


class LocalExperimentLogger:
    code_suffix = [".py", ".yaml"]

    def __init__(self, path, name, paras, tags, configs):
        self.path = Path(path) / f"{str(uuid.uuid4())}.pkl"
        while self.path.is_dir() or self.path.is_file():
            self.path = Path(path) / f"{str(uuid.uuid4())}.pkl"
        self.key = self.path.stem
        self.asset_path.mkdir(parents=True)
        setup_logging(self.output_file)

        # Logging metas.
        self.name = name
        self.paras = paras
        self.tags = tags
        self.configs = configs

        # Initialize metrics.
        self.metrics = {}
        self.meta = {
            "projectName": "_local",
            "workspaceName": "_local",
            "startTimeMillis": time.time() * 1000,
        }
        self.finished = False

    def set_configs(self, configs):
        self.configs = configs

    @property
    def asset_path(self):
        return self.path.parent / self.path.stem

    @property
    def zip_path(self):
        return self.path.parent / f"{self.path.stem}.zip"

    @property
    def output_file(self):
        return self.path.parent / f"{self.path.stem}.txt"

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
            "finished": self.finished,
        }

    #### Loggings. ####
    def log_metric(self, k, v, step):
        m = (step, v)
        if k in self.metrics:
            self.metrics[k].append(m)
        else:
            self.metrics[k] = [m]

    def log_metrics(self, ms, step):
        for k, v in ms.items():
            self.log_metric(k, v, step)

    def log_asset(self, asset, filename):
        path = self.asset_path / filename
        path.parent.mkdir(exist_ok=True, parents=True)
        save_obj(asset, path)
        return path

    def log_file(self, file):
        file = Path(file)
        dest = self.asset_path / file.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(file, dest)

    def log_folder(self, folder):
        folder = Path(folder)
        for f in folder.glob("**/*"):
            if f.is_dir():
                continue
            dest = (
                self.asset_path
                / folder.name
                / f.absolute().relative_to(folder.absolute())
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, dest)

    #### Mics. ####
    def _convert_metrics(self):
        if "value" in list(self.metrics.values())[0]:
            return
        metrics = {
            name: {"value": [], "max_step": 0, "min_step": float("inf")}
            for name in self.metrics.keys()
        }
        for name, ms in self.metrics.items():
            for m in ms:
                step, v = m
                assert step > 0
                if step > metrics[name]["max_step"]:
                    diff = step - metrics[name]["max_step"]
                    metrics[name]["max_step"] = step
                    metrics[name]["value"].extend([float("nan")] * diff)
                if step < metrics[name]["min_step"]:
                    metrics[name]["min_step"] = step
                if v is not None:
                    metrics[name]["value"][step - 1] = float(v)
        self.metrics = metrics

    #### End. ####
    def end(self):
        """. Record end time and durations.
        - [x] add end time and durations.
        - [x] convert metrics.
        - [x] save data dict
        - [x] compress folder.
        - [x] delete files.
        """
        self.meta["endTimeMillis"] = time.time() * 1000
        self.meta["durationMillis"] = (
            self.meta["endTimeMillis"] - self.meta["startTimeMillis"]
        )
        self._convert_metrics()
        self.finished = True
        save_obj(self.data_dict, self.path)
        with zipfile.ZipFile(self.zip_path, mode="w") as archive:
            archive.write(self.path, arcname=self.path.name)
            archive.write(self.output_file, arcname=self.output_file.name)
            for f in self.asset_path.glob("**/*"):
                archive.write(f, arcname=f.relative_to(self.asset_path.parent))
        self.path.unlink()
        self.output_file.unlink()
        shutil.rmtree(self.asset_path)
