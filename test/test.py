import contextlib
import time
from datetime import datetime
from pathlib import Path

import yaml
from pyinstrument import Profiler
from stracker import SExperiment


class testSExperiment:
    def __init__(self):
        self.exp_path = Path("/Volumes/DATA/exps/test")

        self.test_init()
        print("Pass init test.")

    def test_init(self):
        for d in self.exp_path.glob("*"):
            if d.name in ["copy", "move"]:
                continue
            assert d.is_dir()
            exp = SExperiment(d)

            with open(exp.path / "meta.yaml", "r") as f:
                meta = yaml.safe_load(f)

            assert exp.key == d.stem
            assert exp.key == meta["key"]
            assert exp.name == meta["name"]
            assert exp.tags == meta["tags"]
            assert exp.duration_millis == meta["duration_millis"]
            assert exp.start_time == datetime.fromtimestamp(meta["start_time_millis"] / 1000).strftime("%c")
            assert exp.end_time == datetime.fromtimestamp(meta["end_time_millis"] / 1000).strftime("%c")
            assert exp.hyperparameters.flatten_dict == meta["hyperparameters"]

            assert exp.output_file.is_file()
            assert exp.asset_dir.is_dir()
            assert exp.metric_dir.is_dir()

            set(f.stem for f in exp.metric_dir.glob("*.txt")) == set(exp._metric_to_path.keys())
            for mn, mp in exp._metric_to_path.items():
                assert mp.stem == mn
                assert mp.is_file()

            set(str(f.relative_to(exp.asset_dir)) for f in exp.asset_dir.glob("**/*")) == set(exp._asset_to_path.keys())
            for an, ap in exp._asset_to_path.items():
                assert str(ap.relative_to(exp.asset_dir)) == an
                assert ap.is_file()

        path = self.exp_path / "not_exist"
        assert SExperiment(path, ignore_not_exist=True).meta == SExperiment.empty_meta

        with contextlib.suppress(RuntimeError):
            exp = SExperiment(path, ignore_not_exist=False)
            raise ValueError("Should not be here.")

    def test_copy(self):
        for d in self.exp_path.glob("*"):
            if d.name in ["copy", "move"]:
                continue
            assert d.is_dir()
            exp = SExperiment(d)
            exp.copy(self.exp_path / "copy")

            exp1 = SExperiment(self.exp_path / "copy" / exp.key)
            assert exp1 == exp
            assert exp1.hyperparameters == exp.hyperparameters


    def test_compress_uncompress(self):
        for d in self.exp_path.glob("*"):
            assert d.is_dir()
            exp = SExperiment(d)

testSExperiment()
