import enum
import filecmp
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.append(str(Path().parent.absolute()))

import numpy as np
from tqdm import tqdm
from utils.experiment import LocalCometProject

pro_path = Path("/Volumes/DATA/exps/crowdsourcing")
pro = LocalCometProject(pro_path)

# pro.bring_front(keys=pro.keys[:10])
exp_path = pro_path / "exps"
exp_path.mkdir(exist_ok=True)
gen = True

if gen:
    for k in tqdm(pro.keys):
        if (pro_path / f"{k}.pkl").is_file():
            exp = pro.get(k)

            path = exp_path / f"{k}"
            path.mkdir(exist_ok=True)

            # meta
            meta = {
                "key": str(exp.key),
                "name": str(exp.name),
                "tags": [str(t) for t in exp.tags],
                "hyperparameters": exp.configs.tree.tree,
                "duration_millis": float(exp.meta["durationMillis"]),
                "start_time_millis": float(exp.meta["startTimeMillis"]),
                "end_time_millis": float(exp.meta["endTimeMillis"]),
            }
            with open(path / "meta.yaml", "w") as f:
                yaml.dump(meta, f, sort_keys=False)

            with open(path / "meta.yaml", "r") as f:
                meta = yaml.safe_load(f)

            # outputfile
            output_path = path / "output.txt"
            exp.output_file.rename(output_path)

            # asset
            asset_path = path / "assets"
            asset_path.mkdir(exist_ok=True)
            exp.asset_dir.rename(asset_path)

            # metrics
            metric_path = path / "metrics"
            metric_path.mkdir()
            for name, v in exp.metrics.items():
                ms = [
                    [v["value"][i], t]
                    for t, i in enumerate(range(len(v["value"])), start=v["min_step"])
                ]
                np.savetxt(str(metric_path / f"{name}.txt"), np.array(ms))
