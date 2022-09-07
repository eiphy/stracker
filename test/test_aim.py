import shutil
from pathlib import Path

import numpy as np
import yaml
from aim import Run
from naapc import NDict
from stracker import ExperimentAssetsManager

# root = Path("/Volumes/DATA/exps/crowdsourcing")
root = Path("/Volumes/DATA/test")
src = root / "aim"

for d in src.glob("*"):
    assert d.is_dir()
    with open(d / "meta.yaml", "r") as f:
        meta = yaml.safe_load(f)
    run = Run(repo=str(root))
    run.name = meta["name"]
    run["hparams"] = NDict.from_flatten_dict(meta["hyperparameters"]).raw_dict

    for t in meta["tags"]:
        run.add_tag(t)

    run.track(meta["duration_millis"], "duration_millis")
    run.track(meta["start_time_millis"], "start_time_millis")
    run.track(meta["end_time_millis"], "end_time_millis")
    for f in (d / "metrics").glob("*.txt"):
        if f.name.startswith("."):
            continue
        ms = np.loadtxt(f)
        ms = ms.tolist()
        for m, step in ms:
            run.track(float(m), f.stem, step=int(step))

    with open(d / "output.txt", "r") as f:
        output = f.read()
    print(output)

    asset_path = src / run.hash
    (d / "assets").rename(asset_path)

    assert (src / run.hash).is_dir()
    exp = ExperimentAssetsManager(src / run.hash)
    for a in exp.assets:
        a.read()
    a = exp.get("configs.yaml")
    if a is None:
        a = exp.get("config.yaml")
    assert a is not None
    orig_config = a.read()
    assert NDict(run["hparams"]) == NDict(orig_config)

    for f in d.glob("**/*"):
        if f.is_file():
            f.unlink()
        elif f.is_dir():
            shutil.rmtree(f)
    shutil.rmtree(d)
