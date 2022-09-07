import json
import pickle
from pathlib import Path
from typing import Union

import yaml
from naapc import NDict


def safe_read(path: Path):
    if path.suffix in [".txt", ".py", ".sh"]:
        with open(path, "r") as f:
            return f.read()
    elif path.suffix == ".yaml":
        with open(path, "r") as f:
            return yaml.safe_load(f)
    elif path.suffix == ".json":
        with open(path, "r") as f:
            return json.load(f)
    elif path.suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(path)
    elif path.suffix == ".pt":
        import torch

        try:
            return torch.load(path)
        except Exception:
            try:
                return torch.load(path, map_location=torch.device("cpu"))
            except Exception:
                if path.name.endswith("pred_label.pt"):
                    path.unlink()
                else:
                    raise RuntimeError
    elif path.suffix == ".pyc":
        with open(path, "rb") as f:
            return f.read()
    else:
        raise TypeError(f"Unexpected file type {path.suffix} of {path}.")


def gen_readable_size_str(size: Union[float, int]):
    if size <= 1024:
        return f"{size:.3f}B"
    elif size <= 1024 * 1024:
        return f"{size / 1024:.3f}KB"
    elif size <= 1024**3:
        return f"{size / 1024 / 1024:.3f}MB"
    else:
        return f"{size / 1024 / 1024 / 1024:.3f}GB"


def get_aim_hp_ndict(run, path):
    return NDict(run["hparams"])[path]
