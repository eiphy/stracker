import os
import shutil
import zipfile
from pathlib import Path
from typing import Union

from .exp import ExperimentAssetsManager
from .utils import gen_readable_size_str


class ProjectAssetsManager:
    def __init__(self, path: Path):
        if path.suffix == ".zip":
            assert path.is_file()
            self._path = path
            self.is_compressed = True
        else:
            assert path.suffix == ""
            path.mkdir(exist_ok=True, parents=True)
            self._path = path
            self.is_compressed = False

        self.update()

    def get(self, key=None, path=None):
        assert not self.is_compressed, "Cannot get from compressed project!"
        key = key or path.stem
        assert key is not None
        return self._exps.get(key, None)

    def remove_exp(self, key=None, path=None):
        exp = self.get(key, path)
        if exp is not None:
            if exp.path.is_file():
                exp.path.unlink()
            else:
                shutil.rmtree(exp.path)
        self.update()

    def compress(self, remove_old=False):
        assert not self.is_compressed, "Project is already compressed!"
        path = self._path.parent / f"{self._path.name}.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for f in self._path.glob("**/*"):
                archive.write(f, arcname=f.relative_to(self._path.parent))
        if remove_old:
            shutil.rmtree(self._path)
        self._path = path
        self.is_compressed = True
        self.update()

    def uncompress(self, remove_old=False):
        assert self.is_compressed, "Project is not compressed!"
        with zipfile.ZipFile(self._path, "r") as archive:
            for f in archive.namelist():
                archive.extract(f, self._path.parent)
        if remove_old:
            self._path.unlink()
        path = self._path.parent / self._path.stem
        self._path = path
        self.is_compressed = False
        self.update()

    def update(self):
        if self.is_compressed:
            self._exps = {}
            self._size = os.path.getsize(self._path)
        else:
            self._exps = {
                f.stem: ExperimentAssetsManager(f)
                for f in self._path.glob("*")
                if f.is_dir() or f.suffix == ".zip"
            }
            self._size = sum(e.size for e in self._exps.values())
        self._size_str = gen_readable_size_str(self._size)

    @property
    def path(self):
        return self._path

    @property
    def keys(self):
        return list(self._exps.keys())

    @property
    def size(self):
        return self._size

    @property
    def readable_size(self):
        return self._size_str

    def __len__(self):
        return len(self._exps)

    def __repr__(self):
        return f"{'Compressed project' if self.is_compressed else 'Project'}{'' if self.is_compressed else f' {len(self)} experiments'} ({self.readable_size}) at {self.path}"
