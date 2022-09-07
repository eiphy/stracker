import os
import shutil
import zipfile
from pathlib import Path
from typing import Union

from .asset import Asset
from .utils import gen_readable_size_str


class ExperimentAssetsManager:
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

        self._update_asset_mapping()

    def add(self, src_path: Path, filename: Union[str, Path] = None):
        src_path = Path(src_path)
        assert not self.is_compressed, "Cannot add to compressed experiment!"
        assert src_path.is_file()
        filename = filename or src_path.name
        dest_path = self._path / filename
        dest_path.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy(src_path, dest_path)
        self._update_asset_mapping()

    def add_code(self, code_root: Path, code_suffix=None):
        if code_suffix is None:
            code_suffix = [".py", ".yaml", ".cpp", ".sh", ".json", ".cc", ".c"]
        code_root = Path(code_root)
        (self.path / "code").mkdir(exist_ok=True)
        for f in code_root.glob("**/*"):
            if f.is_dir() or f.suffix not in code_suffix:
                continue
            dest = self.path / "code" / f.absolute().relative_to(code_root.absolute())
            shutil.copy(f, dest)

    def get(self, path=None, key=None):
        """
        :param: path: relative path to root.
        """
        assert not self.is_compressed, "Cannot get from compressed experiment!"
        if path is not None:
            return self._assets.get(str(path), None)
        assert key is not None
        tmp = [a for a in self._assets.values() if a.key == key]
        assert len(tmp) == 1
        return tmp[0]

    def remove_asset(self, path=None, key=None):
        """
        :param: path: relative path to root.
        """
        asset = self.get(path=path, key=key)
        if asset is not None:
            asset.path.unlink()
        self._update_asset_mapping()

    def compress(self):
        assert not self.is_compressed, "Experiment is already compressed!"
        path = self._path.parent / f"{self._path.name}.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for f in self._path.glob("**/*"):
                archive.write(f, arcname=f.relative_to(self._path.parent))
        shutil.rmtree(self._path)
        self._path = path
        self.is_compressed = True
        self._update_asset_mapping()

    def uncompress(self):
        assert self.is_compressed, "Experiment is not compressed!"
        with zipfile.ZipFile(self.path, "r") as archive:
            for f in archive.namelist():
                archive.extract(f, self._path.parent)
        self._path.unlink()
        self._path = self.path.parent / self.path.stem
        self.is_compressed = False
        self._update_asset_mapping()

    def _update_asset_mapping(self):
        if self.is_compressed:
            self._assets = {}
            self._key_asset_mapping = {}
            self._size = os.path.getsize(self._path)
        else:
            self._assets = {
                str(f.relative_to(self._path)): Asset(i, f, self._path)
                for i, f in enumerate(f for f in self._path.glob("**/*") if f.is_file())
            }
            self._key_asset_mapping = {a.key: a for a in self._assets.values()}
            self._size = sum(a.size for a in self._assets.values())
        self._size_str = gen_readable_size_str(self._size)

    @property
    def path(self):
        return self._path

    @property
    def key(self):
        return self._path.stem

    @property
    def size(self):
        return self._size

    @property
    def readable_size(self):
        return self._size_str

    @property
    def assets(self):
        return list(self._assets.values())

    @property
    def asset_paths(self):
        return list(self._assets.keys())

    def __len__(self):
        return len(self._assets)

    def __repr__(self):
        return f"{'Compressed experiment' if self.is_compressed else 'Experiment'} ({self.key}) ({self.readable_size}){'' if self.is_compressed else f' {len(self)} assets'} at {self.path}"
