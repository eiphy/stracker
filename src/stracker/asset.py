import os
from pathlib import Path

from .utils import gen_readable_size_str, safe_read


class Asset:
    def __init__(self, key, path, exp_asset_root):
        path = Path(path)
        assert path.is_file()
        exp_asset_root = Path(exp_asset_root)
        assert exp_asset_root.is_dir()
        assert exp_asset_root in path.parents

        self._key = key
        self._path = path
        self._exp_asset_root = exp_asset_root

        self._size = os.path.getsize(str(path))
        self._size_str = gen_readable_size_str(self._size)

    def read(self):
        return safe_read(self._path)

    @property
    def path(self):
        return self._path

    @property
    def key(self):
        return self._key

    @property
    def path(self):
        return self._path

    @property
    def exp_asset_root(self):
        return self._exp_asset_root

    @property
    def name(self):
        return self._path.name

    @property
    def rel_path(self):
        return str(self._path.relative_to(self._exp_asset_root))

    @property
    def size(self):
        return self._size

    @property
    def readable_size(self):
        return self._size_str

    def __repr__(self):
        return f"Asset {self.name} ({self.readable_size}) at ({self.rel_path})"
