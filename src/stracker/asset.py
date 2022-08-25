from pathlib import Path


class Asset:
    def __init__(self, path, dir_path):
        path = Path(path)
        assert path.is_file()
        self.path = path

    @property
    def name(self):
        return self.path.name

    @property
    def rel_path(self):
        return self.path

    # def _read(self):

