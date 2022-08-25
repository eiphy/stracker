from pathlib import Path


class SProjectDB:
    def __init__(self, path):
        path = Path(path)

class SProject:
    def __init__(self, path):
        self.path = Path(path)

        self.exp_dir.mkdir(exist_ok=True, parents=True)
        self.archive_dir.mkdir(exist_ok=True, parents=True)
        self.front_dir.mkdir(exist_ok=True, parents=True)
        self.raw_dir.mkdir(exist_ok=True, parents=True)

    @property
    def exp_dir(self):
        return self.path / "exps"

    @property
    def archive_dir(self):
        return self.path / "archive"

    @property
    def front_dir(self):
        return self.path / "front"

    @property
    def raw_dir(self):
        return self.path / "raw"
