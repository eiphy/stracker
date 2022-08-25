from pathlib import Path


class SLogger:
    def __init__(self, path, name, hyperparameters, tags):
        path = Path(path)
