from pathlib import Path

import numpy as np


# TODO json, yaml, pickle.
class Metric:
    def __init__(self, path):
        self.path = Path(path)
        assert self.path.is_file(), f"File not exist {self.path}."
        self.reset()

    @property
    def name(self):
        return self.path.stem

    @property
    def array(self):
        if self._value is None:
            self._read()
        return np.array(self._value)

    @property
    def min_step(self):
        if self._min_step is None:
            self._read()
        return self._min_step
    
    @property
    def max_step(self):
        if self._max_step is None:
            self._read()
        return self._max_step

    @property
    def step_array(self):
        if self._min_step is None:
            self._read()
        assert self._max_step is not None
        return np.arange(self._min_step, self._max_step + 1)

    def reset(self):
        self._value = None
        self._min_step = None
        self._max_step = None

    def _read(self):
        assert self.path.suffix == ".txt"
        ms = np.loadtxt(str(self.path)).tolist()
        ms = sorted(ms, key=lambda x: x[1])
        offset = ms[0][1]
        value = []
        for v, _i in ms:
            i = _i - offset
            if i < len(value):
                value.append(float("nan"))
            elif i == len(value):
                value.append(v)
            else:
                raise RuntimeError
        self._value = value
        self._min_step = ms[0][1]
        self._max_step = ms[-1][1]

    def __repr__(self):
        return f"Metric {self.name}: min step-{self.min_step} max step-{self.max_step}"
