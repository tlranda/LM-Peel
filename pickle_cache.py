import pathlib
import pickle

class PickleCache:
    def __init__(self, record: pathlib.Path):
        if not isinstance(record, pathlib.Path):
            record = pathlib.Path(record)
        self.record = record
        self.data = dict()
        if self.record.exists():
            self.from_pickle()

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __contains__(self, key):
        return key in self.data

    def to_pickle(self):
        with open(self.record, 'wb') as f:
            pickle.dump(self.data, f)

    def from_pickle(self):
        with open(self.record, 'rb') as f:
            self.data = pickle.load(f)

