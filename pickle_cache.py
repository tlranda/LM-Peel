import pathlib
import pickle

class PickleCache:
    def __init__(self, record: pathlib.Path, heavy=False):
        if not isinstance(record, pathlib.Path):
            record = pathlib.Path(record)
        self.record = record
        self.data = dict()
        self.loaded = False
        if heavy and self.record.exists():
            self.from_pickle()

    def __getitem__(self, key):
        # Current check with load
        if self.loaded:
            return self.data[key]
        # Current check without load
        if key in self.data:
            return self.data[key]
        # Light load for basic access
        return self.from_pickle(light=True)[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __contains__(self, key):
        # Heavy check ready for all data
        if self.loaded:
            return key in self.data
        # Light check current data
        if key in self.data:
            return True
        # If we contain-check, we're likely to access it; become heavy
        return key in self.from_pickle(light=False)

    def to_pickle(self, light=True):
        if not self.loaded:
            self.from_pickle(light=False, merge=True)
        with open(self.record, 'wb') as f:
            pickle.dump(self.data, f)
        # Drop data
        if light:
            self.data = dict()
            self.loaded = False

    def from_pickle(self, light=False, merge=True):
        if merge:
            og_data = self.data
        if not self.record.exists():
            return og_data
        with open(self.record, 'rb') as f:
            data = pickle.load(f)
        if merge:
            data.update(og_data.items())
        # Hold onto data
        if not light:
            self.data = data
            self.loaded = True
        return data

