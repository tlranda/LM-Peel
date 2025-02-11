import time

class TimerDict():
    """
        td = TimerDict()
        td.open('first section')
        td.close('first section')
        chkpt = str(td) == "{'first section': 0.00000001}"
        td.dump() == (eval(chkpt), set())
        td.open('never closed')
        td.dump() == (eval(chkpt), {'never closed',})
    """
    def __init__(self):
        self.timers = dict()
        self.opened_intervals = set()
        self.closed_intervals = set()

    def __repr__(self):
        return self.timers.__repr__()

    def __getitem__(self, key):
        override = time.time()
        if key in self.opened_intervals:
            self.close(key, override=override)
        else:
            self.open(key, override=override)

    def open(self, key, override=None):
        propose = time.time()
        if override is not None:
            propose = override
        if key in self.opened_intervals:
            raise KeyError(f"Double open for key '{key}'")
        self.opened_intervals.add(key)
        self.timers[key] = propose

    def close(self, key, override=None):
        propose = time.time()
        if override is not None:
            propose = override
        if key not in self.opened_intervals:
            raise KeyError(f"Closing key '{key}' that was never opened")
        if key in self.closed_intervals:
            raise KeyError(f"Double close for key '{key}'")
        self.timers[key] = propose - self.timers[key]
        self.closed_intervals.add(key)

    def dump(self):
        closed_timers = dict((k,self.timers[k]) for k in self.closed_intervals)
        return (closed_timers, self.opened_intervals.difference(self.closed_intervals))

