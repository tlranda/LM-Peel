import numpy as np

import argparse
import pathlib
import re

prs = argparse.ArgumentParser()
prs.add_argument("file", type=pathlib.Path,
                 help="File of haystack text trace to read")
args = prs.parse_args()

total = re.compile(r".*?\(total=(\d+)\)")
needles_at = re.compile(r"Haystack needles @(\d+\.\d+): (\d+)")

current_haystack = None
needle_dict = dict()
with open(args.file,'r') as f:
    for line in f.readlines():
        line = line.rstrip()
        # Find a new total section
        totmatch = re.match(total, line)
        if totmatch is not None:
            current_haystack = int(totmatch.groups()[0])
            continue
        # Find a new needle
        needmatch = re.match(needles_at, line)
        if needmatch is None:
            continue
        needle_precision = float(needmatch.groups()[0])
        needles_found = int(needmatch.groups()[1])
        if needle_precision not in needle_dict.keys():
            needle_dict[needle_precision] = []
        needle_dict[needle_precision].append(needles_found / current_haystack)
# Get means
for key in sorted(needle_dict.keys()):
    avg_needles = np.mean(needle_dict[key])
    print(f"Average Needles @ {key}: {avg_needles}")

