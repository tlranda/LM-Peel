# Dependent libraries
import numpy as np

# Builtin libraries
import argparse
import pathlib
import re

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument("filename", type=pathlib.Path,
                     help="File to parse")
    return prs

def parse(args=None, prs=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    return args

get_nums_regex = re.compile(r".*?(-?\d+\.?\d*)")
def extract_mean_std(line):
    matches = re.findall(get_nums_regex, line)
    return float(matches[0]), float(matches[1])

def extract_num(line):
    matches = re.findall(get_nums_regex,line)
    return float(matches[0])

def extract_all_numbers(line):
    matches = re.findall(get_nums_regex,line)
    return [float(_) for _ in matches]

def main():
    args = parse()
    clt_generation_mean = []
    clt_generation_std  = []
    clt_dist_mean   = []
    clt_dist_median = []
    clt_distribution_mean = []
    clt_distribution_std  = []
    mare = []
    msre = []
    r2s = []
    tokens = dict()
    # Parse
    with open(args.filename, 'r') as f:
        for line in f.readlines():
            # Distribution
            if line.startswith("CLT_Mean:"):
                mean, std = extract_mean_std(line)
                clt_distribution_mean.append(mean)
                clt_distribution_std.append(std)
            elif line.startswith("CLT_GEN_Mean:"):
                mean, std = extract_mean_std(line)
                clt_generation_mean.append(mean)
                clt_generation_std.append(std)
            elif line.startswith("CLT_MEAN_Mean:"):
                mean, median = extract_mean_std(line)
                clt_dist_mean.append(mean)
                clt_dist_median.append(median)
            # Metrics
            elif line.startswith("R2 Score:"):
                r2s.append(extract_all_numbers(line)[-1])
            elif line.lstrip().startswith("MARE:"):
                mare.append(extract_num(line))
            elif line.lstrip().startswith("MSRE:"):
                msre.append(extract_num(line))
            elif line.startswith("Class ['"):
                clip = line.rindex('[')
                nums = extract_all_numbers(line[clip:])
                per_token = nums[:-1]
                total = nums[-1]
                for tokidx, token in enumerate(per_token):
                    if str(tokidx) not in tokens:
                        tokens[str(tokidx)] = [token]
                    else:
                        tokens[str(tokidx)].append(token)
                if 'n_tokens' not in tokens:
                    tokens['n_tokens'] = [len(per_token)]
                else:
                    tokens['n_tokens'].append(len(per_token))
                if 'total' not in tokens:
                    tokens['total'] = [total]
                else:
                    tokens['total'].append(total)
    clt_distribution_mean = np.abs(np.asarray(clt_distribution_mean))
    clt_distribution_std = np.abs(np.asarray(clt_distribution_std))
    clt_dist_mean = np.abs(np.asarray(clt_dist_mean))
    clt_dist_median = np.abs(np.asarray(clt_dist_median))
    clt_generation_mean = np.abs(np.asarray(clt_generation_mean))
    clt_generation_std = np.abs(np.asarray(clt_generation_std))
    r2s = np.asarray(r2s)
    mare = np.asarray(mare)
    msre = np.asarray(msre)
    for key in tokens:
        tokens[key] = np.asarray(tokens[key])
    # Print
    print("Mean of means (DISTRIBUTION):", clt_distribution_mean.mean())
    print("Std  of means (DISTRIBUTION):", clt_distribution_mean.std())
    print("Mean of stds (DISTRIBUTION):", clt_distribution_std.mean())
    print("Std  of stds (DISTRIBUTION):", clt_distribution_std.std())
    print("Mean of means (GENERATION):", clt_generation_mean.mean())
    print("Std  of means (GENERATION):", clt_generation_mean.std())
    print("Mean of stds (GENERATION):", clt_generation_std.mean())
    print("Std  of stds (GENERATION):", clt_generation_std.std())
    print("Mean of DIST-means:", clt_dist_mean.mean())
    print("Std  of DIST-means:", clt_dist_mean.std())
    print("Mean of DIST-medians:", clt_dist_median.mean())
    print("Std  of DIST-medians:", clt_dist_median.std())
    print("--"*32)
    print("Mean R2:", r2s.mean())
    print("Std  R2:", r2s.std())
    print("Max  R2:", r2s.max())
    print("Mean MARE:", mare.mean())
    print("Std  MARE:", mare.std())
    print("Mean MSRE:", msre.mean())
    print("Std  MSRE:", msre.std())
    for tkey in sorted(tokens.keys()):
        print("Mean",tkey,":", tokens[tkey].mean())
        print("Std",tkey,":", tokens[tkey].std())
        print("N",tkey,":", len(tokens[tkey]))

if __name__ == '__main__':
    main()

