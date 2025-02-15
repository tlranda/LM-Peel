# Dependent libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Builtin libraries
import argparse
import pathlib
import re

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument("filename", type=pathlib.Path,
                     help="File to parse")
    prs.add_argument('--export', type=pathlib.Path, default=None,
                     help="Filename to save figure to (if not given, show plot)")
    return prs

def parse(args=None, prs=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    return args

get_nums_regex = re.compile(r".*?(\d+.?\d*)")
def extract_mean_std(line):
    matches = re.findall(get_nums_regex, line)
    return float(matches[0]), float(matches[1])

def main():
    args = parse()
    clt_generation_mean = []
    clt_generation_std  = []
    clt_distribution_mean = []
    clt_distribution_std  = []
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
    clt_distribution_mean = np.abs(np.asarray(clt_distribution_mean))
    clt_distribution_std = np.abs(np.asarray(clt_distribution_std))
    clt_generation_mean = np.abs(np.asarray(clt_generation_mean))
    clt_generation_std = np.abs(np.asarray(clt_generation_std))
    # Print
    print("Mean of means (DISTRIBUTION):", clt_distribution_mean.mean())
    print("Std  of means (DISTRIBUTION):", clt_distribution_mean.std())
    print("Mean of stds (DISTRIBUTION):", clt_distribution_std.mean())
    print("Std  of stds (DISTRIBUTION):", clt_distribution_std.std())
    print("Mean of means (GENERATION):", clt_generation_mean.mean())
    print("Std  of means (GENERATION):", clt_generation_mean.std())
    print("Mean of stds (GENERATION):", clt_generation_std.mean())
    print("Std  of stds (GENERATION):", clt_generation_std.std())
    # Plot
    dist_fig, (dist_mean_ax, dist_std_ax) = plt.subplots(1,2, figsize=(12,6))
    dist_mean_ax = sns.kdeplot(x=clt_distribution_mean,
                               ax=dist_mean_ax,
                               label="Distribution Means",
                               common_norm=False,
                               clip=(0.,max(clt_distribution_mean)),
                               log_scale=True,
                               )
    dist_std_ax = sns.kdeplot(x=clt_distribution_std,
                              ax=dist_std_ax,
                              label="Distribution Stddevs",
                              common_norm=False,
                              clip=(0.,max(clt_distribution_std)),
                              #log_scale=True,
                              )
    dist_fig.suptitle("CLT Distribution of Possible Values")
    dist_mean_ax.set_xlabel("Mean Relative Error")
    dist_mean_ax.set_ylabel("Kernel Density Estimate")
    dist_std_ax.set_xlabel("Stddev Relative Error")
    dist_std_ax.set_ylabel("")
    gen_fig, (gen_mean_ax, gen_std_ax) = plt.subplots(1,2, figsize=(12,6))
    gen_mean_ax = sns.kdeplot(x=clt_generation_mean,
                              ax=gen_mean_ax,
                              label="Generated Values",
                              common_norm=False,
                              clip=(0.,max(clt_generation_mean)),
                              #log_scale=True,
                              )
    gen_std_ax = sns.kdeplot(x=clt_generation_std,
                             ax=gen_std_ax,
                             label="Generated Stddevs",
                             common_norm=False,
                             clip=(0.,max(clt_generation_std)),
                             #log_scale=True,
                             )
    gen_fig.suptitle("CLT Distribution of Generated Values")
    gen_mean_ax.set_xlabel("Mean Relative Error")
    gen_mean_ax.set_ylabel("Kernel Density Estimate")
    gen_std_ax.set_xlabel("Stddev Relative Error")
    gen_std_ax.set_ylabel("")
    plt.show()

if __name__ == '__main__':
    main()

