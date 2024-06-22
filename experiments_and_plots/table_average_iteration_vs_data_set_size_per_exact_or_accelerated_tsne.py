"""
This script creates a table with statistical values (minimum, average, standard deviation, maximum) on the run times of
embedding iterations for the different data sets. Specifically, these data are compared for the accelerated iterations
that use the polar quad tree acceleration structure as well as for the non-accelerated, exact hyperbolic t-SNE
implementation.
"""

###########
# IMPORTS #
###########

from pathlib import Path
import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from configs import setup_experiment

####################
# READING THE DATA #
####################

_, cfgs, paths = setup_experiment([1010, 1000, 1100])

print("Starting reading...")
print("Reading OVERVIEW...", end="")
results_path = Path(os.path.join(paths["results_path"], "samples_per_data_set"))
df = pd.read_csv(results_path.joinpath("overview.csv"))
print("OK")
timings_dfs = []
cnt = 1
rcrds = df.to_records()
for record in rcrds:
    csv_path = record.run_directory.replace(".", str(results_path)) + "/timings.csv"
    print(f"({cnt}/{len(rcrds)}) Reading {csv_path}...", end="")
    timing_df = pd.read_csv(csv_path)
    timing_df = timing_df[(timing_df.time_type == "tot_gradient")]

    for cn in df.columns:
        timing_df[cn] = record[cn]
    timings_dfs.append(timing_df)
    print("OK")
    cnt += 1

print("Plotting...", end="")
timings_df = pd.concat(timings_dfs, axis=0, ignore_index=True)
timings_df["early_exag"] = np.repeat(False, timings_df.shape[0])
timings_df.loc[timings_df.it_n <= 250, "early_exag"] = True
del timings_dfs

for cfg in cfgs:
    # Filter out only the exact, i.e., non-accelerated data
    plot_times_df = timings_df.copy()
    plot_times_df = plot_times_df[(plot_times_df.name == cfg["name"])]

    # Print Min, Avg, Std, Max of the timings per dataset per size
    grouped = plot_times_df.groupby(["dataset", "sample_size"])
    print(f"Statistics {cfg['name']}:")
    print("\nMIN:")
    print(grouped["total_time"].min())
    print("\nMEAN:")
    print(grouped["total_time"].mean())
    print("\nSTD:")
    print(grouped["total_time"].std())
    print("\nMAX:")
    print(grouped["total_time"].max())
