"""
This script creates a table with statistical values (minimum, average, standard deviation, maximum) on the run times of
embedding iterations for the different data sets. Specifically, these data are compared for the accelerated iterations
that use the polar quad tree acceleration structure as well as for the non-accelerated, exact hyperbolic t-SNE
implementation.
"""

###########
# IMPORTS #
###########

import os
from pathlib import Path
import pandas as pd
import numpy as np

from configs import setup_experiment

####################
# READING THE DATA #
####################

_, cfgs, paths = setup_experiment([1000, 1100, 1010])

results_path = Path(os.path.join(paths["results_path"], "samples_per_data_set"))
df = pd.read_csv(results_path.joinpath("overview.csv"))
timings_dfs = []
for record in df.to_records():
    timing_df = pd.read_csv(record.run_directory.replace(".", str(results_path)) + "/timings.csv")
    timing_df = timing_df[(timing_df.time_type == "tot_gradient")]

    for cn in df.columns:
        timing_df[cn] = record[cn]
    timings_dfs.append(timing_df)

timings_df = pd.concat(timings_dfs, axis=0, ignore_index=True)
timings_df["early_exag"] = np.repeat(False, timings_df.shape[0])
timings_df.loc[timings_df.it_n <= 250, "early_exag"] = True
del timings_dfs

############################
# COMPUTING THE STATISTICS #
############################

# Work with the "equal length" data, as this splitting technique proved to be more efficient, filtering by
# "equal_length" contains both accelerated and exact data.
# plot_times_df = plot_times_df[(plot_times_df.splitting_strategy == "equal_length")]

for cfg in cfgs:
    # Filter out only the exact, i.e., non-accelerated data
    plot_times_df = timing_df.copy()
    plot_times_df = plot_times_df[(plot_times_df.name == cfg["name"])]

    # Print Min, Avg, Std, Max of the timings per dataset per size
    grouped = plot_times_df.groupby(["dataset", "sample_size"])
    print(f"Statistics {cfg['name']}:")
    print(f'min: {grouped["total_time"].min()}')
    print(f'mean: {grouped["total_time"].mean()}')
    print(f'std: {grouped["total_time"].std()}')
    print(f'max: {grouped["total_time"].max()}')
