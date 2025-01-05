"""
This script creates a pickle file containing simulated race data. First, an existing pickle file is read to get
competitor start times and control sequences. Then, a simulated race is created by randomly generating the
performance of competitors within a class, then stepping through the race to calculate split times for each control.
An optional parameter allows for the inclusion of a 'trend' in the performance of competitors vs split order. This is
useful for testing the split time analysis scripts.
"""

import pandas as pd
import numpy as np
from tkinter import filedialog
import time

import utilities


def simulate_split_performance(input_split_df, split_order_perf_trend=0.0, class_perf_var=0.1, competitor_perf_var=0.1):
    # *************************************************************
    # Calculate data required from existing split performance data
    # *************************************************************

    # Temporarily reduce to one race and one class for testing
    input_split_df = input_split_df[(input_split_df['race_id'] == 1)].copy()

    start_time = time.time()

    # Calculate average split times for each control sequence
    input_split_df['avg_split_time'] = input_split_df.groupby(['race_id', 'ctrl_seq'])['split_time'].transform(
        lambda x: x.mean())

    # Calculate the average time to reach the split control sequence for each class
    input_split_df['time_to_split'] = input_split_df['timestamp'] - input_split_df['start_time']
    input_split_df['class_avg_time_to_split'] = input_split_df.groupby(['race_id', 'ctrl_seq', 'class_id'])[
        'time_to_split'].transform(lambda x: x.mean())

    print(f'Time to calculate average split times: {time.time() - start_time} seconds')
    start_time = time.time()

    for race_id in input_split_df['race_id'].unique():

        # Calculate the leg order of control sequences within each class for each race
        for class_id in input_split_df['class_id'].unique():
            leg_order = {}
            class_mask = (input_split_df['class_id'] == class_id) & (input_split_df['race_id'] == race_id)
            ctrl_sequences = input_split_df[class_mask]['ctrl_seq'].unique()
            # Find the order of legs by checking the order of the control sequences by timestamp for one competitor.
            # If not all controls are present, use a different competitor
            for competitor_id in input_split_df[class_mask]['competitor_id'].unique():
                competitor_mask = class_mask & (input_split_df['competitor_id'] == competitor_id)
                if len(input_split_df[competitor_mask]) == len(ctrl_sequences):
                    # Determine the order of ctrl_seq based on timestamp
                    ctrl_order_mapping = input_split_df[competitor_mask].sort_values('timestamp').reset_index(drop=True)
                    ctrl_order_mapping['leg_order'] = ctrl_order_mapping.index

                    # Create a dictionary to map ctrl_seq to ctrl_order
                    ctrl_order_dict = dict(zip(ctrl_order_mapping['ctrl_seq'], ctrl_order_mapping['leg_order']))

                    # Apply the mapping to all rows to create the new ctrl_order column
                    input_split_df.loc[class_mask, 'leg_order'] = input_split_df['ctrl_seq'].map(ctrl_order_dict)
                    break

        print(f'Time to calculate leg orders: {time.time() - start_time} seconds')
        start_time = time.time()

    # *************************************************************
    # Set up a new dataframe to store simulated splits
    # *************************************************************

    # Copy over competitor_id, class_id, start_time, ctrl_seq, nat_perf to new simulated dataframe
    sim_splits_df = input_split_df[
        ['competitor_id', 'class_id', 'start_time', 'ctrl_seq', 'avg_split_time',
         'class_avg_time_to_split', 'leg_order', 'race_id']].copy()

    # Grouping by class_id, add a new column with randomized natural performance. Multiply by an additional random number
    # to simulate the overall performance level of the class
    for class_id in sim_splits_df['class_id'].unique():
        class_mask = sim_splits_df['class_id'] == class_id
        class_comp = sim_splits_df[class_mask]['competitor_id'].unique()
        comp_perf = np.random.normal(1, competitor_perf_var, len(class_comp))
        comp_perf_dict = dict(zip(class_comp, comp_perf))
        sim_splits_df.loc[class_mask, 'nat_perf'] = (sim_splits_df.loc[class_mask, 'competitor_id'].map(comp_perf_dict)
                                                     * np.random.normal(1, class_perf_var, 1))

    # Normalize natural performance across all classes to have a mean of 1
    sim_splits_df['nat_perf'] = sim_splits_df['nat_perf'] / np.mean(sim_splits_df['nat_perf'])

    print(f'Time to set up random natural performance: {time.time() - start_time} seconds')
    start_time = time.time()

    # *************************************************************
    # Simulate
    # *************************************************************

    # Set up null timestamp, split_time, columns
    sim_splits_df['timestamp'] = np.nan
    sim_splits_df['split_time'] = np.nan

    # We don't simulate the first control sequence, so set the timestamp and split time to the start time
    # plus the average time to reach the first control
    first_split_mask = sim_splits_df['leg_order'] == 0
    sim_splits_df.loc[first_split_mask, 'timestamp'] = np.round(sim_splits_df[first_split_mask]['start_time'] + \
                                                       sim_splits_df[first_split_mask]['class_avg_time_to_split'])

    # Remove competitors who have no leg_order 0
    # Identify competitors with leg_order equal to 0
    competitors_with_leg_order_0 = input_split_df[input_split_df['leg_order'] == 0]['competitor_id'].unique()

    # Filter the DataFrame to keep only rows associated with these competitors
    input_split_df = input_split_df.loc[input_split_df['competitor_id'].isin(competitors_with_leg_order_0)]

    # Simulate all races
    for race_id in input_split_df['race_id'].unique():
        # Set initial time to minimum start time
        race_mask = sim_splits_df['race_id'] == race_id
        race_start_time = np.min(input_split_df.loc[race_mask, 'start_time'])
        race_time = race_start_time
        splits_remaining = len(input_split_df.loc[race_mask])

        # dictionary to keep track of split order for each control sequence
        split_order_dict = {}
        for ctrl_seq in sim_splits_df['ctrl_seq'].unique():
            split_order_dict[ctrl_seq] = 0

        print(f'Starting race {race_id} simulation')

        while splits_remaining > 0:
            # Print race time every 10 minutes
            if (race_time - race_start_time) % 600 == 0:
                print(f'Simulated up to race time: {race_time - race_start_time}. Splits remaining: {splits_remaining}')

            # Masks for the current race and timestep
            timestep_mask = (sim_splits_df['timestamp'] == race_time) & (sim_splits_df['race_id'] == race_id)
            if len(sim_splits_df[timestep_mask]) == 0:
                race_time += 1
                continue

            # Set the split order for all legs starting at this time
            sim_splits_df.loc[timestep_mask, 'split_order'] = sim_splits_df.loc[timestep_mask, 'ctrl_seq'].map(
                split_order_dict)
            sim_splits_df.loc[race_mask & timestep_mask, 'split_order'] = sim_splits_df.loc[
                race_mask & timestep_mask, 'ctrl_seq'].map(split_order_dict)

            # Calculate the split time for the control sequence as an integer
            sim_splits_df.loc[timestep_mask, 'split_time'] = np.round(sim_splits_df.loc[timestep_mask, 'nat_perf'] * \
                                                                      sim_splits_df.loc[
                                                                          timestep_mask, 'avg_split_time'] * \
                                                                      (1 + split_order_perf_trend *
                                                                       sim_splits_df.loc[
                                                                           timestep_mask, 'split_order']))

            # Calculate the timestamp for the next control sequence
            for competitor_id in sim_splits_df[timestep_mask]['competitor_id'].unique():
                competitor_mask = (sim_splits_df['competitor_id'] == competitor_id)
                curr_leg_order = sim_splits_df[timestep_mask & competitor_mask]['leg_order'].values[0]
                # Continue if this is the last control sequence for the competitor
                if len(sim_splits_df[race_mask & competitor_mask]) == curr_leg_order + 1:
                    continue
                next_leg_mask = (sim_splits_df['leg_order'] == curr_leg_order + 1)
                sim_splits_df.loc[race_mask & competitor_mask & next_leg_mask, 'timestamp'] = \
                    sim_splits_df[timestep_mask & competitor_mask]['timestamp'].values[0] + \
                    sim_splits_df[timestep_mask & competitor_mask]['split_time'].values[0]
                # If we failed to calculate the timestamp, remove all remaining splits for this competitor
                if len(sim_splits_df.loc[race_mask & competitor_mask & next_leg_mask, 'timestamp']) == 0:
                    splits_remaining -= len(sim_splits_df[race_mask & competitor_mask]) - (curr_leg_order + 1)

            # Update the split order for the next control sequence
            for ctrl_seq in sim_splits_df.loc[timestep_mask, 'ctrl_seq'].unique():
                split_order_dict[ctrl_seq] += len(
                    sim_splits_df[timestep_mask & (sim_splits_df['ctrl_seq'] == ctrl_seq)])

            # Subtract splits remaining. If no competitors remain, we can exit the loop
            splits_remaining -= len(sim_splits_df[timestep_mask])
            if splits_remaining <= 0:
                break

            # Increment the simulated race time
            race_time += 1

        print(f'Time to simulate race {race_id}: {time.time() - start_time} seconds')
        start_time = time.time()

    return sim_splits_df


if __name__ == '__main__':
    # Set parameters for initial data import
    min_start_time = 6 * 3600  # To remove competitors given artificial start times of midnight
    max_start_time = 11.75 * 3600  # To remove competitors with start times past the expected window
    random_classes_only = True

    # Import all database paths in folder using tkinter file dialog
    folder_path = filedialog.askdirectory()
    db_file_paths = utilities.find_db_files(folder_path)

    if len(db_file_paths) == 0:
        print('No database files found in folder')
        exit()

    # # Open folder to save simulated race data using tkinter file dialog
    # save_path = filedialog.askdirectory()
    #
    # if len(save_path) == 0:
    #     print('No save path selected')
    #     exit()

    # Set random seed for reproducibility
    np.random.seed(0)

    for db_file_path in db_file_paths:
        print(db_file_path)

        # Populate list of classes with randomly assigned start. Functions will use all classes when set to None
        if random_classes_only:
            class_list = utilities.get_random_classes(db_file_path, pull_from_db=True)
        else:
            class_list = None

        # Read in split performance data
        splits_df = utilities.import_all_splits(db_file_path, class_list=class_list, min_start_time=min_start_time,
                                                max_start_time=max_start_time)

        # Simulate split performance
        sim_splits = simulate_split_performance(splits_df)

        sim_splits.to_pickle(db_file_path[:-3] + '_sim_splits.pkl')
