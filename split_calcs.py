"""
Functions to import and calculate performance of splits. Main function import splits and creates pickle files of the
splits performance dataframe.
"""

import ast
import sqlite3
from tkinter import filedialog
import pandas as pd
import time
import numpy as np

import utilities

def import_all_splits(db_filename: str, class_list: list[str] = None):
    """
    Imports all split times between two consecutive controls for a given race_id
    :param db_filename: Path to the database file
    :param class_list: List of classes to import. If empty, import all classes
    :return: Dataframe with all legs and splits times
    """

    func_start_time = time.time()

    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()

    # Create indices if they don't exist
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_card_no_race_id ON results(card_no, race_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_races_competitors_card_no_race_id ON races_competitors(card_no, race_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_races_competitors_class_id ON races_competitors(class_id)')
    conn.commit()

    query_start_time = time.time()

    query_base = (
        'SELECT results.race_id, start_time, controls, control_times, competitor_id, class_id FROM results INNER JOIN races_competitors '
        'ON results.card_no = races_competitors.card_no AND results.race_id = races_competitors.race_id')

    if class_list is not None:
        query = (f'{query_base} WHERE class_id IN ({",".join(["?"] * len(class_list))})')
        values = class_list
        cursor.execute(query, values)
    else:
        query = query_base
        cursor.execute(query)
    rows = cursor.fetchall()

    conn.close()

    query_end_time = time.time()
    print('Time to run split import query: ' + str(query_end_time - query_start_time))
    print(f'Fetched {len(rows)} rows')

    splits_dict_list = []

    for row in rows:
        race_id = row[0]
        start_time = row[1]
        controls = ast.literal_eval(row[2])
        control_times = ast.literal_eval(row[3])
        competitor_id = row[4]
        class_id = row[5]

        for i in range(len(controls) - 1):
            control_sequence = f'{controls[i]}-{controls[i + 1]}'
            try:
                splits_dict = {'race_id': race_id,
                               'ctrl_seq': control_sequence,
                               'timestamp': control_times[i],
                               'start_time': start_time,
                               'competitor_id': competitor_id,
                               'class_id': class_id,
                               'split_time': control_times[i + 1] - control_times[i], }
                splits_dict_list.append(splits_dict)
            except TypeError:
                continue  # Some of the control times are NULL for mispunches

    splits_df = pd.DataFrame(splits_dict_list)

    # Drop any rows where split_time is negative or greater than 5 hours, this is likely invalid data
    splits_df = splits_df[splits_df['split_time'] >= 0]
    splits_df = splits_df[splits_df['split_time'] <= 5 * 3600]

    func_end_time = time.time()
    print('Total time to import splits: ' + str(func_end_time - func_start_time))

    return splits_df


def split_order_arbitrary_time_col(split_perf_df: pd.DataFrame, arbitrary_time_col: str, order_col_name: str):
    """
    Calculate the split order based on an arbitrary time column
    :param split_perf_df: DataFrame with splits
    :param arbitrary_time_col: Column to use for split order
    :param order_col_name: Name of the new column with split order
    :return: DataFrame with split order
    """

    # Create separate DataFrames for 'timestamp' and arbitrary_time_col
    timestamp_df = split_perf_df[['timestamp', 'race_id', 'ctrl_seq']].copy()
    timestamp_df['time'] = timestamp_df['timestamp']
    timestamp_df[arbitrary_time_col] = None

    arb_col_df = split_perf_df[[arbitrary_time_col, 'race_id', 'ctrl_seq']].copy()
    arb_col_df[f'{arbitrary_time_col}_index'] = arb_col_df.index
    arb_col_df['time'] = arb_col_df[arbitrary_time_col]
    arb_col_df['timestamp'] = None

    # Concatenate the DataFrames
    combined_df = pd.concat([timestamp_df, arb_col_df], ignore_index=True)

    # Calculate order of arbitrary times and timestamps combined
    combined_df['split_order_combined'] = combined_df.groupby(['race_id', 'ctrl_seq'])['time'].rank(method='min') - 1

    # Drop all rows where arbitrary_time_col is None and timestamp column
    combined_df = combined_df.dropna(subset=[arbitrary_time_col])
    combined_df.drop(columns=['timestamp'], inplace=True)

    # Merge the combined DataFrame with the original DataFrame based on the index
    split_perf_df = split_perf_df.merge(combined_df[[f'{arbitrary_time_col}_index', 'split_order_combined']], left_index=True,
                                        right_on=f'{arbitrary_time_col}_index')

    # Calculate split order based on arbitrary column alone
    split_perf_df[f'split_order_{arbitrary_time_col}'] = split_perf_df.groupby(['race_id', 'ctrl_seq'])[arbitrary_time_col].rank(
        method='min') - 1

    # Calculate the split order at for the arbitrary time column
    split_perf_df[order_col_name] = split_perf_df['split_order_combined'] - split_perf_df[f'split_order_{arbitrary_time_col}']

    # Drop columns that are no longer needed
    split_perf_df.drop(columns=[f'{arbitrary_time_col}_index', 'split_order_combined', f'split_order_{arbitrary_time_col}'], inplace=True)
    
    return split_perf_df


def calc_split_performances(splits_df: pd.DataFrame, min_start_time=0, max_start_time=86400):
    """
    Gets normalized split performances for all competitors from a splits dataframe
    :param splits_df: Dictionary with all split times
    :param min_start_time: Minimum start time of competitors (UTC time of day in seconds)
    :param max_start_time: Maximum start time of competitors (UTC time of day in seconds)
    :return: DataFrame with performances, normalized performances, and timestamps. Also return random classes
    """

    func_start_time = time.time()

    split_perf_columns = ['race_id', 'ctrl_seq', 'timestamp', 'start_time', 'class_id', 'class_perf', 'overall_perf',
                          'norm_class_perf',
                          'norm_overall_perf']

    split_perf_df = pd.DataFrame(columns=split_perf_columns)

    # Masks
    start_time_mask = ((splits_df['start_time'] != None)
                       & (splits_df['start_time'] % 86400 >= min_start_time)
                       & (splits_df['start_time'] % 86400 <= max_start_time))

    # Copy over existing columns from splits df, passing through mask
    split_perf_df['race_id'] = splits_df['race_id'][start_time_mask]
    split_perf_df['ctrl_seq'] = splits_df['ctrl_seq'][start_time_mask]
    split_perf_df['class_id'] = splits_df['class_id'][start_time_mask]
    split_perf_df['timestamp'] = splits_df['timestamp'][start_time_mask]
    split_perf_df['start_time'] = splits_df['start_time'][start_time_mask]
    split_perf_df['competitor_id'] = splits_df['competitor_id'][start_time_mask]

    # ************************************************************************
    # Calculate performance by class and overall, normalized to mean
    # ************************************************************************

    split_perf_df['class_perf'] = splits_df.groupby(['race_id', 'ctrl_seq', 'class_id'])['split_time'].transform(
        lambda x: (x / x.mean()))
    split_perf_df['overall_perf'] = splits_df.groupby(['race_id', 'ctrl_seq'])['split_time'].transform(
        lambda x: (x / x.mean()))

    # ************************************************************************
    # Calculate performance normalized to the individual, normalized to mean
    # ************************************************************************

    split_perf_df['norm_class_perf'] = split_perf_df.groupby(['race_id', 'competitor_id'])['class_perf'].transform(
        lambda x: (x / x.mean()))
    split_perf_df['norm_overall_perf'] = split_perf_df.groupby(['race_id', 'competitor_id'])['overall_perf'].transform(
        lambda x: (x / x.mean()))

    # ************************************************************************
    # Calculate split_order
    # ************************************************************************

    split_perf_df['split_order'] = split_perf_df.groupby(['race_id', 'ctrl_seq'])['timestamp'].rank(method='min') - 1

    # ************************************************************************
    # Calculate split_order_at_start
    # ************************************************************************

    split_perf_df = split_order_arbitrary_time_col(split_perf_df, 'start_time', 'split_order_at_start')

    # ************************************************************************
    # Calculate split_order_at_exp_timestamp
    # ************************************************************************

    # Calculate time to get to the split for each competitor
    split_perf_df['time_to_split'] = split_perf_df['timestamp'] - split_perf_df['start_time']

    # Calculate expected timestamp of arrival at control based on start time and average time to reach the split within the class
    split_perf_df['exp_timestamp'] = split_perf_df['start_time'] + split_perf_df.groupby(['race_id', 'ctrl_seq', 'class_id'])['time_to_split'].transform(
        lambda x: x.mean())

    # Calculate the order of competitors at the expected timestamp
    split_perf_df = split_order_arbitrary_time_col(split_perf_df, 'exp_timestamp', 'split_order_at_exp_timestamp')



    func_end_time = time.time()
    print('Time to calculate all split performances: ' + str(func_end_time - func_start_time))

    return split_perf_df


if __name__ == '__main__':
    # Import all database paths
    folder_path = filedialog.askdirectory()
    db_file_paths = utilities.find_db_files(folder_path)

    min_start_time = 6 * 3600  # To remove competitors given artificial start times of midnight
    max_start_time = 11.75 * 3600 # To remove competitors with start times past the expected window
    random_classes_only = True

    for db_file_path in db_file_paths:
        print(db_file_path)

        # Populate list of classes with randomly assigned start. Functions will use all classes when set to None
        if random_classes_only:
            class_list = utilities.get_random_classes(db_file_path, pull_from_db=True)
        else:
            class_list = None

        splits_df = import_all_splits(db_file_path, class_list=class_list)
        split_performances = calc_split_performances(splits_df, min_start_time=min_start_time, max_start_time=max_start_time)

        splits_df.to_pickle(db_file_path[:-3] + '_splits.pkl')
        split_performances.to_pickle(db_file_path[:-3] + '_splits_perf.pkl')
