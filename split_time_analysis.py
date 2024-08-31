"""
Produces plots which show the relative performance of runners' split times over the course of a race
"""

import ast
import sqlite3
from tkinter import filedialog
import numpy as np
import pandas as pd
import time

import plotter
import utilities


def import_all_splits(db_filename: str, race_id: str, class_list: list[str] = None):
    """
    Imports all split times between two consecutive controls for a given race_id
    :param db_filename: Path to the database file
    :param race_id: Race ID from database
    :param class_list: List of classes to import. If empty, import all classes
    :return: Dataframe with all legs and splits times
    """

    func_start_time = time.time()

    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()

    query_start_time = time.time()

    if class_list is not None:
        query = (
            f'SELECT start_time, controls, control_times, competitor_id, class_id FROM results INNER JOIN races_competitors '
            f'ON results.card_no = races_competitors.card_no AND results.race_id = races_competitors.race_id '
            f'WHERE results.race_id = ? AND class_id IN ({",".join(["?"] * len(class_list))})')
        values = [race_id] + class_list
        cursor.execute(query, values)
    else:
        query = query = (
            f'SELECT start_time, controls, control_times, competitor_id, class_id FROM results INNER JOIN races_competitors '
            f'ON results.card_no = races_competitors.card_no AND results.race_id = races_competitors.race_id '
            f'WHERE results.race_id = ?')
        cursor.execute(query, (race_id,))
    rows = cursor.fetchall()

    conn.close()

    query_end_time = time.time()
    print('Time to run split import query: ' + str(query_end_time - query_start_time))
    print(f'Fetched {len(rows)} rows')

    splits_dict_list = []

    for row in rows:
        start_time = row[0]
        controls = ast.literal_eval(row[1])
        control_times = ast.literal_eval(row[2])
        competitor_id = row[3]
        class_id = row[4]

        for i in range(len(controls) - 1):
            control_sequence = f'{controls[i]}-{controls[i + 1]}'
            try:
                splits_dict = {'ctrl_seq': control_sequence,
                               'timestamp': control_times[i],
                               'start_time': start_time,
                               'competitor_id': competitor_id,
                               'class_id': class_id,
                               'split_time': control_times[i + 1] - control_times[i],}
                splits_dict_list.append(splits_dict)
            except TypeError:
                continue  # Some of the control times are NULL for mispunches

    splits_df = pd.DataFrame(splits_dict_list)

    func_end_time = time.time()
    print('Total time to import splits: ' + str(func_end_time - func_start_time))

    return splits_df


def calc_split_performances(splits_df, min_start_time=0, class_list: list[str] = None):
    """
    Gets normalized split performances for all competitors from a splits dataframe
    :param db_filename: Database file path
    :param race_id: Race ID from database
    :param splits_df: Dictionary with all split times
    :param min_start_time: Minimum start time of competitors
    :param class_list: List of classes to import. If empty, import all classes
    :param normalize_to_class: Normalize split performances to average within class, not across all classes
    :return: DataFrame with performances, normalized performances, and timestamps. Also return random classes
    """

    func_start_time = time.time()

    split_perf_columns = ['ctrl_seq', 'timestamp', 'start_time', 'class_id', 'class_perf', 'overall_perf', 'norm_class_perf',
                          'norm_overall_perf']

    split_perf_df = pd.DataFrame(columns=split_perf_columns)

    # Masks
    start_time_mask = (splits_df['start_time'] != None) & (splits_df['start_time'] % 86400 >= min_start_time)
    if class_list is not None:
        class_mask = splits_df['class_id'].isin(class_list)
    else:
        class_mask = np.ones_like(splits_df['class_id'][start_time_mask], dtype=bool)

    # Copy over existing columns from splits df, passing through mask
    split_perf_df['ctrl_seq'] = splits_df['ctrl_seq'][start_time_mask][class_mask]
    split_perf_df['class_id'] = splits_df['class_id'][start_time_mask][class_mask]
    split_perf_df['timestamp'] = splits_df['timestamp'][start_time_mask][class_mask]
    split_perf_df['start_time'] = splits_df['start_time'][start_time_mask][class_mask]
    split_perf_df['competitor_id'] = splits_df['competitor_id'][start_time_mask][class_mask]

    # Calculate performance by class and overall
    split_perf_df['class_perf'] = splits_df.groupby(['ctrl_seq', 'class_id'])['split_time'].transform(
        lambda x: (x / x.mean()))
    split_perf_df['overall_perf'] = splits_df.groupby('ctrl_seq')['split_time'].transform(
        lambda x: (x / x.mean()))

    # Calculate performance normalized to the individual
    split_perf_df['norm_class_perf'] = split_perf_df.groupby('competitor_id')['class_perf'].transform(
        lambda x: (x / x.mean()))
    split_perf_df['norm_overall_perf'] = split_perf_df.groupby('competitor_id')['overall_perf'].transform(
        lambda x: (x / x.mean()))

    func_end_time = time.time()
    print('Time to calculate all split performances: ' + str(func_end_time - func_start_time))

    return split_perf_df


if __name__ == '__main__':
    # Open a file dialog to save the resulting database
    db_file_path = filedialog.askopenfilename(
        title="Open Event Database",
        defaultextension=".db",
        filetypes=[("SQLite3 Database", "*.db"), ("All Files", "*.*")]
    )

    # tz = -4 # Ontario Summer time
    tz = +2  # Sweden Summer time

    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()

    min_start_time = 7 * 3600  # To remove competitors given artificial start times of midnight
    average_window = 60 * 30
    y_min_zoom = 0.6
    y_max_zoom = 1.4
    use_start_time = True
    random_classes_only = True

    # Populate list of classes with randomly assigned start. Functions will use all classes when set to None
    if random_classes_only:
        class_list = utilities.get_random_classes(db_file_path, pull_from_db=True)
    else:
        class_list = None

    # race_ids = [1, 2, 3, 4]
    race_ids = [4]
    # race_ids = [3]
    for race_id in race_ids:
        cursor.execute('SELECT name FROM races WHERE id = ?', (race_id,))
        race_name = cursor.fetchone()[0]
        splits_df = import_all_splits(db_file_path, race_id, class_list=class_list)
        split_performances = calc_split_performances(splits_df, min_start_time=min_start_time,
                                                     class_list=class_list)
        plotter.plot_split_performances(split_performances, race_name, tz, average_window=average_window,
                                use_start_time=use_start_time)
        plotter.plot_split_performances(split_performances, race_name, tz, y_min=y_min_zoom,
                                        y_max=y_max_zoom,
                                        average_window=average_window, use_start_time=use_start_time)

    print("Done")
