"""
Produces plots which show the relative performance of runners' split times over the course of a race
"""

import ast
import sqlite3
from tkinter import filedialog
import numpy as np
import pandas as pd
import random

import plotter

def get_random_classes(db_filename):
    """
    Gets class ids of all random classes in database
    :param db_filename:
    :return:
    """

    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()

    # Get all class ids from database
    cursor.execute('SELECT id FROM classes ORDER BY id ASC')
    class_id_rows = cursor.fetchall()

    # Get all start times, associated with each athlete
    query = ('SELECT results.race_id, competitor_id, start_time, class_id FROM results '
             'INNER JOIN races_competitors ON results.card_no = races_competitors.card_no AND results.race_id = races_competitors.race_id ')
    start_time_df = pd.read_sql_query(query, conn)

    conn.close()

    random_classes = []
    for row in class_id_rows:
        if is_random_class(start_time_df, row[0]):
            random_classes.append(row[0])

    return random_classes


def is_random_class(start_time_df, class_id, num_iter=500, threshold=0.1):
    """
    Determines if class start times are randomly assigned across a multi-day event
    :param class_id: Class id in Database
    :return: True or False
    """

    # Retrieve class-specific start times
    start_time_df_class = start_time_df[start_time_df['class_id'] == class_id]

    # Calculate the midpoint of the start window for each day
    mid_start_times = {}
    race_ids = start_time_df_class['race_id'].unique()
    for race_id in race_ids:
        start_times = start_time_df_class[start_time_df_class['race_id'] == race_id]['start_time']
        mid_start_times[race_id] = np.average(start_times)

    # Calculate probability that a person is in the same "half" of the start window on two separate days
    sum_prob = 0
    prob_count = 0
    for i in range(num_iter):
        competitor_id = random.choice(list(start_time_df_class['competitor_id'].unique()))
        rand_race_ids = random.sample(list(race_ids), 2)
        try:
            start_time_1 = start_time_df_class[(start_time_df_class['competitor_id'] == competitor_id)
                                               & (start_time_df_class['race_id'] == rand_race_ids[0])]['start_time']
            start_time_2 = start_time_df_class[(start_time_df_class['competitor_id'] == competitor_id)
                                               & (start_time_df_class['race_id'] == rand_race_ids[1])]['start_time']
            is_first_half_race_1 = 1 if start_time_1.iloc[0] < mid_start_times[rand_race_ids[0]] else 0
            is_first_half_race_2 = 1 if start_time_2.iloc[0] < mid_start_times[rand_race_ids[1]] else 0
        except TypeError:
            continue
        except IndexError:
            continue
        prob_count += 1
        if is_first_half_race_1 == is_first_half_race_2:
            sum_prob += 1

    prob_same_half = sum_prob / prob_count

    if prob_same_half > 0.5 + threshold:
        return False

    return True


def import_all_splits(db_filename, race_id, random_classes_only=False, random_classes=[]):
    """
    Imports all split times between two consecutive controls for a given race_id
    :param db_filename: Path to the database file
    :param race_id: Race ID from database
    :return: Dictionary with all legs and splits times, as well as standard deviation. Also return random classes
    """

    # If flag is set, only use classes where start times are randomly assigned. Optionally get list from argument
    if random_classes_only and random_classes == []:
        random_classes = get_random_classes(db_filename)

    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()

    if random_classes_only:
        query = f'SELECT controls, control_times FROM results WHERE race_id = ? AND class_id IN ({",".join(["?"]*len(random_classes))})'
        values = [race_id] + random_classes
        cursor.execute(query, values)
    else:
        cursor.execute('SELECT controls, control_times FROM results WHERE race_id = ?', (race_id,))
    rows = cursor.fetchall()

    dict = {}
    for row in rows:
        controls = ast.literal_eval(row[0])
        control_times = ast.literal_eval(row[1])

        for i in range(len(controls) - 1):
            control_sequence = f'{controls[i]}-{controls[i + 1]}'
            if control_sequence not in dict:
                dict[control_sequence] = {}
                dict[control_sequence]['times'] = []
            try:
                dict[control_sequence]['times'].append(control_times[i + 1] - control_times[i])
            except TypeError:
                continue  # Some of the control times are NULL for mispunches

    sequences_to_remove = []
    for control_sequence, sub_dict in dict.items():
        # Remove control times if empty
        if len(sub_dict['times']) == 0:
            sequences_to_remove.append(control_sequence)
            continue
        sub_dict['avg'] = np.average(sub_dict['times'])
        sub_dict['dev'] = np.std(sub_dict['times'])

    for control_sequence in sequences_to_remove:
        del dict[control_sequence]

    conn.close()

    return dict, random_classes


def calc_split_performance(start_time, controls, control_times, splits_dict):
    """
    Calculates the per-split performance of a given competitor, relative to their average split performance
    :param start_time: Start time of the competitor
    :param controls: List of controls
    :param control_times: List of control times
    :param splits_dict: Imported dictionary of all split times
    :return: DataFrame with performances, normalized performances, and timestamps
    """

    # Calculate split performance of competitor relative to all other competitors
    split_performances = pd.DataFrame(columns=['start_time', 'timestamp', 'performance'])
    for i in range(len(controls) - 1):
        control_sequence = f'{controls[i]}-{controls[i + 1]}'
        try:
            split_time = control_times[i + 1] - control_times[i]
        except TypeError:
            continue  # Some of the control times are NULL for mispunches

        split_timestamp = (control_times[i] + control_times[i + 1]) / 2
        split_performance = split_time / splits_dict[control_sequence]['avg']
        split_performances.loc[len(split_performances)] = [start_time, split_timestamp, split_performance]

    # Return empty if no valid splits were found
    if split_performances.empty:
        return split_performances

    split_performances['normalized_performance'] = split_performances['performance'] / np.mean(
        split_performances['performance'])

    return split_performances


def get_all_split_performances(db_filename, race_id, splits_dict, min_start_time=0, random_classes_only=False, random_classes=[]):
    """
    Gets normalized split performances for all competitors, removing reference to the competitors and control sequences
    :param db_filename: Database file path
    :param race_id: Race ID from database
    :return: DataFrame with performances, normalized performances, and timestamps. Also return random classes
    """

    # If flag is set, only use classes where start times are randomly assigned. Optionally get list from argument
    if random_classes_only and random_classes == []:
        random_classes = get_random_classes(db_filename)

    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()

    if random_classes_only:
        query = f'SELECT start_time, controls, control_times FROM results WHERE race_id = ? AND class_id IN ({",".join(["?"]*len(random_classes))})'
        values = [race_id] + random_classes
        cursor.execute(query, values)
    else:
        cursor.execute('SELECT start_time, controls, control_times FROM results WHERE race_id = ?', (race_id,))

    split_performances = pd.DataFrame(columns=['start_time', 'timestamp', 'performance', 'normalized_performance'])
    for row in cursor.fetchall():
        # Check for minimum start time requirement (in UTC)
        if row[0] is None or int(row[0]) % 86400 < min_start_time:
            continue
        df = calc_split_performance(int(row[0]), ast.literal_eval(row[1]), ast.literal_eval(row[2]), splits_dict)
        if split_performances.empty:
            split_performances = df.copy()
        elif not df.empty:
            split_performances = pd.concat([split_performances, df], ignore_index=True)

    split_performances.reset_index(inplace=True)

    conn.close()

    return split_performances, random_classes


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

    min_start_time = 6 * 3600  # To remove competitors given artificial start times of midnight
    average_window = 60*30
    y_min_zoom = 0.6
    y_max_zoom = 1.4
    use_start_time = False
    random_classes_only = True

    # race_ids = [1, 2, 3, 4]
    race_ids = [1]
    # race_ids = [3]
    for race_id in race_ids:
        cursor.execute('SELECT name FROM races WHERE id = ?', (race_id,))
        race_name = cursor.fetchone()[0]
        splits_dict, random_classes = import_all_splits(db_file_path, race_id, random_classes_only=random_classes_only)
        split_performances, _ = get_all_split_performances(db_file_path, race_id, splits_dict,
                                                        min_start_time=min_start_time,
                                                        random_classes_only=random_classes_only,
                                                        random_classes=random_classes)
        plotter.plot_split_performances(split_performances, race_name, tz, average_window=average_window,
                                use_start_time=use_start_time)
        plotter.plot_split_performances(split_performances, race_name, tz, y_min=y_min_zoom, y_max=y_max_zoom,
                                average_window=average_window, use_start_time=use_start_time)
        plotter.plot_split_performances(split_performances, race_name, tz, normalized=False, average_window=average_window,
                                use_start_time=use_start_time)
        plotter.plot_split_performances(split_performances, race_name, tz, normalized=False, y_min=y_min_zoom, y_max=y_max_zoom,
                                average_window=average_window, use_start_time=use_start_time)

    print("Done")
