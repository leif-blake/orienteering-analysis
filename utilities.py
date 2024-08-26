"""
A collection of utlity functions
"""

import time
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import sqlite3
import random


def get_offset_from_local(tz: float):
    """
    Returns the offset in seconds of the given timezone from local time
    :param tz:
    :return:
    """

    # Get the current time in UTC
    utc_time = time.gmtime()

    # Get the current time in local time
    local_time = time.localtime()

    # Convert both to timestamps
    utc_timestamp = time.mktime(utc_time)
    local_timestamp = time.mktime(local_time)

    # Calculate the difference to get the offset in seconds
    local_offset = local_timestamp - utc_timestamp

    tz_utc_offset = tz * 3600

    return int(tz_utc_offset - local_offset)


def hhmmss_to_seconds(time_str):
    """
    Converts a hh:mm:ss string to seconds.
    :param time_str: Input time string
    :return: Seconds as an integer
    """
    if time_str is None:
        return None

    # Split the string into hours, minutes, and seconds
    h, m, s = map(int, time_str.split(':'))

    # Convert to seconds
    total_seconds = h * 3600 + m * 60 + s
    return total_seconds


def get_element_text_or_none(parent: ET.Element, key: str):
    """
    Returns the text from an element, or None if it doesn't exist.
    :param parent: Parent XML element
    :param key: Key String for child element
    :return:
    """
    attribute = parent.find(key)
    if attribute is None:
        return None
    return attribute.text


def select_id_query_match_null(table, columns: list, values: list):
    """
    Build a select query for table id that matches NoneType to NULL
    :param table: Table name
    :param columns: Table columns in sqlite
    :param values: Values to match in columns
    :return: SELECT query string that matches ass columns to values
    """

    condition_strings = [column + (' IS NULL' if value is None else f' = ?') for column, value in zip(columns, values)]
    query_string = f'SELECT id FROM {table} WHERE {" AND ".join(condition_strings)}'

    non_null_values = []
    for value in values:
        if value is not None:
            non_null_values.append(value)

    return query_string, tuple(non_null_values)


def window_avg_line(df, column_to_avg, time_column='timestamp', average_window=300, time_step=30):
    """
    Calculates a line based on a windowed average
    :param df: DataFrame containing time-series data to average
    :param column_to_avg: Name of the column to average
    :param time_column: Name of the time column
    :param average_window: Averaging window in seconds
    :param time_step: Time step of line in seconds
    :return:
    """

    start_time = np.min(df[time_column])
    end_time = np.max(df[time_column])
    time_series = np.arange(start_time + average_window // 2, end_time - average_window // 2, time_step)

    averages_df = pd.DataFrame(columns=[time_column, f'{column_to_avg}_avg'])
    for timestamp in time_series:
        average_mask = (df[time_column] > timestamp - average_window / 2) & (
                    df[time_column] < timestamp + average_window / 2)
        average = np.mean(df[column_to_avg][average_mask])
        averages_df.loc[len(averages_df)] = [timestamp, average]

    return averages_df

def get_random_classes(db_filename, pull_from_db=False, write_to_db=False):
    """
    Gets class ids of all random classes in database. Checks whether a user is more likely to start late on one day
    given that they started late on another.
    :param db_filename: Database filename
    :param pull_from_db: Pull list of random classes from database using is_random column if True
    :param write_to_db: Write is_random columns to database if True
    :return: List of random classes (class ids)
    """

    random_classes = []
    non_random_classes = []

    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()

    # Get all class ids from database
    cursor.execute('SELECT id, is_random FROM classes ORDER BY id ASC')
    class_id_rows = cursor.fetchall()

    if pull_from_db:
        cursor.execute('SELECT id, is_random FROM classes ORDER BY id ASC')
        class_id_rows = cursor.fetchall()

        for row in class_id_rows:
            if row[1] == 1:
                random_classes.append(row[0])
            else:
                non_random_classes.append(row[0])

    else:
        # Get all start times, associated with each athlete
        query = ('SELECT results.race_id, competitor_id, start_time, class_id FROM results '
                 'INNER JOIN races_competitors ON results.card_no = races_competitors.card_no AND results.race_id = races_competitors.race_id ')
        start_time_df = pd.read_sql_query(query, conn)

        for row in class_id_rows:
            if is_random_class(start_time_df, row[0]):
                random_classes.append(row[0])
            else:
                non_random_classes.append(row[0])

    if write_to_db:
        query = (f'UPDATE classes SET is_random = 1 WHERE id IN ({",".join(["?"]*len(random_classes))})')
        cursor.execute(query, random_classes)
        query = (f'UPDATE classes SET is_random = 0 WHERE id IN ({",".join(["?"]*len(non_random_classes))})')
        cursor.execute(query, non_random_classes)
        conn.commit()

    conn.close()

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