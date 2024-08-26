"""
A collection of utlity functions
"""

import time
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np


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
    time_series = np.arange(start_time + average_window / 2, end_time - average_window / 2, time_step)

    averages_df = pd.DataFrame(columns=[time_column, f'{column_to_avg}_avg'])
    for timestamp in time_series:
        average_mask = (df[time_column] > timestamp - average_window / 2) & (
                    df[time_column] < timestamp + average_window / 2)
        average = np.mean(df[column_to_avg][average_mask])
        averages_df.loc[len(averages_df)] = [timestamp, average]

    return averages_df