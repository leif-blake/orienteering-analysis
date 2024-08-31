"""
Plotting functions
"""

import pandas as pd
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates

import utilities


def plot_split_performances(df: pd.DataFrame, race_name, tz, normalize_to_individual=False, normalize_to_class=True,
                            alpha=0.2, y_min=None, y_max=None,
                            average_window=300, time_step=30, use_start_time=False):
    """
    Plots split performance against time (normalized or not)
    :param df: Split performances DataFrame
    :param race_name: Name of the race
    :param tz: Timezone of the race
    :param normalize_to_individual: Whether to normalize the split performance for an individual athlete or not
    :param normalize_to_class: Whether to normalize the split performance to the class or overall
    :param alpha: Alpha parameter for plotting individual split performance
    :param y_min: Minimum y value for y axis
    :param y_max: Maximum y value for y axis
    :param average_window: Averaging window width for line in seconds
    :param use_start_time: When True, use start time for x axis
    :return:
    """

    # Build columns
    perf_column = 'norm_' if normalize_to_individual else ''
    perf_column += 'class_perf' if normalize_to_class else 'overall_perf'
    time_column = 'start_time' if use_start_time else 'timestamp'

    # Get average line
    averages_df = utilities.window_avg_line(df, perf_column, time_column=time_column,
                                            average_window=average_window, time_step=time_step)

    # Convert Unix timestamps to datetime objects, adjust for time zone difference
    tz_local_offset = utilities.get_offset_from_local(tz)
    times = [datetime.datetime.fromtimestamp(ts + tz_local_offset) for ts in df[time_column]]
    average_times = [datetime.datetime.fromtimestamp(ts + tz_local_offset) for ts in averages_df[time_column]]

    # Format the x-axis to show time in hh:mm:ss
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator())  # set the x-axis major locator to show every hour

    # Create the plot
    plt.plot(times, df[perf_column], '.', alpha=alpha, color='darkorange')
    plt.plot(average_times, averages_df[f'{perf_column}_avg'], color='black')

    plt.xlabel(f'{"Athlete Start Time" if use_start_time else "Time"} (hh:mm:ss)')
    plt.ylabel('Relative Performance (lower is better)')
    plt.title(f'{race_name}\n'
              f'{"Normalized " if normalize_to_individual else ""}Split Performance vs {"Athlete Start " if use_start_time else ""}Time\n'
              f'Line is moving average with {average_window / 60} minute width')

    plt.grid(True)

    # Set y_min/Y_max if desired
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()
