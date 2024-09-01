"""
Plotting functions
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import matplotlib.dates as mdates

import utilities
import stats

def plot_split_perf_vs_time(df: pd.DataFrame, race_id, race_name, tz, normalize_to_individual=False, normalize_to_class=True,
                            alpha=0.2, y_min=None, y_max=None,
                            average_window=300, time_step=30, use_start_time=False):
    """
    Plots split performance against time (normalized or not)
    :param df: Split performances DataFrame
    :param race_id: Race ID from database
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
    averages_df = utilities.window_avg_line(df[df['race_id'] == race_id], perf_column, time_column=time_column,
                                            average_window=average_window, time_step=time_step)

    # Convert Unix timestamps to datetime objects, adjust for time zone difference
    tz_local_offset = utilities.get_offset_from_local(tz)
    times = [datetime.datetime.fromtimestamp(ts + tz_local_offset) for ts in df[df['race_id'] == race_id][time_column]]
    average_times = [datetime.datetime.fromtimestamp(ts + tz_local_offset) for ts in averages_df[time_column]]

    # Format the x-axis to show time in hh:mm:ss
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator())  # set the x-axis major locator to show every hour

    # Create the plot
    plt.plot(times, df[df['race_id'] == race_id][perf_column], '.', alpha=alpha, color='darkorange')
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

def plot_split_vs_order(df: pd.DataFrame, race_id, race_name, normalize_to_individual=False, normalize_to_class=True,
                        order_avg_window=5, order_avg_step=1, y_min=None, y_max=None, alpha=0.2, class_id=None,
                        x_max=None, plot_trend=False, trend_type='linear'):
    """
    This function plots the relative performance of a race split against the split order for a given race.

    :param df: A pandas DataFrame containing the race data.
    :param race_id: The ID of the race for which the plot is generated.
    :param race_name: The name of the race.
    :param normalize_to_individual: A boolean indicating whether the performance should be normalized to the individual.
    :param normalize_to_class: A boolean indicating whether the performance should be normalized to the class or all competitors.
    :param order_avg_window: The window size for calculating the moving average of the performance.
    :param order_avg_step: The step size for calculating the moving average of the performance.
    :param y_min: The minimum value for the y-axis.
    :param y_max: The maximum value for the y-axis.
    :param alpha: The transparency level of the race points on the plot.
    :param class_id: The ID of the class for which the plot is generated. Defaults to None, in which case all classes are used
    :param x_max: The maximum value for the x-axis.
    :param plot_trend: A boolean indicating whether the trend line should be plotted.
    :param trend_type: Type of trend to calculate. Defaults to 'linear'.
    :return: None
    """

    # Build columns
    perf_column = 'norm_' if normalize_to_individual else ''
    perf_column += 'class_perf' if normalize_to_class else 'overall_perf'

    # Define a mask to filter for race and class, if desired
    if class_id is not None:
        mask = (df['race_id'] == race_id) & (df['class_id'] == class_id)
    else:
        mask = (df['race_id'] == race_id)

    # Get average line
    averages_df = utilities.window_avg_line(df[mask], perf_column, time_column='split_order',
                                            average_window=order_avg_window, time_step=order_avg_step)

    # Create the plot
    plt.plot(df[mask]['split_order'], df[mask][perf_column], '.', alpha=alpha,
             color='darkorange')
    plt.plot(averages_df['split_order'], averages_df[f'{perf_column}_avg'], label=f'{order_avg_window}-width average', color='black')

    # Plot trend
    if plot_trend and class_id is not None:
        trend_params = stats.fit_split_perf_vs_order(df, race_id, class_id, normalize_to_class=normalize_to_class, order_cutoff=x_max)
        plt.plot(averages_df['split_order'], stats.trend(averages_df['split_order'], trend_params, trend_type), label='Trend', color='red')

    plt.xlabel('Split Order')
    plt.ylabel('Relative Performance (lower is better)')
    class_title_str = f', class {class_id}' if class_id is not None else ''
    plt.title(f'{race_name}{class_title_str}\n'
              f'{"Normalized " if normalize_to_individual else ""}Split Performance vs Split Order')
    plt.grid(True)
    plt.legend()

    # Set y_min/Y_max if desired
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)

    if x_max is not None:
        plt.xlim(0, x_max)

    plt.tight_layout()
    plt.show()


def plot_all_split_vs_order_trends(df: pd.DataFrame, race_id, race_name, normalize_to_class=True,
                        y_min=None, y_max=None, order_cutoff=None, alpha=0.2, trend_type='linear'):

    # Define a mask to filter by race
    mask = (df['race_id'] == race_id)

    # Get x range
    if order_cutoff is None:
        x_values = np.arange(0, df[mask]['split_order'].max())
    else:
        x_values = np.arange(0, order_cutoff)

    for class_id in df[mask]['class_id'].unique():
        trend_params = stats.fit_split_perf_vs_order(df, race_id, class_id, normalize_to_class=normalize_to_class,
                                                     order_cutoff=order_cutoff, trend_type=trend_type)
        plt.plot(x_values, stats.trend(x_values, trend_params, trend_type, normalize_to_0=True), '--', alpha=alpha)

    overall_trend_params = stats.fit_split_perf_vs_order_all_classes(df, race_id, normalize_to_class=normalize_to_class,
                                                                     order_cutoff=order_cutoff, trend_type=trend_type)
    plt.plot(x_values, stats.trend(x_values, overall_trend_params, trend_type, normalize_to_0=True), color='darkorange',
             linewidth=3,
             label='Overall Trend for All Classes')

    plt.xlabel('Split Order')
    plt.ylabel('Relative Performance (lower is better)')
    plt.title(f'{race_name}\n'
              f'Trends of Split Performance vs Split Order by Class')

    plt.grid(True)
    plt.legend()

    # Set y_min/Y_max if desired
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)

    if order_cutoff is not None:
        plt.xlim(0, order_cutoff)

    plt.tight_layout()
    plt.show()
