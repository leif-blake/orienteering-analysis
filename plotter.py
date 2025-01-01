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
import trends


def plot_split_perf_vs_time(df: pd.DataFrame, race_id, race_name, tz, normalize_to_individual=False,
                            normalize_to_class=True, alpha=0.2, y_min=None, y_max=None, average_window=300,
                            time_step=30, use_start_time=False, highlight_class_id=None, order_cutoff=None):
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
    :param time_step: Time step for average line in seconds
    :param use_start_time: When True, use start time for x axis
    :param highlight_class_id: Highlight data points from one class in a different color
    :param order_cutoff: Maximum value for order of splits to plot
    :return:
    """

    # Build columns
    perf_column = 'norm_' if normalize_to_individual else ''
    perf_column += 'class_perf' if normalize_to_class else 'overall_perf'
    time_column = 'start_time' if use_start_time else 'timestamp'

    # Race mask
    if order_cutoff is not None:
        race_mask = (df['race_id'] == race_id) & (df['split_order'] <= order_cutoff)
    else:
        race_mask = df['race_id'] == race_id

    # Get average line
    averages_df = utilities.window_avg_line(df[race_mask], perf_column, time_column=time_column,
                                            average_window=average_window, time_step=time_step)

    # For adjusting time to local race time
    tz_offset = tz * 3600.0

    # Format the x-axis to show time in hh:mm:ss
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator())  # set the x-axis major locator to show every hour

    # Create the plot
    if highlight_class_id is not None:
        class_mask = df['class_id'] == highlight_class_id
        plt.plot(pd.to_datetime(df[race_mask & ~class_mask][time_column] + tz_offset, unit='s', utc=True),
                 df[race_mask & ~class_mask][perf_column], '.', alpha=alpha, color='darkorange')
        plt.plot(pd.to_datetime(df[race_mask & class_mask][time_column] + tz_offset, unit='s', utc=True),
                 df[race_mask & class_mask][perf_column], '.', color='blue', alpha=0.5,
                 label=f'Class {highlight_class_id}')
    else:
        plt.plot(pd.to_datetime(df[race_mask][time_column] + tz_offset, unit='s', utc=True),
                 df[race_mask][perf_column], '.', alpha=alpha, color='darkorange')

    # Plot windowed average line
    plt.plot(pd.to_datetime(averages_df[time_column] + tz_offset, unit='s', utc=True),
             averages_df[f'{perf_column}_avg'], color='black', label=f'{average_window // 60}-minute width average')

    plt.xlabel(f'{"Athlete Start Time" if use_start_time else "Time"} (hh:mm:ss)')
    plt.ylabel('Relative Performance (lower is better)')
    plt.title(f'{race_name}\n'
              f'{"Normalized " if normalize_to_individual else ""}Split Performance vs '
              f'{"Athlete Start " if use_start_time else ""}Time')

    plt.grid(True)
    plt.legend()

    # Set y_min/Y_max if desired
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()


def plot_split_vs_order(df: pd.DataFrame, race_id, race_name, normalize_to_individual=False, normalize_to_class=True,
                        order_avg_window=5, order_avg_step=1, y_min=None, y_max=None, alpha=0.2, class_id=None,
                        order_cutoff=None, plot_trend=False, trend_type='linear', highlight_class_id=None, split_order_col='split_order'):
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
    :param order_cutoff: The maximum value for split orders to plot.
    :param plot_trend: A boolean indicating whether the trend line should be plotted.
    :param trend_type: Type of trend to calculate. Defaults to 'linear'.
    :param highlight_class_id: Highlight data points from one class in a different color.
    :param split_order_col: The column name for the split order. Defaults to 'split_order'.
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
    averages_df = utilities.window_avg_line(df[mask], perf_column, time_column=split_order_col,
                                            average_window=order_avg_window, time_step=order_avg_step)

    # Create the plot
    if class_id is None and highlight_class_id is not None:
        class_mask = df['class_id'] == highlight_class_id
        plt.plot(df[mask & ~class_mask][split_order_col], df[mask & ~class_mask][perf_column], '.', alpha=alpha,
                 color='darkorange')
        plt.plot(df[mask & class_mask][split_order_col], df[mask & class_mask][perf_column], '.', color='blue', alpha=0.5,
                 label=f'Class {highlight_class_id}')
    else:
        plt.plot(df[mask][split_order_col], df[mask][perf_column], '.', alpha=alpha,
                 color='darkorange')
    plt.plot(averages_df[split_order_col], averages_df[f'{perf_column}_avg'], label=f'{order_avg_window}-width average',
             color='black')

    # Plot trend
    if plot_trend:
        if class_id is None:
            trend_params = stats.fit_split_perf_vs_order(df, race_id,
                                                               normalize_to_class=normalize_to_class,
                                                               order_cutoff=order_cutoff, split_order_col=split_order_col,
                                                                     trend_type=trend_type)
        else:
            trend_params = stats.fit_split_perf_vs_order_class(df, race_id, class_id, normalize_to_class=normalize_to_class,
                                                               order_cutoff=order_cutoff, trend_type=trend_type)
        plt.plot(averages_df[split_order_col], trends.trend(averages_df[split_order_col], trend_params, trend_type),
                 label='Trend', color='red')

    # Select split order title on x axis
    if split_order_col == 'split_order':
        plt.xlabel('Split Order')
    elif split_order_col == 'split_order_at_exp_timestamp':
        plt.xlabel('Split Order at Expected Time at Leg')
    elif split_order_col == 'split_order_at_start':
        plt.xlabel('Split Order at Start')
    else:
        plt.xlabel(f'Split Order (from column {split_order_col})')

    plt.xlabel(f'Split Order (from column {split_order_col})')
    plt.ylabel('Relative Performance (lower is better)')
    class_title_str = f', class {class_id}' if class_id is not None else ''
    plt.title(f'{race_name}{class_title_str}\n'
              f'{"Normalized " if normalize_to_individual else ""}Split Performance vs Split Order')
    plt.grid(True)
    plt.legend()

    # Set y_min/Y_max if desired
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)

    if order_cutoff is not None:
        plt.xlim(0, order_cutoff)

    plt.tight_layout()
    plt.show()

    if plot_trend:
        return trend_params
    else:
        return None


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
        trend_params = stats.fit_split_perf_vs_order_class(df, race_id, class_id, normalize_to_class=normalize_to_class,
                                                           order_cutoff=order_cutoff, trend_type=trend_type)
        plt.plot(x_values, trends.trend(x_values, trend_params, trend_type, normalize_to_0=True), '--', alpha=alpha)

    overall_trend_params = stats.fit_split_perf_vs_order_all_classes(df, race_id, normalize_to_class=normalize_to_class,
                                                                     order_cutoff=order_cutoff, trend_type=trend_type)
    plt.plot(x_values, trends.trend(x_values, overall_trend_params, trend_type, normalize_to_0=True),
             color='darkorange',
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


def plot_split_perf_vs_order_trends_competitor(df: pd.DataFrame, race_id, race_name, normalize_to_class=True,
                                               trend_type='linear', y_min=None, y_max=None, alpha=0.2):
    """
    :param df: pandas DataFrame containing performance data
    :param race_id: unique identifier for the race
    :param race_name: name of the race
    :param normalize_to_class: boolean indicating whether to normalize performance to class average
    :param trend_type: type of trend line to fit ('linear' or 'exp_decay_offset')
    :param y_min: minimum value for y-axis of the plot
    :param y_max: maximum value for y-axis of the plot
    :param alpha: transparency level for the individual competitor trend slope markers
    :return: None

    This function plots the trend of split performance versus split order for a given race. The plot includes the
    individual slopes from the trends of the competitors, the best fit slope for the competitors, and the resolved
    best fit line for the aggregate performance trend.
    """
    # Get trends parameters
    overall_trend_params, slope_trend_params, split_order_list, trend_slope_list = (
        stats.fit_split_perf_vs_order_all_competitors(df, race_id, normalize_to_class=normalize_to_class,
                                                      trend_type=trend_type))
    # Range of values for plotting best fit lines
    x_values = np.arange(0, np.max(split_order_list))

    # Create plot with two y axis
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # Plot individual slopes from best fits of competitors
    ax2.plot(split_order_list, trend_slope_list, '.', alpha=alpha,
             label='Slopes from trends of individual competitors', color='darkorange')

    # Plot best fit of slopes
    if trend_type == 'linear':
        ax2.plot(x_values, trends.trend(x_values, slope_trend_params, 'constant'), '--', label='Best fit of slopes')
    elif trend_type == 'exp_decay_offset':
        ax2.plot(x_values, trends.trend(x_values, slope_trend_params, 'exp_decay_der'), '--', label='Best fit of slopes')

    # Plot resolved best fit line for performance
    ax1.plot(x_values, trends.trend(x_values, overall_trend_params, trend_type, normalize_to_0=True), color='black',
             linewidth=2,
             label='Aggregate performance trend')

    ax1.set_xlabel('Split Order')
    ax1.set_ylabel('Relative Performance (lower is better)')
    ax2.set_ylabel('Slope of Individual Competitor Performance Trends')
    plt.title(f'{race_name}\n'
              f'Trend of Split Performance vs Split Order')

    plt.grid(True)
    plt.legend()

    # Set y_min/Y_max if desired
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()


def plot_split_order_vs_time(df: pd.DataFrame, race_id, race_name, tz, use_start_time=False, highlight_class_id=None,
                             y_min=None, y_max=None, alpha=0.2, order_cutoff=None, average_window=300,
                             time_step=30):
    """
    Function to plot split order against time for a given race.
    
    :param df: A pandas DataFrame containing the race data.
    :param race_id: The ID of the race for which the plot is generated.
    :param race_name: The name of the race.
    :param tz: Timezone of the race
    :param use_start_time: Boolean indicating whether to use athlete start time or timestamp.
    :param highlight_class_id: Highlight data points from one class in a different color.
    :param y_min: The minimum value for the y-axis.
    :param y_max: The maximum value for the y-axis.
    :param alpha: The transparency level of the race points on the plot.
    :param order_cutoff: The maximum value for split orders to plot.
    :param average_window: Averaging window width for line in seconds
    :param time_step: Time step for average line in seconds
    :return: None
    """

    # Define a mask to filter for the race
    if order_cutoff is not None:
        race_mask = (df['race_id'] == race_id) & (df['split_order'] <= order_cutoff)
    else:
        race_mask = df['race_id'] == race_id

    # Determine the time column to use
    time_column = 'start_time' if use_start_time else 'timestamp'

    # Extract time and split_order columns
    times = df[race_mask][time_column]
    split_orders = df[race_mask]['split_order']

    # Get average line
    averages_df = utilities.window_avg_line(df[race_mask], 'split_order', time_column=time_column,
                                            average_window=average_window, time_step=time_step)

    # For adjusting time to local race time
    tz_offset = tz * 3600.0

    # Format the x-axis to show time in hh:mm:ss
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator())  # set the x-axis major locator to show every hour

    # Create the plot
    if highlight_class_id is not None:
        class_mask = df['class_id'] == highlight_class_id
        plt.plot(pd.to_datetime(df[race_mask & ~class_mask][time_column] + tz_offset, unit='s', utc=True),
                 df[race_mask & ~class_mask]['split_order'], '.', alpha=alpha, color='darkorange')
        plt.plot(pd.to_datetime(df[race_mask & class_mask][time_column] + tz_offset, unit='s', utc=True),
                 df[race_mask & class_mask]['split_order'], '.', color='blue', alpha=0.5,
                 label=f'Class {highlight_class_id}')
    else:
        plt.plot(pd.to_datetime(df[race_mask][time_column] + tz_offset, unit='s', utc=True),
                 df[race_mask]['split_order'], '.', alpha=alpha, color='darkorange')

    # Plot windowed average line
    plt.plot(pd.to_datetime(averages_df[time_column] + tz_offset, unit='s', utc=True),
             averages_df[f'split_order_avg'], color='black', label=f'{average_window // 60}-minute width average')

    plt.xlabel('Athlete Start Time' if use_start_time else 'Timestamp')
    plt.ylabel('Split Order')
    plt.title(f'{race_name}\nSplit Order vs {"Athlete Start Time" if use_start_time else "Timestamp"}')

    plt.grid(True)
    if highlight_class_id is not None:
        plt.legend()

    # Set y_min/y_max if desired
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()
