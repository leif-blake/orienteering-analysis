"""
Functions to calculate trend lines and other statistical measures
"""

import numpy as np
import pandas as pd

import trends


def fit_split_perf_vs_order_class(split_perf_df: pd.DataFrame, race_id: int or str, class_id: int or str,
                            normalize_to_class=True, order_cutoff=None, trend_type='linear', split_order_col='split_order'):
    """
    Fits an exponential decay trend for split performance versus order using the provided DataFrame.

    :param split_perf_df: pandas DataFrame containing the split performance data.
    :param race_id: Identifier for the race.
    :param class_id: Identifier for the class.
    :param normalize_to_class: If True, normalizes the performance to the class average.
    :param order_cutoff: Maximum order value to consider for the trend fitting. If None, considers all orders.
    :param trend_type: Type of trend to calculate. Defaults to 'linear'.
    :param split_order_col: Column name for split order.
    :return: List of optimized trend parameters.
    :rtype: list
    """

    # Get column for trend line
    trend_col = 'class_perf' if normalize_to_class else 'overall_perf'

    # Extracting the order and performance data for the given race_id and class_id
    race_class_data = split_perf_df[(split_perf_df['race_id'] == race_id) & (split_perf_df['class_id'] == class_id)]

    # If order_cutoff is specified, filters the data to include only splits up to the order_cutoff
    if order_cutoff is not None:
        race_class_data = race_class_data[race_class_data[split_order_col] <= order_cutoff]

    # Extracting the x and y data to use for calculating the trend. Assuming 'order' is the x axis and 
    # 'performance' is the y axis
    x_data = race_class_data[split_order_col].values
    y_data = race_class_data[trend_col].values

    # Call the calc_trend function with the prepared x_data and y_data
    trend_params = None
    if trend_type == 'linear':
        trend_params = trends.calc_trend_linear(x_data, y_data, initial_guess=[0.0, 1.0])
    elif trend_type == 'exp_decay_offset':
        trend_params = trends.calc_trend_exp_decay_offset(x_data, y_data, initial_guess=[0.2, 1.05, 0.9])

    return trend_params


def fit_split_perf_vs_order_all_classes(split_perf_df: pd.DataFrame, race_id: int or str, normalize_to_class=True,
                                        order_cutoff=None, trend_type='linear', split_order_col='split_order'):
    """
    Fits a trend for split performance versus order using the provided DataFrame. Combines all class
    data to get one trend line

    :param split_perf_df: pandas DataFrame containing the split performance data.
    :param race_id: Identifier for the race.
    :param normalize_to_class: If True, normalizes the performance to the class average, otherwise normalize to overall.
    :param order_cutoff: Maximum order value to consider for the trend fitting. If None, considers all orders.
    :param trend_type: Type of trend to calculate. Defaults to 'linear'.
    :param split_order_col: Column name for split order.
    :return: List of optimized trend exponential decay parameters [a, b, c].
    :rtype: list
    """

    # Define a mask to filter by race
    mask = (split_perf_df['race_id'] == race_id)

    # Get x range
    if order_cutoff is None:
        x_values = np.arange(0, split_perf_df[mask][split_order_col].max())
    else:
        x_values = np.arange(0, order_cutoff)

    # Get trends for individual classes
    split_order_list = []
    trend_point_list = []
    total_num_splits = split_perf_df[mask].shape[0]
    average_num_splits = total_num_splits / split_perf_df[mask]['class_id'].nunique()
    for class_id in split_perf_df[mask]['class_id'].unique():
        num_splits_class = split_perf_df[mask & (split_perf_df['class_id'] == class_id)].shape[0]
        trend_params = fit_split_perf_vs_order_class(split_perf_df, race_id, class_id,
                                                     normalize_to_class=normalize_to_class, order_cutoff=order_cutoff,
                                                     trend_type=trend_type)
        split_order_list += x_values.tolist()
        trend_point_list += (trends.trend(x_values, trend_params, trend_type, normalize_to_0=True) * num_splits_class / total_num_splits).tolist()

    # Get overall trend
    overall_trend_params = trends.calc_trend(np.array(split_order_list), np.array(trend_point_list), trend_type)

    return overall_trend_params

def fit_split_perf_vs_order_competitor(split_perf_df: pd.DataFrame, race_id: int or str, competitor_id,
                                       normalize_to_class=True, split_order_col='split_order'):
    """
    Fits a trend for split performance versus order for an individual competitor using the provided DataFrame. Uses linear trend.
    :param split_perf_df: DataFrame containing the split performance data.
    :param race_id: Race id in database.
    :param competitor_id: Comptetitor id in database.
    :param normalize_to_class: If True, normalizes the performance to the class average, otherwise normalize to overall.
    :param split_order_col: Column name for split order.
    :return: List of optimized trend parameters.
    :rtype: list
    """

    # Get column for trend line. Use columns normalized to individual performance
    trend_col = 'norm_class_perf' if normalize_to_class else 'norm_overall_perf'

    # Extracting the order and performance data for the given race_id and class_id
    race_class_data = split_perf_df[(split_perf_df['race_id'] == race_id) & (split_perf_df['competitor_id'] == competitor_id)]

    # Extracting the x and y data to use for calculating the trend. Assuming 'order' is the x axis and
    # 'performance' is the y axis
    x_data = race_class_data[split_order_col].values
    y_data = race_class_data[trend_col].values

    # Call the calc_trend function with the prepared x_data and y_data
    trend_params = trends.calc_trend_linear(x_data, y_data, initial_guess=[0.0, 1.0])

    return trend_params


def fit_split_perf_vs_order_all_competitors(split_perf_df: pd.DataFrame, race_id: int or str, normalize_to_class=True, trend_type='linear',
                                        split_order_col='split_order'):
    """
    Fits a trend for split performance versus order using the provided DataFrame. Combines all individual competitor
    data to get one trend line

    :param split_perf_df: pandas DataFrame containing the split performance data.
    :param race_id: Identifier for the race.
    :param normalize_to_class: If True, normalizes the performance to the class average, otherwise normalize to overall.
    :param trend_type: Type of trend to calculate. Defaults to 'linear'.
    :param split_order_col: Column name for split order.
    :return: List of optimized trend parameters.
    :rtype: list
    """

    # Define a mask to filter by race
    mask = (split_perf_df['race_id'] == race_id)

    # Get x range
    x_values = np.arange(0, split_perf_df[mask][split_order_col].max())

    # Get slopes of trends for individual competitors
    split_order_list = []
    trend_slope_list = []
    for competitor_id in split_perf_df[mask]['competitor_id'].unique():
        competitor_orders = split_perf_df[mask & (split_perf_df['competitor_id'] == competitor_id)][split_order_col].values
        trend_params = fit_split_perf_vs_order_competitor(split_perf_df, race_id, competitor_id,
                                                          normalize_to_class=normalize_to_class)
        # Record the slope for each trend line
        split_order_list += competitor_orders.tolist()
        trend_slope_list += [trend_params[0]] * len(competitor_orders)

    # Get overall trend
    overall_trend_params = []
    slope_trend_params = []
    if trend_type == 'linear':
        slope_trend_params = trends.calc_trend(np.array(split_order_list), np.array(trend_slope_list), 'constant')
        overall_trend_params = [slope_trend_params[0], 1.0]
    elif trend_type == 'exp_decay_offset':
        slope_trend_params = trends.calc_trend(np.array(split_order_list), np.array(trend_slope_list), 'exp_decay_der')
        overall_trend_params = [slope_trend_params[0], slope_trend_params[1], 1 - slope_trend_params[0]]

    return overall_trend_params, slope_trend_params, split_order_list, trend_slope_list

def fit_split_perf_vs_order(split_perf_df: pd.DataFrame, race_id: int or str,
                            normalize_to_class=True, order_cutoff=None, trend_type='linear', split_order_col='split_order'):
    """
    Fits a trend for split performance versus order using the provided DataFrame. Does not differentiate between classes.

    :param split_perf_df: pandas DataFrame containing the split performance data.
    :param race_id: Identifier for the race.
    :param normalize_to_class: If True, normalizes the performance to the class average. Otherwise, normalizes to overall.
    :param order_cutoff: Maximum order value to consider for the trend fitting. If None, considers all orders.
    :param trend_type: Type of trend to calculate. Defaults to 'linear'.
    :param split_order_col: Column name for split order.
    :return: List of optimized trend parameters.
    :rtype: list
    """

    # Get column for trend line
    trend_col = 'class_perf' if normalize_to_class else 'overall_perf'

    # Extracting the order and performance data for the given race_id and class_id
    race_class_data = split_perf_df[split_perf_df['race_id'] == race_id]

    # If order_cutoff is specified, filters the data to include only splits up to the order_cutoff
    if order_cutoff is not None:
        race_class_data = race_class_data[race_class_data[split_order_col] <= order_cutoff]

    # Extracting the x and y data to use for calculating the trend. Assuming 'order' is the x axis and
    # 'performance' is the y axis
    x_data = race_class_data[split_order_col].values
    y_data = race_class_data[trend_col].values

    # Call the calc_trend function with the prepared x_data and y_data
    trend_params = None
    if trend_type == 'linear':
        trend_params = trends.calc_trend_linear(x_data, y_data, initial_guess=[0.0, 1.0])
    elif trend_type == 'exp_decay_offset':
        trend_params = trends.calc_trend_exp_decay_offset(x_data, y_data, initial_guess=[0.2, 1.05, 0.9])

    return trend_params