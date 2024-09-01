"""
Functions to calculate trend lines and other statistical measures
"""

import numpy as np
import pandas as pd
import scipy.optimize as opt


def trend_exp_decay(x_values, coeffs, normalize_to_0=False):
    """
    Computes the exponential decay trend given input x_values and coefficients.

    :param x_values: The x-values for which the trend needs to be computed.
    :param coeffs: A tuple of (a, b, c) where a, b, and c are coefficients.
    :param normalize_to_0: If True, normalizes the trend to equal 1 at x = 0.
    :return: Computed trend value.
    """
    a, b, c = coeffs  # Unpack the coefficients

    if normalize_to_0:
        return (1 / (a + c)) * (a * np.power(b, -x_values) + c)
    else:
        return a * np.power(b, -x_values) + c


def calc_trend_exp_decay(x_data, y_data, initial_guess=None):
    """
    Optimizes the trend exponential decay parameters using the least squares method.

    :param x_data: List of x-axis data points.
    :param y_data: List of y-axis data points.
    :param initial_guess: Initial coefficients.
    :return: List of optimized trend exponential decay parameters [a, b, c].
    """

    if initial_guess is None:
        initial_guess = [1.0, 1.0, 1.0]

    def residuals(params, x, y):
        a, b, c = params
        return y - trend_exp_decay(x, [a, b, c])

    opt_params = opt.least_squares(residuals, initial_guess, loss='soft_l1', args=(x_data, y_data)).x

    return opt_params


def trend_linear(x_values, coeffs, normalize_to_0=False):
    """
    Computes the linear trend given input x_values and coefficients.

    :param x_values: The x-values for which the trend needs to be computed.
    :param coeffs: A tuple of (m, b) where m is the slope and b is the y-intercept.
    :param normalize_to_0: If True, normalizes the trend to equal 1 at x = 0.
    :return: Computed trend value.
    """
    m, b = coeffs  # Unpack the coefficients

    if normalize_to_0:
        return (1 / b) * (m * x_values + b)
    else:
        return m * x_values + b


def calc_trend_linear(x_data, y_data, initial_guess=None):
    """
    Optimizes the linear trend parameters using the least squares method.

    :param x_data: List of x-axis data points.
    :param y_data: List of y-axis data points.
    :param initial_guess: Initial coefficients for the linear trend.
    :return: List of optimized linear trend parameters [m, b].
    """

    if initial_guess is None:
        initial_guess = [0.0, 1.0]

    def residuals(params, x, y):
        m, b = params
        return y - trend_linear(x, [m, b])

    opt_params = opt.least_squares(residuals, initial_guess, loss='linear', args=(x_data, y_data)).x

    return opt_params


def trend(x_values, coeffs, trend_type, normalize_to_0=False):
    """
    :param x_values: List of x values.
    :param coeffs: List of coefficients for the trend equation.
    :param trend_type: Type of trend to calculate. Defaults to 'linear'.
    :return: The trend values based on the given x values, coefficients, and trend type.

    This function calculates the trend values based on the given x values and coefficients. The trend type can be specified as 'linear' or 'exp'. By default, the function calculates a linear trend. If an unsupported trend type is provided, a ValueError is raised.

    Example usage:
        x = [1, 2, 3]
        coeffs = [1, 2, 3]
        trend_values = trend(x, coeffs, 'linear')
        print(trend_values)  # Output: [6, 8, 10]

        x = [1, 2, 3]
        coeffs = [1, 2, 3]
        trend_values = trend(x, coeffs, 'exp')
        print(trend_values)  # Output: [14.778112197861299, 48.49860094734067, 159.3093279271219]
    """
    if trend_type == 'linear':
        return trend_linear(x_values, coeffs, normalize_to_0=normalize_to_0)
    elif trend_type == 'exp':
        return trend_exp_decay(x_values, coeffs, normalize_to_0=normalize_to_0)
    else:
        raise ValueError("Unsupported trend type. Use 'linear' or 'exp'.")


def calc_trend(x_data, y_data, trend_type, initial_guess=None):
    """
    Calls the appropriate function to calculate trend parameters based on the trend_type.

    :param x_data: List of x-axis data points.
    :param y_data: List of y-axis data points.
    :param trend_type: Type of trend to calculate. Defaults to 'linear'.
    :param initial_guess Initial coefficients for the trend.
    :return: List of optimized trend parameters.
    :rtype: list
    :raises ValueError: If the provided trend_type is not supported.
    """

    if trend_type == 'linear':
        return calc_trend_linear(x_data, y_data, initial_guess=initial_guess)
    elif trend_type == 'exp':
        return calc_trend_exp_decay(x_data, y_data, initial_guess=initial_guess)
    else:
        raise ValueError("Unsupported trend type. Use 'linear' or 'exp'.")


def fit_split_perf_vs_order(split_perf_df: pd.DataFrame, race_id: int or str, class_id: int or str,
                            normalize_to_class=True, order_cutoff=None, trend_type='linear'):
    """
    Fits an exponential decay trend for split performance versus order using the provided DataFrame.

    :param split_perf_df: pandas DataFrame containing the split performance data.
    :param race_id: Identifier for the race.
    :param class_id: Identifier for the class.
    :param normalize_to_class: If True, normalizes the performance to the class average.
    :param order_cutoff: Maximum order value to consider for the trend fitting. If None, considers all orders.
    :param trend_type: Type of trend to calculate. Defaults to 'linear'.
    :return: List of optimized trend exponential decay parameters [a, b, c].
    :rtype: list
    """

    # Get column for trend line
    trend_col = 'class_perf' if normalize_to_class else 'overall_perf'

    # Extracting the order and performance data for the given race_id and class_id
    race_class_data = split_perf_df[(split_perf_df['race_id'] == race_id) & (split_perf_df['class_id'] == class_id)]

    # If order_cutoff is specified, filters the data to include only splits up to the order_cutoff
    if order_cutoff is not None:
        race_class_data = race_class_data[race_class_data['split_order'] <= order_cutoff]

    # Extracting the x and y data to use for calculating the trend. Assuming 'order' is the x axis and 
    # 'performance' is the y axis
    x_data = race_class_data['split_order'].values
    y_data = race_class_data[trend_col].values

    # Call the calc_trend function with the prepared x_data and y_data
    trend_params = None
    if trend_type == 'linear':
        trend_params = calc_trend_linear(x_data, y_data, initial_guess=[0.0, 1.0])
    elif trend_type == 'exp':
        trend_params = calc_trend_exp_decay(x_data, y_data, initial_guess=[0.2, 1.05, 0.9])

    return trend_params


def fit_split_perf_vs_order_all_classes(split_perf_df: pd.DataFrame, race_id: int or str,normalize_to_class=True,
                                        order_cutoff=None, trend_type='linear'):
    """
    Fits an exponential decay trend for split performance versus order using the provided DataFrame. Combines all class
    data to get one trend line

    :param split_perf_df: pandas DataFrame containing the split performance data.
    :param race_id: Identifier for the race.
    :param normalize_to_class: If True, normalizes the performance to the class average, otherwise normalize to overall.
    :param order_cutoff: Maximum order value to consider for the trend fitting. If None, considers all orders.
    :param trend_type: Type of trend to calculate. Defaults to 'linear'.
    :return: List of optimized trend exponential decay parameters [a, b, c].
    :rtype: list
    """

    # Get column for trend line
    trend_col = 'class_perf' if normalize_to_class else 'overall_perf'

    # Define a mask to filter by race
    mask = (split_perf_df['race_id'] == race_id)

    # Get x range
    if order_cutoff is None:
        x_values = np.arange(0, split_perf_df[mask]['split_order'].max())
    else:
        x_values = np.arange(0, order_cutoff)

    # Get trends for individual classes
    split_order_list = []
    trend_point_list = []
    total_num_splits = split_perf_df[mask].shape[0]
    average_num_splits = total_num_splits / split_perf_df[mask]['class_id'].nunique()
    for class_id in split_perf_df[mask]['class_id'].unique():
        num_splits_class = split_perf_df[mask & (split_perf_df['class_id'] == class_id)].shape[0]
        trend_params = fit_split_perf_vs_order(split_perf_df, race_id, class_id, normalize_to_class=normalize_to_class,
                                                     order_cutoff=order_cutoff, trend_type=trend_type)
        split_order_list += x_values.tolist()
        trend_point_list += (trend(x_values, trend_params, trend_type, normalize_to_0=True) * num_splits_class / total_num_splits).tolist()

    # Get overall trend
    overall_trend_params = calc_trend(np.array(split_order_list), np.array(trend_point_list), trend_type)

    return overall_trend_params
