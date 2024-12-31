"""
Trend line functions
"""

import numpy as np
from scipy import optimize as opt


def trend(x_values, coeffs, trend_type, normalize_to_0=False):
    """
    Computes the trend values based on the given x_values, coefficients, and trend type.

    :param x_values: List of x values.
    :param coeffs: List of coefficients for the trend equation.
    :param trend_type: Type of trend to calculate. Supported types are 'linear', 'exp_decay_offset', 'exp_decay', 'constant'.
    :param normalize_to_0: If True, normalizes the trend to equal 1 at x = 0 (applicable to 'linear' and 'exp_decay_offset' trends).
    :return: The trend values based on the given x values, coefficients, and trend type.
    :raises ValueError: If the provided trend_type is not supported.

    Example usage:
        x = [1, 2, 3]
        coeffs = [1, 2, 3]
        trend_values = trend(x, coeffs, 'linear')
        print(trend_values)  # Output: [6, 8, 10]

        x = [1, 2, 3]
        coeffs = [1, 2, 3]
        trend_values = trend(x, coeffs, 'exp_decay_offset')
        print(trend_values)  # Output: [14.778112197861299, 48.49860094734067, 159.3093279271219]
    """
    valid_trend_types = ['linear', 'exp_decay_offset', 'exp_decay_der', 'constant']

    if trend_type == valid_trend_types[0]:
        return trend_linear(x_values, coeffs, normalize_to_0=normalize_to_0)
    elif trend_type == valid_trend_types[1]:
        return trend_exp_decay_offset(x_values, coeffs, normalize_to_0=normalize_to_0)
    elif trend_type == valid_trend_types[2]:
        return trend_exp_decay_der(x_values, coeffs)
    elif trend_type == valid_trend_types[3]:
        return trend_constant(x_values, coeffs)
    else:
        raise ValueError(f"Unsupported trend type. Use one of {valid_trend_types}.")


def calc_trend(x_data, y_data, trend_type, initial_guess=None):
    """
    Calls the appropriate function to calculate trend parameters based on the trend_type.

    :param x_data: List of x-axis data points.
    :param y_data: List of y-axis data points.
    :param trend_type: Type of trend to calculate. Defaults to 'linear'.
    :param initial_guess: Initial coefficients for the trend.
    :return: List of optimized trend parameters.
    :rtype: list
    :raises ValueError: If the provided trend_type is not supported.
    """
    valid_trend_types = ['linear', 'exp_decay_offset', 'exp_decay_der', 'constant']
    if trend_type == valid_trend_types[0]:
        return calc_trend_linear(x_data, y_data, initial_guess=initial_guess)
    elif trend_type == valid_trend_types[1]:
        return calc_trend_exp_decay_offset(x_data, y_data, initial_guess=initial_guess)
    elif trend_type == valid_trend_types[2]:
        return calc_trend_exp_decay_der(x_data, y_data, initial_guess=initial_guess)
    elif trend_type == valid_trend_types[3]:
        return calc_trend_constant(y_data)
    else:
        raise ValueError(f'Unsupported trend type. Use one of {valid_trend_types}')


def trend_exp_decay_offset(x_values, coeffs, normalize_to_0=False):
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


def calc_trend_exp_decay_offset(x_data, y_data, initial_guess=None):
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
        return y - trend_exp_decay_offset(x, [a, b, c])

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


def trend_constant(x_values, coeffs):
    """
    Computes the constant trend given input x_values and coefficients.
    :param x_values: The x-values for which the trend needs to be computed.
    :param coeffs: A tuple with a single value (c) which is the constant value.
    :param normalize_to_0: If True, this has no effect as the constant trend is unchanged.
    :return: Computed trend value.
    """
    c = coeffs[0]  # Unpack the coefficient
    return np.full_like(x_values, c)


def calc_trend_constant(y_data):
    """
    Optimizes the constant trend parameters using the least squares method.
    :param x_data: List of x-axis data points.
    :param y_data: List of y-axis data points.
    :param initial_guess: Initial coefficient for the constant trend.
    :return: List of optimized constant trend parameter [c].
    """

    initial_guess = [np.mean(y_data)]

    def residuals(params, y):
        c = params[0]
        return y - np.full_like(y, c)

    opt_params = opt.least_squares(residuals, initial_guess, loss='soft_l1', args=(y_data,)).x
    return opt_params


def trend_exp_decay_der(x_values, coeffs):
    """
    Computes the exponential decay trend given input x_values and coefficients.
    This version trends to 0 as x increases.

    :param x_values: The x-values for which the trend needs to be computed.
    :param coeffs: A tuple of (a, b) where a and b are coefficients.
    :param normalize_to_0: If True, normalizes the trend to equal 1 at x = 0.
    :return: Computed trend value.
    """
    a, b = coeffs  # Unpack the coefficients

    return -a * np.log(b) / np.power(b, x_values)


def calc_trend_exp_decay_der(x_data, y_data, initial_guess=None):
    """
    Optimizes the trend exponential decay parameters using the least squares method.
    This version trends to 0 as x increases.

    :param x_data: List of x-axis data points.
    :param y_data: List of y-axis data points.
    :param initial_guess: Initial coefficients.
    :return: List of optimized trend exponential decay parameters [a, b].
    """
    if initial_guess is None:
        initial_guess = [10, 1.5]

    def residuals(params, x, y):
        a, b = params
        return y - trend_exp_decay_der(x, [a, b])

    opt_params = opt.least_squares(residuals, initial_guess, loss='soft_l1', args=(x_data, y_data)).x

    return opt_params
