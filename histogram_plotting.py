"""
Produces plots which show the relative performance of runners' split times over the course of a race
"""

import sqlite3
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scipy_stats

import plotter
import stats
import utilities
import numpy as np

if __name__ == '__main__':
    # Import all database paths in folder
    folder_path = filedialog.askdirectory()
    db_file_paths = utilities.find_db_files(folder_path)

    # tz = -4 # Ontario Summer time
    tz = +2  # Sweden Summer time

    # race_ids = [1, 2, 3, 4]
    race_ids = [2]
    use_simulated_data = False

    for db_file_path in db_file_paths:
        conn = sqlite3.connect(db_file_path)
        cursor = conn.cursor()

        split_performances = pd.read_pickle(
            db_file_path[:-3] + f'{"_sim" if use_simulated_data else ""}_splits_perf.pkl')

        # Plot histogram of split performance for each race
        for race_id in race_ids:
            cursor.execute('SELECT name FROM races WHERE id = ?', (race_id,))
            race_name = cursor.fetchone()[0]

            # Subset split performances for the specific race
            race_split_performances = split_performances[split_performances['race_id'] == race_id]

            split_values = race_split_performances.loc[(race_split_performances["class_perf"] < 30) &
                                                       (race_split_performances[
                                                            "class_perf"] > 0.01), "class_perf"].copy()

            # Normalize the split values by the sum of the split values and convert to numpy array
            # split_values = (split_values / np.sum(split_values)).to_numpy()
            split_values = split_values.to_numpy()

            # split_values = split_values/np.sum(split_values)

            # Plot histogram using matplotlib
            plt.figure()
            plt.hist(split_values, bins=300, color='darkorange')
            # plt.axvline(x=1, color='r', linestyle='-', label='x=1')
            plt.axvline(x=np.mean(split_values), color='k', linestyle='--', label='x=mean')
            plt.title(f'Split Performances in class for: {race_name}\n'
                      f'Min: {np.min(split_values):.3f}, Max: {np.max(split_values):.3f}')
            plt.xlabel('Split Performance')
            plt.ylabel('Frequency (Log scale)')
            # Logarithmic y-axis
            plt.yscale('log')
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # # Plot a gamma pdf which approximates the histogram
            # shape, loc, scale = scipy_stats.gamma.fit(split_values)
            # x = np.linspace(0, np.max(split_values), 1000)
            # y = scipy_stats.gamma.pdf(x, shape, loc, scale)
            # plt.plot(x, y, 'r-', linewidth=2, label='gamma pdf best fit')

            # # Plot a beta pdf which approximates the histogram
            # a, b, loc, scale = scipy_stats.beta.fit(split_values, floc=0,
            #                                         fscale=np.max(race_split_performances["class_perf"]))
            # x = np.linspace(0, np.max(race_split_performances["class_perf"]), 1000)
            # y = scipy_stats.beta.pdf(x, a, b, loc, scale)
            # plt.plot(x, y, 'g-', linewidth=2, label='beta pdf best fit')

            # # Plot a lognormal pdf which approximates the histogram
            # shape, loc, scale = scipy_stats.lognorm.fit(split_values)
            # x = np.linspace(0, np.max(split_values), 1000)
            # y = scipy_stats.lognorm.pdf(x, shape, loc, scale)
            # plt.plot(x, y, 'b-', linewidth=2, label='lognormal pdf best fit')

            # Plot a lognormal pdf which approximates the histogram
            shape, loc, scale = (0.4, 0.15, 1.5)
            x = np.linspace(0.3, 5, 1000)
            y = scipy_stats.lognorm.pdf(x, shape, loc, scale)
            plt.plot(x, y * 1000 * 1 / (x - 0.15) ** 2, 'b--', linewidth=2, label='lognormal pdf')

            # # Plot a Weibull pdf which approximates the histogram
            # c, loc, scale = scipy_stats.weibull_min.fit(split_values)
            # x = np.linspace(0, np.max(split_values), 1000)
            # y = scipy_stats.weibull_min.pdf(x, c, loc, scale)
            # plt.plot(x, y, 'm-', linewidth=2, label='weibull pdf best fit')

            plt.xlim(0, 5)

            plt.legend()
            plt.show()
