"""
Produces plots which show the relative performance of runners' split times over the course of a race
"""

import sqlite3
from tkinter import filedialog
import pandas as pd

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
    race_ids = [1]
    average_window = 60 * 30
    y_min_zoom = 0
    y_max_zoom = 8
    use_start_time = True
    split_order_cutoff = None
    class_ids_to_plot = []
    plot_trend = False
    trend_type = 'linear'
    class_to_highlight = None
    use_simulated_data = False

    # ****************************************************************************
    # Plot split order vs time
    # ****************************************************************************

    # List of trend slopes
    slopes = []
    intercepts = []
    adj_slopes_exp = []

    total_rows = 0

    for db_file_path in db_file_paths:
        conn = sqlite3.connect(db_file_path)
        cursor = conn.cursor()

        splits_df = pd.read_pickle(db_file_path[:-3] + f'{"_sim" if use_simulated_data else ""}_splits.pkl')
        split_performances = pd.read_pickle(db_file_path[:-3] + f'{"_sim" if use_simulated_data else ""}_splits_perf.pkl')
        total_rows += len(split_performances)

        for race_id in race_ids:
            cursor.execute('SELECT name FROM races WHERE id = ?', (race_id,))
            race_name = cursor.fetchone()[0]
            class_ids_to_plot = splits_df[splits_df['race_id'] == race_id]['class_id'].unique()
            trend_params_exp = plotter.plot_split_vs_order(split_performances, race_id, race_name, order_avg_window=15, y_min=y_min_zoom,
                                        y_max=y_max_zoom, order_cutoff=split_order_cutoff,
                                        highlight_class_id=class_to_highlight, split_order_col='split_order_at_exp_timestamp', plot_trend=plot_trend,
                                        trend_type=trend_type)
            slopes.append(trend_params_exp[0])
            intercepts.append(trend_params_exp[1])
            adj_slopes_exp.append(trend_params_exp[0] / trend_params_exp[1])

            # Plot split order vs time
            # plotter.plot_split_order_vs_time(split_performances, race_id, race_name, tz, average_window=average_window,
            #                                  highlight_class_id=class_to_highlight, order_cutoff=split_order_cutoff)
            # plotter.plot_exp_split_order_vs_split_order_hist(split_performances[split_performances['class_perf'] < 1], race_id, race_name, order_cutoff=split_order_cutoff)
            # plotter.plot_exp_split_order_vs_split_order_hist(split_performances[split_performances['class_perf'] > 1],
            #                                                  race_id, race_name, order_cutoff=split_order_cutoff)

        conn.close()

        print("Done one file")

    print(f'Slope avg: {sum(slopes) / len(slopes)}, std: {np.std(slopes)}')
    print(f'Intercept avg: {sum(intercepts) / len(intercepts)}, std: {np.std(intercepts)}')
    print(f'Adj slope avg: {sum(adj_slopes_exp) / len(adj_slopes_exp)}, std: {np.std(adj_slopes_exp)}')
    print(f'Total rows: {total_rows}')

    # ****************************************************************************
    # Plot box plot of trend slopes
    # ****************************************************************************

    # trend_slopes_df_exp = pd.DataFrame()
    # trend_slopes_df_start = pd.DataFrame()
    # split_order_cutoffs = np.arange(50, 1050, 50)
    #
    # for split_order_cutoff in split_order_cutoffs:
    #     adj_slopes_exp = []
    #     adj_slopes_start = []
    #
    #     for db_file_path in db_file_paths:
    #         conn = sqlite3.connect(db_file_path)
    #         cursor = conn.cursor()
    #
    #         split_performances = pd.read_pickle(db_file_path[:-3] + f'{"_sim" if use_simulated_data else ""}_splits_perf.pkl')
    #
    #         for race_id in race_ids:
    #             cursor.execute('SELECT name FROM races WHERE id = ?', (race_id,))
    #             race_name = cursor.fetchone()[0]
    #             trend_params_exp = stats.fit_split_perf_vs_order(split_performances, race_id,
    #                                                              order_cutoff=split_order_cutoff, split_order_col='split_order_at_exp_timestamp',
    #                                                              trend_type=trend_type)
    #             adj_slopes_exp.append(trend_params_exp[0] / trend_params_exp[1])
    #             trend_params_start = stats.fit_split_perf_vs_order(split_performances, race_id,
    #                                                              order_cutoff=split_order_cutoff,
    #                                                              split_order_col='split_order_at_start',
    #                                                              trend_type=trend_type)
    #             adj_slopes_start.append(trend_params_start[0] / trend_params_start[1])
    #
    #
    #         conn.close()
    #
    #         print(f"Done evaluating database {db_file_path} for split order cutoff {split_order_cutoff}")
    #
    #     trend_slopes_df_exp[f'{split_order_cutoff}'] = adj_slopes_exp
    #     trend_slopes_df_start[f'{split_order_cutoff}'] = adj_slopes_start
    #
    #     # Print averages
    #     print(f'Average slope for split order cutoff {split_order_cutoff} at expected timestamp: {sum(adj_slopes_exp) / len(adj_slopes_exp)}')
    #     print(f'Average slope for split order cutoff {split_order_cutoff} at start timestamp: {sum(adj_slopes_start) / len(adj_slopes_start)}')
    #
    #     # Print std deviations
    #     print(f'Standard deviation for split order cutoff {split_order_cutoff} at expected timestamp: {np.std(adj_slopes_exp)}')
    #     print(f'Standard deviation for split order cutoff {split_order_cutoff} at start timestamp: {np.std(adj_slopes_start)}')
    #
    #
    # plotter.plot_box_whisker(trend_slopes_df_exp, split_order_cutoffs,
    #                          'Adjusted trend slope vs split order cutoff\nUsing Split Order at Expected Timestamp',
    #                          'Split order cutoff', 'Adjusted trend slope', y_min=-0.0026, y_max=0.0001)
    # plotter.plot_box_whisker(trend_slopes_df_start, split_order_cutoffs,
    #                          'Adjusted trend slope vs split order cutoff\nUsing Split Order at Competitor Start Time',
    #                          'Split order cutoff', 'Adjusted trend slope', y_min=-0.0026, y_max=0.0001)