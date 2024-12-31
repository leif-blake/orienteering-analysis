"""
Produces plots which show the relative performance of runners' split times over the course of a race
"""

import sqlite3
from tkinter import filedialog
import pandas as pd

import plotter
import utilities
from trends import trend

if __name__ == '__main__':
    # Import all database paths in folder
    folder_path = filedialog.askdirectory()
    db_file_paths = utilities.find_db_files(folder_path)

    # tz = -4 # Ontario Summer time
    tz = +2  # Sweden Summer time

    # race_ids = [1, 2, 3, 4]
    race_ids = [2]
    average_window = 60 * 30
    y_min_zoom = 0.8
    y_max_zoom = 1.2
    use_start_time = True
    split_order_cutoff = 500
    class_ids_to_plot = []
    plot_trend = True
    trend_type = 'exp_decay_offset'
    class_to_highlight = 8

    for db_file_path in db_file_paths:
        conn = sqlite3.connect(db_file_path)
        cursor = conn.cursor()

        splits_df = pd.read_pickle(db_file_path[:-3] + '_splits.pkl')
        split_performances = pd.read_pickle(db_file_path[:-3] + '_splits_perf.pkl')

        for race_id in race_ids:
            cursor.execute('SELECT name FROM races WHERE id = ?', (race_id,))
            race_name = cursor.fetchone()[0]
            class_ids_to_plot = splits_df[splits_df['race_id'] == race_id]['class_id'].unique()
            # plotter.plot_split_performances(split_performances, race_id, race_name, tz, average_window=average_window,
            #                         use_start_time=use_start_time)
            plotter.plot_split_perf_vs_time(split_performances, race_id, race_name, tz, y_min=y_min_zoom,
                                            y_max=y_max_zoom, average_window=average_window,
                                            use_start_time=use_start_time, highlight_class_id=class_to_highlight,
                                            order_cutoff=split_order_cutoff)
            # plotter.plot_split_vs_order(split_performances, race_id, race_name, order_avg_window=15, y_min=y_min_zoom,
            #                             y_max=y_max_zoom, order_cutoff=split_order_cutoff,
            #                             highlight_class_id=class_to_highlight)
            # for index, class_id_to_plot in enumerate(class_ids_to_plot):
            #     if index % 10 != 0:
            #         continue
            #     plotter.plot_split_vs_order(split_performances, race_id, race_name, y_min=y_min_zoom,
            #                                 y_max=y_max_zoom, order_avg_window=15, class_id=class_id_to_plot,
            #                                 x_max=split_order_cutoff, plot_trend=plot_trend, trend_type=trend_type)
            # plotter.plot_split_vs_order(split_performances, race_id, race_name, order_avg_window=15, y_min=y_min_zoom,
            #                             y_max=y_max_zoom, class_id=class_to_highlight, order_cutoff=split_order_cutoff,
            #                             plot_trend=plot_trend, trend_type=trend_type)
            # plotter.plot_all_split_vs_order_trends(split_performances, race_id, race_name, y_min=y_min_zoom,
            #                                        y_max=y_max_zoom, order_cutoff=split_order_cutoff,
            #                                        trend_type=trend_type)
            # plotter.plot_split_order_vs_time(split_performances, race_id, race_name, tz, average_window=average_window,
            #                                  highlight_class_id=class_to_highlight, order_cutoff=split_order_cutoff)
            plotter.plot_split_perf_vs_order_trends_competitor(split_performances, race_id, race_name,
                                                               trend_type=trend_type)

        conn.close()

        print("Done")
