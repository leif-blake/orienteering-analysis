"""
Produces plots which show the relative performance of runners' split times over the course of a race
"""

import sqlite3
from tkinter import filedialog
import pandas as pd

import plotter
import utilities

if __name__ == '__main__':
    # Import all database paths in folder
    folder_path = filedialog.askdirectory()
    db_file_paths = utilities.find_db_files(folder_path)

    # tz = -4 # Ontario Summer time
    tz = +2  # Sweden Summer time

    average_window = 60 * 30
    y_min_zoom = 0.8
    y_max_zoom = 1.2
    use_start_time = True

    for db_file_path in db_file_paths:
        conn = sqlite3.connect(db_file_path)
        cursor = conn.cursor()

        splits_df = pd.read_pickle(db_file_path[:-3] + '_splits.pkl')
        split_performances = pd.read_pickle(db_file_path[:-3] + '_splits_perf.pkl')

        race_ids = [1, 2, 3, 4]
        # race_ids = [4]
        # race_ids = [3]
        for race_id in race_ids:
            cursor.execute('SELECT name FROM races WHERE id = ?', (race_id,))
            race_name = cursor.fetchone()[0]
            # plotter.plot_split_performances(split_performances, race_id, race_name, tz, average_window=average_window,
            #                         use_start_time=use_start_time)
            plotter.plot_split_perf_vs_time(split_performances, race_id, race_name, tz, y_min=y_min_zoom,
                                            y_max=y_max_zoom, average_window=average_window,
                                            use_start_time=use_start_time)
            plotter.plot_split_vs_order(split_performances, race_id, race_name, tz, y_min=y_min_zoom,
                                            y_max=y_max_zoom, order_avg_window=10, class_id=8, x_max=200)

        conn.close()

        print("Done")
