"""
Produces plots which show the relative performance of runners' split times over the course of a race
"""

import sqlite3
from tkinter import filedialog
import pandas as pd

import plotter

if __name__ == '__main__':
    # Choose the database to use
    db_file_path = filedialog.askopenfilename(
        title="Open Event Database",
        defaultextension=".db",
        filetypes=[("SQLite3 Database", "*.db"), ("All Files", "*.*")]
    )

    # tz = -4 # Ontario Summer time
    tz = +2  # Sweden Summer time

    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()

    average_window = 60 * 30
    y_min_zoom = 0.8
    y_max_zoom = 1.2
    use_start_time = True

    split_performances = pd.read_pickle(db_file_path[:-3] + '_splits_perf.pkl')

    race_ids = [1, 2, 3, 4]
    # race_ids = [4]
    # race_ids = [3]
    for race_id in race_ids:
        cursor.execute('SELECT name FROM races WHERE id = ?', (race_id,))
        race_name = cursor.fetchone()[0]
        # plotter.plot_split_performances(split_performances, race_id, race_name, tz, average_window=average_window,
        #                         use_start_time=use_start_time)
        plotter.plot_split_performances(split_performances, race_id, race_name, tz, y_min=y_min_zoom,
                                        y_max=y_max_zoom,
                                        average_window=average_window, use_start_time=use_start_time)

    conn.close()

    print("Done")
