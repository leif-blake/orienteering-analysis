"""
This script prompts the user for xml results files and a database file to save them to
"""

import sqlite3
from tkinter import Tk
from tkinter import filedialog
from iof_xml_parser import parse_results_to_sqlite

if __name__ == '__main__':
    # Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing

    # # Prompt user for time zone
    # tz = float(input("Enter timezone as UTC offset (ex: for EDT enter -4): "))
    # tz = -4  # Ontario Summer Time Zone
    tz = +2  # Sweden Summer Time Zone

    # Open a file dialog to select multiple xml files to load
    xml_file_paths = filedialog.askopenfilenames(
        title="Select IOF Result Files to Import",
        filetypes=[("IOF Result Files", "*.xml")]
    )

    # Open a file dialog to save the resulting database
    db_file_path = filedialog.asksaveasfilename(
        title="Save Event Database",
        defaultextension=".db",
        filetypes=[("SQLite3 Database", "*.db"), ("All Files", "*.*")]
    )

    if db_file_path:
        # Create and save the SQLite3 database at the chosen location
        conn = sqlite3.connect(db_file_path)
        cursor = conn.cursor()

        # Read the SQL file
        with open('event_schema.sql', 'r') as sql_file:
            sql_script = sql_file.read()

        # Execute the SQL script
        cursor.executescript(sql_script)

        # Commit the changes and close the connection
        conn.commit()
        conn.close()
        print(f"Database saved to: {db_file_path}")

        for xml_file_path in xml_file_paths:
            parse_results_to_sqlite(xml_file_path, db_file_path, tz)

        print("Done loading xml results")
