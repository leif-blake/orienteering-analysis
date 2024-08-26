"""
Functions to parse iof xml event files
"""

import xml.etree.ElementTree as ET
import sqlite3
import time
import datetime
import re

import utilities

def regexp(expr, item):
    if item is None:
        return False
    reg = re.compile(expr)
    return reg.search(item) is not None


def parse_results_to_sqlite(xml_filename, db_filename, tz: float):
    """
    Parses an IOF Results XML file and writes to sql file
    :param xml_filename: Full path to XML file
    :param db_filename: Full path to sqlite3 database file
    :param tz: Timezone as UTC offset
    :return:
    """

    tz_offset_from_local = utilities.get_offset_from_local(tz)

    # Connect to database
    conn = sqlite3.connect(db_filename)
    conn.create_function("REGEXP", 2, regexp)
    cursor = conn.cursor()

    # Set up parser for xml
    tree = ET.parse(xml_filename)
    root = tree.getroot()

    race_name = utilities.get_element_text_or_none(root, 'Event/Name')
    race_date = utilities.get_element_text_or_none(root, 'Event/StartDate/Date')
    race_timestamp = time.mktime(datetime.datetime.strptime(race_date, "%Y-%m-%d").timetuple()) - tz_offset_from_local
    cursor.execute('INSERT OR IGNORE INTO races (name, date) VALUES (?, ?)', (race_name, race_timestamp,))
    conn.commit()

    # Retrieve race_id
    cursor.execute('SELECT id FROM races WHERE name = ?', (race_name,))
    race_id = cursor.fetchone()[0]

    # Retrieve largest fictional card number
    cursor.execute("SELECT card_no from results WHERE card_no REGEXP ? ORDER BY card_no ASC LIMIT 1", ('^f.*$',))
    row = cursor.fetchone()
    if row is None:
        last_fic_card_no = -1
    else:
        last_fic_card_no = int(row[0][1:])

    for class_result in root.findall('ClassResult'):
        class_name = utilities.get_element_text_or_none(class_result, 'ClassShortName')
        cursor.execute('INSERT OR IGNORE INTO classes (name) VALUES (?)', (class_name,))
        conn.commit()

        # Retrieve class_id
        cursor.execute('SELECT id FROM classes WHERE name = ?', (class_name,))
        class_id = cursor.fetchone()[0]

        # Insert competitors and results
        for competitor in class_result.findall('PersonResult'):
            # Competitor info
            given_name = utilities.get_element_text_or_none(competitor, 'Person/PersonName/Given')
            surname = utilities.get_element_text_or_none(competitor, 'Person/PersonName/Family')
            person_no = utilities.get_element_text_or_none(competitor, 'Person/PersonId')
            club = utilities.get_element_text_or_none(competitor, 'Club/Name')

            cursor.execute(
                'INSERT OR IGNORE INTO competitors (given_name, surname, person_no, club) VALUES (?, ?, ?, ?)',
                (given_name, surname, person_no, club,))
            conn.commit()

            # Result
            card_no = utilities.get_element_text_or_none(competitor, 'Result/CCardId')
            start_time_tz = utilities.get_element_text_or_none(competitor, 'Result/StartTime/Clock')
            finish_time_tz = utilities.get_element_text_or_none(competitor, 'Result/FinishTime/Clock')

            # If no card number is given for the result, set a fictional card number just to have a reference
            if card_no is None:
                last_fic_card_no += 1
                card_no = f'f{last_fic_card_no}'
            if start_time_tz is None:
                start_time = None
            else:
                start_time = time.mktime(datetime.datetime.strptime(f'{race_date} {start_time_tz}',
                                                                    "%Y-%m-%d %H:%M:%S").timetuple()) - tz_offset_from_local
            if finish_time_tz is None:
                finish_time = None
            else:
                finish_time = time.mktime(datetime.datetime.strptime(f'{race_date} {finish_time_tz}',
                                                                     "%Y-%m-%d %H:%M:%S").timetuple()) - tz_offset_from_local

            control_codes = '['
            control_times = '['
            for split_time in competitor.findall('Result/SplitTime'):
                control_code = utilities.get_element_text_or_none(split_time, 'ControlCode')
                control_time_in_race = utilities.hhmmss_to_seconds(utilities.get_element_text_or_none(split_time, 'Time'))
                control_time = None if control_time_in_race is None else control_time_in_race + start_time
                control_codes += control_code + ','
                control_times += f'{control_time},'
            control_codes = control_codes[:-1] + ']'
            control_times = control_times[:-1] + ']'

            cursor.execute(
                'INSERT OR IGNORE INTO results (race_id, card_no, start_time, finish_time, controls, control_times) VALUES (?, ?, ?, ?, ?, ?)',
                (race_id, card_no, start_time, finish_time, control_codes, control_times,))
            conn.commit()

            # If this is the winning competitor, add the course and associate the class
            if utilities.get_element_text_or_none(competitor, 'Result/ResultPosition') == '1':
                # Add new course
                cursor.execute(
                    'INSERT OR IGNORE INTO courses (controls) VALUES (?)',
                    (control_codes,))
                conn.commit()

                # Retrieve course_id
                cursor.execute('SELECT id FROM courses WHERE controls = ?', (control_codes,))
                course_id = cursor.fetchone()[0]

                # Associate course and class
                cursor.execute(
                    'INSERT OR IGNORE INTO races_courses_classes (race_id, course_id, class_id) VALUES (?, ?, ?)',
                    (race_id, course_id, class_id,))
                conn.commit()

            # Finally, link the competitor to the result
            # Retrieve competitor_id
            query_string, values = utilities.select_id_query_match_null('competitors',
                                                              ['given_name', 'surname', 'club'],
                                                              [given_name, surname, club])
            cursor.execute(query_string, values)
            competitor_id = cursor.fetchone()[0]

            # Associate
            cursor.execute(
                'INSERT OR IGNORE INTO races_competitors (race_id, competitor_id, class_id, card_no) VALUES (?, ?, ?, ?)',
                (race_id, competitor_id, class_id, card_no,))
            conn.commit()

    conn.close()
