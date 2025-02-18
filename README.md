# Introduction
This project is a collection of scripts to analyze and plot orienteering performances. Starting with an IOF XML event file (these can be downloaded with the free version of WinSplits desktop), the scripts do the following:

1. create_database.py
    - Parses multiple XML file into an sqlite database. One database per event
2. race_simulator.py (for simlulated races only)
    - Pulls start time and class data from an events database, then simulates the split times. Saves to pickle file
2. split_calcs.py
    - Pulls data from an events database or simulated race file and calculates split performances in a pandas dataframe. Saves the performance dataframe as a pickle file.
3. split_time_analysis.py
    - Loads a split performance pickle file and generates various plots
