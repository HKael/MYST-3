"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import numpy as np
import data as dt
import pandas as pd


# %% Descriptive statistics

# Function to read the files
def f_leer_archivo(file_path):
    """
    Name:
    -------

         f_leer_archivo

    Description:
    -------

         Function that reads a csv file and returns a Data Frame.

    Parameters:
    -------
        file_path: str
            String with the file name and the directory if needed

    Returns:
    -------
        DataFrame with the historical data from MetaTrader5.
     """
    return pd.read_csv(file_path)


# Function to get pip multiplier
def f_pip_size(param_ins):
    """
    Name:
    -------
        f_pip_size


    Description:
    -------
        Function that obtains the multiplier number to express the price difference in pips.


    Parameters:
    -------
       param_ins: str
        Name of the instrument to be associated with the corresponding pip multiplier.


    Returns:
    -------
        If an instrument name is entered, it returns a whole number, depending on the instrument, as a pip multiplier.
     """
    data_pips = dt.data_pips
    piprow = data_pips.loc[data_pips["Instrument"] == param_ins]
    pipsize = 1/piprow["TickSize"]
    pipsize = float(pipsize)
    return pipsize


# Function to add and format time columns
def f_columnas_tiempos(param_data):
    """
    Name:
    -------
        f_columnas_tiempos



    Description:
    -------
        Transforms closetime and opentime columns to datetime64 format.
        Adds time column, measures seconds of difference between closetime and opentime columns


    Parameters:
    -------
        param_data: DataFrame
            Base DataFrame where the columns are modified and added.

    Returns:
    -------
        closetime and opentime as "datetime64" type columns, a time column that is the seconds
        that each operation (each line) remained open.
     """
    hd = param_data
    hd["Time"] = pd.to_datetime(hd["Time"])
    hd["Time.1"] = pd.to_datetime(hd["Time.1"])
    hd["Time3"] = hd["Time.1"] - hd["Time"]
    hd["Time3"] = hd["Time3"].dt.total_seconds()
    hd = hd.rename(columns = {"Time" : "Open Time","Time.1" : "Close Time", "Time3" : "Tiempo" })
    return hd


# Function to add pip columns
def f_columnas_pips():
    """
     Name:
     -------
        f_columnas_pips


    Description:
    -------
        Add more columns of pips transformations


    Parameters:
    -------
        param_data: DataFrame

    Returns:
    -------
        pips: it is a column where the number of pips resulting from each operation must be, including its sign.
        The best way to properly validate your calculation in code is to do it manually with your computer's calculator
        software.

        When it is purchase: (closeprice - openprice) * multiplier

        When it is sale: (openprice - closeprice) * multiplier.

        pips_acm: The accumulated value of the pips column

        profit_acm: The accumulated value of the profit column

     """
    return


# Function to calculate basic statistics parameters
def f_estadisticas_ba():
    """
    Name:
    -------
        f_estadisticas_ba


    Description:
    -------
        Calculate basic statistics and ranking by instruments



    Parameters:
    -------
        param_data: Data Frame


    Returns:
    -------
        Gets a dictionary with the keys 'df_1_table' and 'df_2_ranking'.

     """
    return

# %% Performance Attribution Measures


# %% Behavioral Finance
