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


def f_pip_size():
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
    return


def f_columnas_tiempos():
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
    return


def f_columnas_pips():
    """
     Name:
     -------



     Description:
     -------



     Parameters:
     -------


    Returns:
     -------

     """
    return


def f_estadisticas_ba():
    """
     Name:
     -------



     Description:
     -------



     Parameters:
     -------


    Returns:
     -------

     """
    return



