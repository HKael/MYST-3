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
import pandas as pd
from data import data_pips0
import datetime
import time


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


# %%
# Function to get pip multiplier
def f_pip_size(param_ins, data_param2):
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

       comparison_file: DataFrame
        DataFrame where the list of pips corresponding the instrument are saved to compare.


    Returns:
    -------
        If an instrument name is entered, it returns a whole number, depending on the instrument, as a pip multiplier.
     """
    data_pips1 = data_param2
    piprow = data_pips1.loc[data_pips1["Instrument"] == param_ins]
    pipsize = 1 / piprow["TickSize"]
    return pipsize


# %%
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
    hd = hd.rename(columns = {"Time" : "Open Time","Time.1" : "Close Time", "Time3" : "Tiempo",
                              "Price" : "Open Price", "Price.1" : "Close Price" })
    return hd


# %%
# Function to add pip columns
def f_columnas_pips(param_data, comparison_file):
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
    rows = []
    for x in range(len(param_data)):
        rows.append(float(f_pip_size(param_data.loc[x, "Symbol"], comparison_file)))
    provdf = pd.DataFrame(rows, columns=["Pips"])
    param_data["Pips"] = (param_data["Close Price"] - param_data["Open Price"]) * (provdf['Pips'])
    param_data["Pips Accumulated"] = param_data["Pips"].cumsum()
    param_data["Profit Accumulated"] = param_data["Profit"].cumsum()
    return param_data


# Function to calculate basic statistics parameters
def f_estadisticas_ba(param_data):
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
    # DataFrame 1
    buy_count, sell_count, pos_count_sell, neg_count_sell, pos_count_buy, neg_count_buy = 0, 0, 0, 0, 0, 0
    profit_median = param_data["Profit"].median()
    pip_median = param_data["Pips"].median()
    # Number of operations
    total_count = (len(param_data))
    for x in range(total_count):
        # Count buy and sell operations
        if param_data.loc[x, "Type"] == "buy":
            buy_count += 1
            # Count positve and negative operations
            if param_data.loc[x, "Profit"] >= 0:
                pos_count_buy += 1

            else:
                neg_count_buy += 1
        else:
            sell_count += 1
            # Count positve and negative operations
            if param_data.loc[x, "Profit"] >= 0:
                pos_count_sell += 1

            else:
                neg_count_sell += 1

    r_efectividad = (pos_count_buy + pos_count_sell) / total_count
    r_proporcion = (pos_count_buy + pos_count_sell) / (neg_count_buy + neg_count_sell)
    r_efectividad_c = pos_count_buy / total_count
    r_efectividad_v = pos_count_sell / total_count

    medida = ["Ops Totales", "Ganadoras", "Ganadoras_c", "Ganadoras_v", "Perdedoras", "Perdedoras_c",
              "Perdedoras_v", "Mediana(Profit)", "Mediana(Pips)", "r_efectividad", "r_proporcion",
              "r_efectividad_c", "r_efectividad_v"]

    values = np.array([total_count, (pos_count_buy + pos_count_sell), pos_count_buy, pos_count_sell,
                       (neg_count_buy + neg_count_sell), neg_count_buy, neg_count_sell, profit_median,
                       pip_median, r_efectividad, r_proporcion, r_efectividad_c, r_efectividad_v])

    description = ["Operaciones Totales", "Operaciones Ganadoras", "Operaciones Ganadoras de Compra",
                   "Operaciones Ganadoras de Venta", "Operaciones Perdedoras", "Operaciones Perdedoras de Compra",
                   "Operaciones Perdedoras de Venta", "Mediana de Profit de Operaciones",
                   "Mediana de Pips de Operaciones", "Ganadoras Totales/Operaciones Totales",
                   "Ganadoras Totales/Perdedoras Totales", "Ganadoras Compras/Operaciones Totales",
                   "Ganadoras Ventas/Operaciones Totales"]

    df1 = pd.DataFrame({'Measurment': medida, 'Value': values, 'Description': description})

    # DatFrame 2
    symbols = np.unique(param_data['Symbol'])  # listo ya tenemos los unicos
    rank = []
    for i in range(len(symbols)):
        pos_count_rank = 0
        neg_count_rank = 0
        for j in range(len(param_data)):
            if symbols[i] == param_data['Symbol'].iloc[j] and param_data['Profit'].iloc[j] > 0:
                pos_count_rank += 1

            elif symbols[i] == param_data['Symbol'].iloc[j] and param_data['Profit'].iloc[j] < 0:
                neg_count_rank += 1

        total = neg_count_rank + pos_count_rank
        rank.append(pos_count_rank / total)

    df2 = pd.DataFrame({'Symbol': symbols, 'Rank': rank})
    df2 = df2.sort_values(by='Rank', ascending=False)

    # Combine both to make the dictionary
    dictionary = {'df_1_tabla': df1, 'df_2_ranking': df2}
    return dictionary

# %% Performance Attribution Measures


# %% Behavioral Finance
