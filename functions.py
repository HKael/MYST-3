"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: MyST LAB 3                                                                                 -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: HKael                                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/HKael/MyST_LAB_3_EAMM                                                -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import numpy as np
import pandas as pd
from data import data_pips0
import datetime
import time
import pandas_datareader as web


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
    hd = hd.rename(columns={"Time": "Open Time", "Time.1": "Close Time", "Time3": "Tiempo",
                            "Price": "Open Price", "Price.1": "Close Price"})
    return hd


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

def f_evolucion_capital(param_data):
    """
    Name:
    -------
        f_evolucion_capital


    Description:
    -------
        Calculates the profit and accumulated profit by date.



    Parameters:
    -------
        param_data: Data Frame


    Returns:
    -------
        Gets a DataFrame with three columns, date, profit, accumulated profit.

     """
    historical_profit = param_data[["Close Time", "Profit"]]
    historical_profit["Close Time"] = pd.to_datetime(historical_profit["Close Time"]).dt.date
    historical_profit = historical_profit.groupby("Close Time").sum()
    historical_profit = historical_profit.asfreq('D')
    historical_profit["Profit"] = historical_profit["Profit"].fillna(0)
    historical_profit["Accumulated Profit"] = historical_profit["Profit"].cumsum() + 100000
    return historical_profit


# %%
def f_estadisticas_mad(param_data):
    """
        Name:
        -------
            f_estadisticas_mad


        Description:
        -------
            A function that returns Performance Attribution Metrics.



        Parameters:
        -------
            param_data: Data Frame


        Returns:
        -------
            Returns a DataFrame with four columns. Performance Attribution Metrics, which are Sharpe Ratio,
            Adjusted Sharpe Ratio, DrawDown and DrawUp's Capital, initial and last date.

    """

    # Sharpe Ratio Original
    log_yield = np.log(param_data['Accumulated Profit'] / param_data['Accumulated Profit'].shift()).dropna()
    rp = log_yield.mean() * 360
    risk_free = .05
    sdp = log_yield.std()
    sharp_r_o = (rp - risk_free) / sdp

    # Sharpe Ratio Ajustado
    price_data_sharp_2 = web.get_data_yahoo("SPY",
                                            start=param_data.index[0],
                                            end=param_data.index[-1], interval='d')
    price_data_sharp_2 = price_data_sharp_2['Adj Close']
    ret_price_data_sharp_2 = np.log(price_data_sharp_2 / price_data_sharp_2.shift()).dropna()
    r_trader = rp
    r_benchmark = ret_price_data_sharp_2.mean()
    sdp_2 = ret_price_data_sharp_2.std()
    sharp_r_o_2 = (r_trader - r_benchmark) / sdp_2

    # DrawDown
    k6 = param_data.reset_index()
    w = k6[k6["Accumulated Profit"] == k6["Accumulated Profit"].max()].index
    f = w[0]
    if f < (len(k6["Accumulated Profit"]) - 1):
        maximum = k6["Accumulated Profit"].max()
        minimum = k6["Accumulated Profit"][f:].min()
        d_down = minimum - maximum
    else:
        new_param = k6.shift().reset_index()
        maximum = k6["Accumulated Profit"].max()
        w2 = new_param[new_param["Accumulated Profit"] == new_param["Accumulated Profit"].max()].index
        f2 = w2[0]
        minimum = new_param["Accumulated Profit"][f2:].min()
        d_down = minimum - maximum

    index_d_down = param_data.index
    condition_d_down_initial = param_data["Accumulated Profit"] == maximum
    condition_d_down_last = param_data["Accumulated Profit"] == minimum
    d_down_initial = index_d_down[condition_d_down_initial].tolist()
    d_down_last = index_d_down[condition_d_down_last].tolist()

    # DrawUp
    w = k6[k6["Accumulated Profit"] == k6["Accumulated Profit"].min()].index
    f = w[0]

    if f < (len(k6["Accumulated Profit"]) - 1):
        maximum = k6["Accumulated Profit"].max()
        minimum = k6["Accumulated Profit"][f:].min()
        d_up = maximum - minimum
    else:
        new_param = k6.shift().reset_index()
        maximum = k6["Accumulated Profit"].max()
        w2 = new_param[new_param["Accumulated Profit"] == new_param["Accumulated Profit"].min()].index
        f2 = w2[0]
        minimum = new_param["Accumulated Profit"][f2:].min()
        d_up = maximum - minimum

    index_d_up = param_data.index
    condition_d_up_initial = param_data["Accumulated Profit"] == minimum
    condition_d_up_last = param_data["Accumulated Profit"] == maximum
    d_up_initial = index_d_up[condition_d_up_initial].tolist()
    d_up_last = index_d_up[condition_d_up_last].tolist()

    # DataFrame
    mad_stats = pd.DataFrame(columns=["Metric", "Measurment", "Value", "Description"])
    mad_stats["Metric"] = ["Sharpe Original", "Sharpe Actualizado", "DrawDown Capital", "DrawDown Capital",
                           "DrawDown Capital", "DrawUp Capital", "DrawUp Capital", "DrawUp Capital"]
    mad_stats["Measurment"] = ["Amount", "Amount", "Initial Date", "Last Date", "DrawDown Capital $",
                               "Initial Date", "Last Date", "DrawUp Capital $"]
    mad_stats["Value"] = [sharp_r_o, sharp_r_o_2, d_down_initial, d_down_last, d_down, d_up_initial, d_up_last, d_up]
    mad_stats["Description"] = ["Sharpe Ratio Original Formula", "Sharpe Ratio Adjusted Formula",
                                "Initial Date of Capital's DrawDown", "Last Date of Capital's DrawDown",
                                "Maximum floating loss recorded", "Initial Date of Capital's DrawUp",
                                "Last Date of Capital's DrawUp", "Maximum floating profit recorded"]
    return mad_stats


# %% Behavioral Finance
def f_be_de(param_data):
    # Profit Ratio
    param_data["Profit GP"] = param_data["Profit"] / param_data["Profit Accumulated"] * 100

    # Profitable Operations
    g_op = param_data[param_data["Profit"] > 0].reset_index(drop=True)
    g_op["Close Time"] = pd.to_datetime(g_op["Close Time"], unit='s')
    g_op["Open Time"] = pd.to_datetime(g_op["Open Time"], unit='s')

    # Non profitable Operations
    p_op = param_data[param_data["Profit"] < 0].reset_index(drop=True)
    p_op["Close Time"] = pd.to_datetime(p_op["Close Time"], unit='s')
    p_op["Open Time"] = pd.to_datetime(p_op["Open Time"], unit='s')
    # Punto de referencia
    result_p = []
    for x in range(len(p_op)):
        a = g_op["Close Time"].between(p_op["Open Time"][x], p_op["Close Time"][x])
        result_p.append(a)
    res_p_df = pd.DataFrame(np.array(result_p)).transpose()

    # Aversion a la perdida
    result_g = []
    for x in range(len(g_op)):
        wu = abs(p_op["Profit GP"] / g_op["Profit GP"][x])
        wu = wu > 2
        result_g.append(wu)
    #
    res_g_df = pd.DataFrame(result_g).reset_index(drop=True)

    res_g_df.loc[(res_g_df[2] == True)]

    inters_v = np.intersect1d(np.array(res_p_df.loc[(res_p_df[2] == True)].index),
                              np.array(res_g_df.loc[(res_g_df[2] == True)].index))
    inter_g_p_1 = g_op.loc[inters_v[0]]
    inter_g_p_2 = g_op.loc[inters_v[1]]
    inter_v_p = p_op.loc[2]

    # Sensibilidad decreciente
    con_1 = g_op["Profit Accumulated"].loc[0] < g_op["Profit Accumulated"].loc[len(g_op) - 1]
    con_2 = (g_op["Profit GP"].loc[0] < g_op["Profit GP"].loc[len(g_op) - 1]) & (
            p_op["Profit GP"].loc[0] < p_op["Profit GP"].loc[len(p_op) - 1])
    con_3 = abs(p_op["Profit GP"].loc[len(p_op) - 1] / g_op["Profit GP"].loc[len(g_op) - 2])

    if con_1 == True and con_2 == True:
        sensibilidad_decreciente = "Si"
    else:
        if con_1 == True and con_3 > 2:
            sensibilidad_decreciente = "Si"
        else:
            if con_2 == True and con_3 > 2:
                sensibilidad_decreciente = "Si"
            else:
                sensibilidad_decreciente = "No"

            # Operaciones de ocurrencia
    g_ts_1 = inter_g_p_1["Close Time"]
    g_ts_1 = g_ts_1.strftime("%m/%d/%Y, %H:%M:%S")
    g_instrumento_1 = inter_g_p_1["Symbol"]
    g_volumen_1 = inter_g_p_1["Volume"]
    g_sentido_1 = inter_g_p_1["Type"]
    g_profit_1 = inter_g_p_1["Profit"]

    p_instrumento_1 = inter_v_p["Symbol"]
    p_volumen_1 = inter_v_p["Volume"]
    p_sentido_1 = inter_v_p["Type"]
    p_profit_1 = inter_v_p["Profit"]

    # Ratio
    ratio_cp_profit_acm_1 = abs(inter_v_p["Profit"] / inter_v_p["Profit Accumulated"])  # Momento de sesgo negativo
    ratio_cg_profit_acm_1 = abs(inter_g_p_1["Profit"] / inter_g_p_1["Profit Accumulated"])  # Ancla
    ratio_cp_cg_1 = abs(inter_v_p["Profit"] / inter_g_p_1["Profit"])  # Ancla/Momento de sesgo negativo

    # Operaciones de ocurrencia
    g_ts_2 = inter_g_p_2["Close Time"]
    g_ts_2 = g_ts_2.strftime("%m/%d/%Y, %H:%M:%S")
    g_instrumento_2 = inter_g_p_2["Symbol"]
    g_volumen_2 = inter_g_p_2["Volume"]
    g_sentido_2 = inter_g_p_2["Type"]
    g_profit_2 = inter_g_p_2["Profit"]

    p_instrumento_2 = inter_v_p["Symbol"]
    p_volumen_2 = inter_v_p["Volume"]
    p_sentido_2 = inter_v_p["Type"]
    p_profit_2 = inter_v_p["Profit"]

    # Ratio
    ratio_cp_profit_acm_2 = abs(inter_v_p["Profit"] / inter_v_p["Profit Accumulated"])  # Momento de sesgo negativo
    ratio_cg_profit_acm_2 = abs(inter_g_p_2["Profit"] / inter_g_p_2["Profit Accumulated"])  # Ancla
    ratio_cp_cg_2 = abs(inter_v_p["Profit"] / inter_g_p_2["Profit"])  # Ancla/Momento de sesgo negativo

    dic_len = len(inters_v) - 1

    r_df_bf = pd.DataFrame(columns=["Ocurrencias", "Status Quo %", "Aversion Perdida %", "Sensibilidad Decreciente"])
    r_df_bf["Ocurrencias"] = [dic_len]
    r_df_bf["Status Quo %"] = [dic_len / len(g_op) * 100]
    r_df_bf["Aversion Perdida %"] = [dic_len / len(g_op) * 100]
    r_df_bf["Sensibilidad Decreciente"] = [sensibilidad_decreciente]

    dict_sesgos = {'Ocurrencias':
                       {'Cantidad': dic_len, "Ocurrencia 1":
                           {'Timestamp': g_ts_1, "Operaciones":
                               {"Ganadora": {"Instrumento": g_instrumento_1, "Volumen": g_volumen_1,
                                             "Sentido": g_sentido_1, "Profit Ganadora": g_profit_1},
                                "Perdedora": {"Instrumento": p_instrumento_1, "Volumen": p_volumen_1,
                                              "Sentido": p_sentido_1, "Profit Perdedora": p_profit_1}},
                            'ratio_cp_profit_acm': ratio_cp_profit_acm_1, 'ratio_cg_profit_acm': ratio_cg_profit_acm_1,
                            'ratio_cp_cg': ratio_cp_cg_1},
                        "Ocurrencia 2":
                            {'Timestamp': g_ts_2, "Operaciones":
                                {"Ganadora": {"Instrumento": g_instrumento_2, "Volumen": g_volumen_2,
                                              "Sentido": g_sentido_2, "Profit Ganadora": g_profit_2},
                                 "Perdedora": {"Instrumento": p_instrumento_2, "Volumen": p_volumen_2,
                                               "Sentido": p_sentido_2, "Profit Perdedora": p_profit_2}},
                             'ratio_cp_profit_acm': ratio_cp_profit_acm_2, 'ratio_cg_profit_acm': ratio_cg_profit_acm_2,
                             'ratio_cp_cg': ratio_cp_cg_2}
                        }, "Resultados": r_df_bf}

    return dict_sesgos
