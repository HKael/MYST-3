"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: data.py : python script for data collection                                                 -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import pandas as pd
import functions as fn


# Historical data from MetaTrader5, trades done throughout the period 06-08-21 / 06-16-21
data_historical = fn.f_leer_archivo("files/historical_data.csv")


# Data describing the instruments and their pips
data_pips = fn.f_leer_archivo("files/instruments_pips.csv").set_index("Symbol")


