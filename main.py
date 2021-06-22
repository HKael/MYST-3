"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
#%%
import numpy as np
import pandas as pd
from data import data_historical, data_pips, data_pips0
from functions import f_columnas_pips, f_columnas_tiempos, f_estadisticas_ba, f_pip_size
import datetime
import time


#%%
historical_data_2 = f_columnas_tiempos(data_historical)
#%%
historical_data_3 = f_columnas_pips(historical_data_2, data_pips0)
#%%
historical_data_4 = f_estadisticas_ba(historical_data_3)





