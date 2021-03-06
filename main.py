"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: MyST LAB 3                                                                                 -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: HKael                                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/HKael/MyST_LAB_3_EAMM                                                -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
#%%
import numpy as np
import pandas as pd
from data import data_historical, data_pips, data_pips0
from functions import f_columnas_pips, f_columnas_tiempos
from functions import f_estadisticas_ba, f_pip_size, f_evolucion_capital, f_estadisticas_mad, f_be_de

import datetime
import time

# Part 1
#%%
historical_data_2 = f_columnas_tiempos(data_historical)
#%%
historical_data_3 = f_columnas_pips(historical_data_2, data_pips0)
#%%
historical_data_4 = f_estadisticas_ba(historical_data_3)

# Part 2
#%%
capital_evolution = f_evolucion_capital(historical_data_3)

#%%
mad = f_estadisticas_mad(capital_evolution)

#%%
be_fi = f_be_de(historical_data_3)




