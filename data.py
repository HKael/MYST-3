"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: data.py : python script for data collection                                                 -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
#%%
import pandas as pd


#%%
# Historical data from MetaTrader5, trades done throughout the period 06-08-21 / 06-16-21
data_historical = pd.read_csv("files/historical_data.csv")

#%%
# Data describing the instruments and their pips
data_pips0 = pd.read_csv("files/instruments_pips.csv")
data_pips = data_pips0.set_index("Instrument")


