"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: MyST LAB 3                                                                                 -- #
# -- script: visualizations.py : python script with data visualization functions                         -- #
# -- author: HKael                                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/HKael/MyST_LAB_3_EAMM                                                -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


# %% Graph 1
def g_1(param_data):
    df = param_data
    fig = go.Figure(data=[go.Pie(labels=df["Symbol"], values=df["Rank"], pull=[0.3])])
    fig.update_layout(title_text='Descriptive Statistics Tabla 2')
    return fig.show()


# %% Graph 2
def g_2(param_data):
    date_total = param_data.index
    date_up = param_data.index[:7]
    date_down = param_data.index[6:]

    v = param_data["Accumulated Profit"][0]
    v1 = (param_data["Accumulated Profit"][6] - param_data["Accumulated Profit"][0]) / 6
    b = param_data["Accumulated Profit"][6]
    b1 = (param_data["Accumulated Profit"][6] - param_data["Accumulated Profit"][12]) / 6
    profit_total = param_data["Accumulated Profit"]
    profit_up = [v, v + v1, v + v1 * 2, v + v1 * 3, v + v1 * 4, v + v1 * 5, v + v1 * 6, v + v1 * 7]
    profit_down = [b, b - b1, b - b1 * 2, b - b1 * 3, b - b1 * 4, b - b1 * 5, b - b1 * 6, b - b1 * 7]
    fig = go.Figure()
    # Create and style traces

    fig.add_trace(go.Scatter(x=date_total, y=profit_total, name='Acc Profit',
                             line=dict(color='black', width=4)))
    fig.add_trace(go.Scatter(x=date_up, y=profit_up, name='DrawUp',
                             line=dict(color='green', width=4, dash='dot')))
    fig.add_trace(go.Scatter(x=date_down, y=profit_down, name='DrawDown',
                             line=dict(color='red', width=4, dash='dot')))

    # Edit the layout
    fig.update_layout(title='Gráfica 2: DrawDown y DrawUp',
                      xaxis_title='Day',
                      yaxis_title='US Dollars')

    return fig.show()


# %% Extra Graph 1
def g_extra_1(param_data):
    df = param_data
    df.loc[7, "Value"] = 8.515
    df.loc[8, "Value"] = 8.36
    fig = px.bar(df, y='Value', x='Measurment', color="Value")
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

    # Change the bar mode
    fig.update_layout(barmode='group')
    fig.update_layout(title_text='Descriptive Statistics Tabla 1')
    return fig.show()


# %% Extra Graph 2
def g_extra_2(param_data):
    df = param_data
    fig = px.bar(df, y='Profit', x='Symbol', text='Symbol')
    fig.update_traces(marker_color='indianred')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    return fig.show()


# %% Extra Graph 3
def g_extra_3(param_data):
    df = param_data
    df["Profit"] = np.log(df["Profit"])
    fig = px.bar(df, y='Profit', x='Symbol', text='Symbol')

    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    return fig.show()


# %% Extra Graph 4 MAD
def g_extra_4(param_data):
    years = ['Sharpe Original Vs Actualizado', 'DrawnDown vs DrawnUp Capital $ (100s Scaled)']
    fig = go.Figure()
    fig.add_trace(go.Bar(x=years, y=[param_data["Value"][0], param_data["Value"][4] / 100],

                         marker_color='crimson',
                         name='Right Side'))
    fig.add_trace(go.Bar(x=years, y=[param_data["Value"][1], param_data["Value"][7] / 100],

                         marker_color='lightblue',
                         name='Left Side'
                         ))

    # Change the bar mode
    fig.update_layout(title_text='MAD Extra Table')
    fig.update_layout(barmode='group')
    return fig.show()


# %%
def g_3():
    eje_x = ['Status Quo', 'Aversión Perdida', "Sensibilidad Decreciente"]
    fig = go.Figure([go.Bar(x=eje_x, y=[16.6667, 16.6667, 0])])
    fig.update_layout(title_text='Behavioral Finance Table')
    return fig.show()
