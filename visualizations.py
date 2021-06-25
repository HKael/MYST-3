"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: visualizations.py : python script with data visualization functions                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import plotly.graph_objects as go


# %% Graph 1

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
