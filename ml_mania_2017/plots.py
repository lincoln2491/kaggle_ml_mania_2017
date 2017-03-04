from plotly import offline as plt
from plotly import graph_objs as go


def plot_opposite_features(data, feature):
    df_l = data[data.result == 1]
    df_h = data[data.result == 0]
    trace_l = go.Scatter(x=df_l['l_' + feature], y=df_l['h_' + feature], mode='markers', name='l')
    trace_h = go.Scatter(x=df_h['l_' + feature], y=df_h['h_' + feature], mode='markers', name='h')
    trace_line = go.Scatter(x=range(int(data['l_' + feature].max())), y=range(int(data['l_' + feature].max())), mode="lines")
    plt.plot([trace_l, trace_h, trace_line])
