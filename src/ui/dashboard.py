# project/src/ui/dashboard.py

import os
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd

# Assume we have a CSV file with historical data, signals, trades, and confidence scores.
# This CSV might be generated from your backtest results or saved live trading data.
DATA_PATH = os.path.join(os.getcwd(), "data", "features", "results.csv")

# results.csv columns example:
# time, open, high, low, close, volume, signal, confidence, trade_side, trade_price, cumulative_pnl

df = pd.read_csv(DATA_PATH, parse_dates=['time']).set_index('time')

# Create candlestick figure
fig = go.Figure(
    data=[
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price"
        )
    ]
)

# Add a scatter plot for signals (e.g., buy = green up arrow, sell = red down arrow)
buy_signals = df[df['signal'] == 'buy']
sell_signals = df[df['signal'] == 'sell']

fig.add_trace(
    go.Scatter(
        x=buy_signals.index,
        y=buy_signals['close'],
        mode='markers',
        marker_symbol='triangle-up',
        marker_color='green',
        marker_size=10,
        name="Buy Signals"
    )
)

fig.add_trace(
    go.Scatter(
        x=sell_signals.index,
        y=sell_signals['close'],
        mode='markers',
        marker_symbol='triangle-down',
        marker_color='red',
        marker_size=10,
        name="Sell Signals"
    )
)

# Add confidence scores as a separate subplot (secondary y-axis or a separate figure)
confidence_trace = go.Scatter(
    x=df.index,
    y=df['confidence'],
    mode='lines',
    name='Model Confidence',
    line=dict(color='blue')
)

# For simplicity, just overlay on same chart with secondary y-axis
fig.update_layout(
    yaxis2=dict(
        overlaying='y',
        side='right',
        position=1.0,
        title='Confidence'
    )
)

fig.add_trace(confidence_trace, secondary_y=True)

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("AI-Driven Trading Strategy Dashboard"),
    html.Div([
        dcc.Graph(
            id='price-chart',
            figure=fig,
            style={'width': '90vw', 'height': '70vh'}
        )
    ]),
    html.Div([
        html.Label("Label Selected Point:"),
        dcc.Dropdown(
            id='label-dropdown',
            options=[
                {'label': 'Ideal Buy', 'value': 'ideal_buy'},
                {'label': 'Ideal Sell', 'value': 'ideal_sell'}
            ],
            value='ideal_buy'
        ),
        html.Button("Annotate", id='annotate-button'),
        html.Div(id='annotation-status')
    ]),
    html.Div([
        html.H3("Performance Summary"),
        # Display some performance metrics (assume they are precomputed and stored as columns)
        html.P(f"Total Return: {df['cumulative_pnl'].iloc[-1]:.2f}"),
        html.P(f"Sharpe Ratio: ..."), # You can compute this beforehand
        html.P(f"Max Drawdown: ..."),
        html.P(f"Win Rate: ...")
    ])
])

# Suppose we want to let users click on the chart and record the annotation
# The dash figure supports a clickData property
@app.callback(
    Output('annotation-status', 'children'),
    Input('annotate-button', 'n_clicks'),
    State('price-chart', 'clickData'),
    State('label-dropdown', 'value')
)
def annotate_point(n_clicks, click_data, label):
    if n_clicks is None:
        return ""
    if not click_data:
        return "Click on the chart to select a point before annotating."

    # Extract x-coordinate (time) from clickData
    clicked_time_str = click_data['points'][0]['x']
    clicked_time = pd.to_datetime(clicked_time_str)

    # Append annotation to a CSV or database
    annotations_file = os.path.join(os.getcwd(), 'data', 'features', 'annotations.csv')
    new_annotation = pd.DataFrame([[clicked_time, label]], columns=['time', 'label'])
    if os.path.exists(annotations_file):
        ann_df = pd.read_csv(annotations_file, parse_dates=['time'])
        ann_df = pd.concat([ann_df, new_annotation], ignore_index=True)
    else:
        ann_df = new_annotation

    ann_df.to_csv(annotations_file, index=False)
    return f"Annotated {clicked_time_str} as {label}."

if __name__ == "__main__":
    app.run_server(debug=True)
