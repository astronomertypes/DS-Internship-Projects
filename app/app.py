"""Step 9b: Plotly Dash dashboard - forecasts, accuracy, order recommendations."""
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output
from dash.exceptions import PreventUpdate

D = Path(__file__).parent / "data"
hist = pd.read_parquet(D / "history.parquet")
fc = pd.read_parquet(D / "forecasts.parquet")
hist["date"], fc["date"] = pd.to_datetime(hist["date"]), pd.to_datetime(fc["date"])
metrics = pd.read_csv(D / "metrics.csv").round(4)
inventory = pd.read_csv(D / "inventory_results.csv").round(3)
orders = pd.read_csv(D / "order_recommendations.csv")

meta = hist[["id", "item_id", "store_id"]].drop_duplicates()
stores = sorted(meta["store_id"].unique())

app = Dash(__name__)
server = app.server  # used by gunicorn in Docker

app.layout = html.Div(style={"maxWidth": "1100px", "margin": "auto", "fontFamily": "sans-serif", "padding": "16px"}, children=[
    html.H2("Demand Forecasting & Inventory Optimization (M5 / Walmart, FOODS_3 subset)"),
    html.Div(style={"display": "flex", "gap": "12px"}, children=[
        dcc.Dropdown(stores, stores[0], id="store", clearable=False, style={"width": "180px"}),
        dcc.Dropdown(id="item", clearable=False, style={"width": "300px"}),
    ]),
    dcc.Graph(id="chart"),
    html.H3("Forecast accuracy on the last 28 days (lower is better)"),
    dash_table.DataTable(metrics.to_dict("records"), [{"name": c, "id": c} for c in metrics.columns]),
    html.H3("Inventory policy comparison (on actual sales)"),
    dash_table.DataTable(inventory.to_dict("records"), [{"name": c, "id": c} for c in inventory.columns]),
    html.H3("Order recommendations for the selected store"),
    dash_table.DataTable(id="orders", page_size=15, sort_action="native",
                         columns=[{"name": c, "id": c} for c in
                                  ["item_id", "on_hand", "forecast_lo", "forecast_mid", "forecast_hi", "order_qty"]]),
])


@app.callback(Output("item", "options"), Output("item", "value"), Input("store", "value"))
def fill_items(store):
    items = sorted(meta.loc[meta["store_id"] == store, "item_id"])
    return items, items[0]


@app.callback(Output("chart", "figure"), Output("orders", "data"), Input("store", "value"), Input("item", "value"))
def update(store, item):
    if not item:
        raise PreventUpdate
    sid = f"{item}_{store}_evaluation"
    h, f = hist[hist["id"] == sid].sort_values("date"), fc[fc["id"] == sid].sort_values("date")
    fig = go.Figure()
    fig.add_scatter(x=h["date"], y=h["sales"], name="Actual", line=dict(color="black"))
    fig.add_scatter(x=list(f["date"]) + list(f["date"][::-1]), y=list(f["lgbm_hi"]) + list(f["lgbm_lo"][::-1]),
                    fill="toself", name="LightGBM 80% interval", line=dict(width=0), opacity=0.25)
    for col, label in (("naive", "Seasonal naive"), ("prophet", "Prophet"), ("lgbm", "LightGBM"),
                       ("lgbm_reconciled", "LightGBM reconciled"), ("tft", "TFT")):
        fig.add_scatter(x=f["date"], y=f[col], name=label)
    fig.update_layout(title=f"{item} at {store}: last 120 days + 28-day forecast", yaxis_title="units/day", height=450)
    o = orders[orders["store_id"] == store].sort_values("order_qty", ascending=False).round(1)
    return fig, o.to_dict("records")


if __name__ == "__main__":
    app.run(debug=True, port=8050)