"""Step 9b: Plotly Dash dashboard (dark lavender / purple / pink theme)."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html
from dash.dash_table.Format import Format, Scheme
from dash.exceptions import PreventUpdate

# ------------------------------------------------------------------ palette
TEXT, MUTED = "#f5f3ff", "#b8accf"
LAV, PURPLE, PINK, PINK_SOFT = "#c4b5fd", "#8b5cf6", "#f472b6", "#f9a8d4"
GRID = "rgba(196,181,253,0.12)"
FONT = '"Inter", "Segoe UI", system-ui, -apple-system, sans-serif'

# ------------------------------------------------------------------ data (exported by src/export_app_data.py)
D = Path(__file__).parent / "data"
hist = pd.read_parquet(D / "history.parquet")
fc = pd.read_parquet(D / "forecasts.parquet")
hist["date"] = pd.to_datetime(hist["date"])
fc["date"] = pd.to_datetime(fc["date"])
metrics = pd.read_csv(D / "metrics.csv")
inv = pd.read_csv(D / "inventory_results.csv")
orders = pd.read_csv(D / "order_recommendations.csv")
rec = pd.read_csv(D / "reconciliation_metrics.csv")

meta = hist[["id", "item_id", "store_id"]].drop_duplicates()
SID = {(r.item_id, r.store_id): r.id for r in meta.itertuples()}   # (item, store) -> series id
stores = sorted(meta["store_id"].unique())
n_items, n_stores = meta["item_id"].nunique(), len(stores)

# ------------------------------------------------------------------ headline numbers
best = metrics.loc[metrics["WAPE_item_store"].idxmin()]

_cov = fc.merge(hist[["id", "date", "sales"]], on=["id", "date"], how="inner")
coverage = float(((_cov["sales"] >= _cov["lgbm_lo"]) & (_cov["sales"] <= _cov["lgbm_hi"])).mean())


def _policy(keyword):
    m = inv[inv["policy"].str.contains(keyword, case=False, regex=False)]
    return m.iloc[0] if len(m) else None


opt, eq = _policy("Optimized"), _policy("scaled")
if opt is not None and eq is not None:
    cost_delta = f"{opt['total_cost'] / eq['total_cost'] - 1:+.1%}"
    fill_val = f"{opt['fill_rate']:.1%}"
    fill_sub = f"vs {eq['fill_rate']:.1%} for naive at the same spend"
else:
    cost_delta, fill_val, fill_sub = "–", "–", "run src/optimize.py first"


# ------------------------------------------------------------------ small layout helpers
def kpi(label, value, sub):
    return html.Div([html.Div(label, className="kpi-label"),
                     html.Div(value, className="kpi-value"),
                     html.Div(sub, className="kpi-sub")], className="kpi")


def chip(label, value, accent=False):
    return html.Div([html.Div(label, className="chip-label"),
                     html.Div(value, className="chip-value")],
                    className="chip chip-accent" if accent else "chip")


def card(title, sub, children):
    return html.Div([html.H2(title, className="section-title"),
                     html.P(sub, className="section-sub"), *children], className="card")


# ------------------------------------------------------------------ shared table styling
HEADER_STYLE = {
    "backgroundColor": "#2a1a4e", "color": LAV, "fontWeight": "700", "fontFamily": FONT,
    "fontSize": "11px", "textTransform": "uppercase", "letterSpacing": "0.07em",
    "padding": "12px 14px", "border": "none", "borderBottom": "1px solid rgba(196,181,253,0.35)",
}
CELL_STYLE = {
    "backgroundColor": "transparent", "color": TEXT, "fontFamily": FONT, "fontSize": "14px",
    "padding": "11px 14px", "border": "none", "borderBottom": "1px solid rgba(196,181,253,0.10)",
    "textAlign": "right", "whiteSpace": "normal", "height": "auto",
}
ZEBRA = {"if": {"row_index": "odd"}, "backgroundColor": "rgba(255,255,255,0.03)"}


def table_kwargs(left_cols):
    return dict(
        style_as_list_view=True,
        cell_selectable=False,
        style_table={"overflowX": "auto"},
        style_header=HEADER_STYLE,
        style_cell=CELL_STYLE,
        style_cell_conditional=[{"if": {"column_id": c}, "textAlign": "left", "fontWeight": "600"}
                                for c in left_cols],
    )


BEST_CELL = {"color": PINK, "fontWeight": "800", "backgroundColor": "rgba(244,114,182,0.12)"}

# ---- model accuracy table (best value per column highlighted)
M_COLS = ["WAPE_item_store", "WAPE_store", "RMSSE"]
m_disp = pd.DataFrame({"model": metrics["model"]})
for c in M_COLS:
    m_disp[c] = metrics[c].map("{:.4f}".format)
m_cond = [ZEBRA] + [{"if": {"row_index": int(metrics[c].idxmin()), "column_id": c}, **BEST_CELL} for c in M_COLS]
metrics_table = dash_table.DataTable(
    data=m_disp.to_dict("records"),
    columns=[{"name": "Model", "id": "model"}, {"name": "WAPE · item-store", "id": "WAPE_item_store"},
             {"name": "WAPE · store", "id": "WAPE_store"}, {"name": "RMSSE", "id": "RMSSE"}],
    style_data_conditional=m_cond, **table_kwargs(["model"]))

# ---- reconciliation table (coherence gap = how far levels are from adding up)
LEVELS = [l for l in ("total", "state", "store", "item_store") if l in set(rec["level"])]
LEVEL_NAMES = {"total": "Total", "state": "State", "store": "Store", "item_store": "Item-store"}
REC_NAMES = {"base (unreconciled)": "Base (unreconciled)", "bottom_up": "Bottom-up",
             "mint_wls_struct": "MinT (wls_struct)"}
piv = rec.pivot_table(index="forecast", columns="level", values="WAPE", aggfunc="first")[LEVELS]
piv["gap"] = rec.groupby("forecast")["coherence_gap"].first()
piv = piv.reset_index()
r_disp = pd.DataFrame({"forecast": piv["forecast"].map(lambda s: REC_NAMES.get(s, s))})
for l in LEVELS:
    r_disp[l] = piv[l].map("{:.4f}".format)
r_disp["gap"] = piv["gap"].map("{:,.2f}".format)
r_cond = [ZEBRA] + [{"if": {"row_index": int(piv[l].idxmin()), "column_id": l}, **BEST_CELL} for l in LEVELS]
r_cond += [{"if": {"row_index": i, "column_id": "gap"}, "color": PINK, "fontWeight": "800"}
           for i, s in enumerate(piv["forecast"]) if str(s).startswith("base")]
recon_table = dash_table.DataTable(
    data=r_disp.to_dict("records"),
    columns=[{"name": "Forecast", "id": "forecast"}] + [{"name": LEVEL_NAMES[l], "id": l} for l in LEVELS]
            + [{"name": "Coherence gap (units)", "id": "gap"}],
    style_data_conditional=r_cond, **table_kwargs(["forecast"]))

# ---- inventory policy table
i_disp = pd.DataFrame({
    "policy": inv["policy"],
    "holding_cost": inv["holding_cost"].map("{:,.0f}".format),
    "stockout_cost": inv["stockout_cost"].map("{:,.0f}".format),
    "total_cost": inv["total_cost"].map("{:,.0f}".format),
    "fill_rate": inv["fill_rate"].map("{:.1%}".format),
    "order_spend": inv["order_spend"].map("{:,.0f}".format),
})
i_cond = [ZEBRA,
          {"if": {"filter_query": '{policy} contains "Optimized"'},
           "backgroundColor": "rgba(139,92,246,0.22)", "fontWeight": "700"},
          {"if": {"row_index": int(inv["total_cost"].idxmin()), "column_id": "total_cost"}, **BEST_CELL}]
inventory_table = dash_table.DataTable(
    data=i_disp.to_dict("records"),
    columns=[{"name": "Policy", "id": "policy"}, {"name": "Holding cost", "id": "holding_cost"},
             {"name": "Stockout cost", "id": "stockout_cost"}, {"name": "Total cost", "id": "total_cost"},
             {"name": "Fill rate", "id": "fill_rate"}, {"name": "Order spend", "id": "order_spend"}],
    style_data_conditional=i_cond, **table_kwargs(["policy"]))

# ---- orders table (filled by the callback)
ORDER_COLS = [("item_id", "Item", None), ("on_hand", "On hand", 0), ("forecast_lo", "Demand p10", 1),
              ("forecast_mid", "Expected demand", 1), ("forecast_hi", "Demand p90", 1), ("order_qty", "Order qty", 0)]
ORDERS_BASE_COND = [ZEBRA, {"if": {"column_id": "order_qty"}, "color": LAV, "fontWeight": "700"}]
orders_table = dash_table.DataTable(
    id="orders", data=[],
    columns=[{"name": n, "id": i} if p is None else
             {"name": n, "id": i, "type": "numeric", "format": Format(precision=p, scheme=Scheme.fixed)}
             for i, n, p in ORDER_COLS],
    sort_action="native", page_action="none", fixed_rows={"headers": True},
    style_data_conditional=ORDERS_BASE_COND,
    **{**table_kwargs(["item_id"]),
       "style_table": {"maxHeight": "460px", "overflowY": "auto", "overflowX": "auto"}})

# ------------------------------------------------------------------ the chart
MODEL_LINES = [  # column, legend label, colour, dash, width
    ("naive", "Seasonal naive", "#9b8ec4", "dot", 1.6),
    ("prophet", "Prophet", PURPLE, "solid", 2.0),
    ("tft", "TFT", LAV, "solid", 2.0),
]


def make_figure(h, f):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["date"], y=h["sales"], name="Actual", mode="lines",
                             line=dict(color=TEXT, width=2)))
    fig.add_trace(go.Scatter(                                   # LightGBM 80% interval band
        x=list(f["date"]) + list(f["date"][::-1]),
        y=list(f["lgbm_hi"]) + list(f["lgbm_lo"][::-1]),
        fill="toself", fillcolor="rgba(244,114,182,0.16)", line=dict(width=0),
        name="LightGBM 80% interval", hoverinfo="skip"))
    lines = list(MODEL_LINES)
    same = np.allclose(f["lgbm"].values, f["lgbm_reconciled"].values, equal_nan=True)
    if same:   # bottom-up reconciliation leaves item-level forecasts unchanged, so draw one line
        lines.append(("lgbm", "LightGBM (= reconciled)", PINK, "solid", 2.6))
    else:
        lines.append(("lgbm", "LightGBM", PINK, "solid", 2.6))
        lines.append(("lgbm_reconciled", "LightGBM reconciled", PINK_SOFT, "dash", 2.0))
    for col, label, color, dash, width in lines:
        fig.add_trace(go.Scatter(x=f["date"], y=f[col], name=label, mode="lines",
                                 line=dict(color=color, dash=dash, width=width)))

    start, end = f["date"].min().strftime("%Y-%m-%d"), f["date"].max().strftime("%Y-%m-%d")
    fig.add_vrect(x0=start, x1=end, fillcolor="rgba(167,139,250,0.07)", line_width=0, layer="below")
    fig.add_annotation(x=start, y=0.98, yref="paper", xanchor="left", yanchor="top", xshift=6,
                       text="Forecast window", showarrow=False, font=dict(color=LAV, size=12))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=TEXT, size=13), height=480, margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1b1230", bordercolor="#a78bfa", font=dict(color=TEXT, family=FONT)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
        xaxis=dict(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED)),
        yaxis=dict(title=dict(text="Units per day", font=dict(color=MUTED)), gridcolor=GRID,
                   zeroline=False, rangemode="tozero", tickfont=dict(color=MUTED)),
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=MUTED, activecolor=PINK),
    )
    return fig


EMPTY_FIG = go.Figure(layout=dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)", height=480))


# ------------------------------------------------------------------ callback logic (plain functions, easy to test)
def build_items(store, current):
    """Items available in a store; keep the currently selected item when possible."""
    items = sorted(meta.loc[meta["store_id"] == store, "item_id"])
    return items, (current if current in items else items[0])


def build_view(store, item):
    sid = SID.get((item, store))
    if not item or sid is None:
        raise PreventUpdate
    h = hist[hist["id"] == sid].sort_values("date")
    f = fc[fc["id"] == sid].sort_values("date")
    fig = make_figure(h, f)

    o_store = orders[orders["store_id"] == store].sort_values("order_qty", ascending=False)
    row = o_store[o_store["item_id"] == item]
    order_txt = f"{int(row['order_qty'].iloc[0]):,}" if len(row) else "–"
    actual_28 = h[h["date"].isin(f["date"])]["sales"].sum()
    chips = [chip("Actual sales · 28-day test window", f"{actual_28:,.0f}"),
             chip("LightGBM forecast", f"{f['lgbm'].sum():,.1f}"),
             chip("TFT forecast", f"{f['tft'].sum():,.1f}"),
             chip("Recommended order · 14-day plan", order_txt, accent=True)]

    title = [f"{item} · {store}", html.Span("  last 120 days + 28-day forecast", className="chart-sub")]
    cond = ORDERS_BASE_COND + [{"if": {"filter_query": f'{{item_id}} = "{item}"'},
                                "backgroundColor": "rgba(244,114,182,0.20)", "color": PINK_SOFT,
                                "fontWeight": "700"}]
    return fig, title, chips, o_store[[c[0] for c in ORDER_COLS]].to_dict("records"), cond


# ------------------------------------------------------------------ app + page CSS
app = Dash(__name__, title="Demand Forecasting · Inventory Optimization")
server = app.server  # used by gunicorn in Docker

app.index_string = """<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
      :root{
        --text:#f5f3ff; --muted:#b8accf; --lav:#c4b5fd; --purple:#8b5cf6; --pink:#f472b6; --pink-soft:#f9a8d4;
        --border:rgba(196,181,253,.18); --panel:#1b1230;
        /* theme variables used by newer Dash dropdowns (harmless if unused) */
        --Dash-Fill-Inverse-Strong:#1b1230; --Dash-Text-Strong:#f5f3ff; --Dash-Text-Primary:#f5f3ff;
        --Dash-Text-Weak:#b8accf; --Dash-Stroke-Strong:rgba(196,181,253,.35); --Dash-Stroke-Weak:rgba(196,181,253,.18);
        --Dash-Fill-Interactive-Strong:#8b5cf6; --Dash-Fill-Primary-Hover:rgba(139,92,246,.28);
        --Dash-Fill-Primary-Active:rgba(244,114,182,.25);
      }
      html{background:#0d0817}
      body{margin:0;min-height:100vh;color:var(--text);font-family:"Inter","Segoe UI",system-ui,-apple-system,sans-serif;
        background:
          radial-gradient(1100px 560px at 8% -10%, rgba(139,92,246,.30), transparent 60%),
          radial-gradient(900px 520px at 100% 0%, rgba(244,114,182,.20), transparent 55%),
          linear-gradient(180deg,#0d0817 0%,#130b24 100%);
        background-attachment:fixed}
      ::-webkit-scrollbar{width:10px;height:10px}
      ::-webkit-scrollbar-thumb{background:rgba(167,139,250,.40);border-radius:8px}
      ::-webkit-scrollbar-track{background:transparent}

      .page{max-width:1240px;margin:0 auto;padding:40px 24px 56px}
      .eyebrow{color:var(--pink-soft);font-size:12px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;margin-bottom:10px}
      .hero h1{font-size:36px;line-height:1.15;font-weight:800;margin:0 0 10px;
        background:linear-gradient(90deg,#f3e8ff,#c4b5fd 45%,#f472b6);-webkit-background-clip:text;background-clip:text;color:transparent}
      .hero p{color:var(--muted);margin:0 0 16px;font-size:15px}
      .badge{display:inline-block;padding:5px 12px;border-radius:999px;font-size:12px;font-weight:600;color:var(--lav);
        background:rgba(139,92,246,.14);border:1px solid rgba(196,181,253,.25);margin:0 8px 8px 0}

      .kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px;margin:26px 0 6px}
      .kpi{position:relative;overflow:hidden;background:rgba(255,255,255,.045);border:1px solid var(--border);
        border-radius:18px;padding:22px 20px 18px;box-shadow:0 10px 40px rgba(0,0,0,.35)}
      .kpi::before{content:"";position:absolute;left:0;right:0;top:0;height:3px;background:linear-gradient(90deg,var(--purple),var(--pink))}
      .kpi-label{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
      .kpi-value{font-size:34px;font-weight:800;margin:8px 0 4px;
        background:linear-gradient(90deg,#e9d5ff,#f9a8d4);-webkit-background-clip:text;background-clip:text;color:transparent}
      .kpi-sub{font-size:13px;color:var(--muted)}

      .card{background:rgba(255,255,255,.045);border:1px solid var(--border);border-radius:18px;padding:24px;
        margin-top:20px;box-shadow:0 10px 40px rgba(0,0,0,.35)}
      .section-title{font-size:19px;font-weight:700;margin:0 0 4px;color:var(--text)}
      .section-sub{color:var(--muted);font-size:13.5px;margin:0 0 16px;line-height:1.5}
      .note{font-size:12.5px;color:var(--muted);margin:12px 2px 0;line-height:1.55}

      .controls{display:flex;flex-wrap:wrap;gap:18px;position:relative;z-index:20}
      .control{min-width:220px;flex:0 1 260px}
      .control label{display:block;font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);margin-bottom:6px}
      .chips{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:18px 0 8px}
      .chip{background:rgba(139,92,246,.10);border:1px solid var(--border);border-radius:14px;padding:12px 14px}
      .chip-accent{background:rgba(244,114,182,.12);border-color:rgba(244,114,182,.45)}
      .chip-label{font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted)}
      .chip-value{font-size:22px;font-weight:700;margin-top:4px;color:var(--text)}
      .chip-accent .chip-value{color:var(--pink-soft)}
      .chart-title{font-size:16px;font-weight:700;margin:14px 0 0;color:var(--text)}
      .chart-sub{font-weight:500;color:var(--muted);font-size:13px}

      .glossary{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px}
      .gloss{background:rgba(139,92,246,.08);border:1px solid var(--border);border-radius:14px;padding:14px 16px}
      .gloss b{display:block;color:var(--pink-soft);margin-bottom:4px;font-size:14px}
      .gloss span{color:var(--muted);font-size:13px;line-height:1.5}
      .footer{text-align:center;color:var(--muted);font-size:12.5px;margin-top:34px}

      /* dropdowns: older Dash (react-select) */
      .Select-control{background:#1b1230 !important;border:1px solid rgba(196,181,253,.35) !important;border-radius:12px !important}
      .Select-value-label,.Select-input>input{color:#f5f3ff !important}
      .Select-placeholder{color:#b8accf !important}
      .Select-arrow{border-top-color:#c4b5fd !important}
      .Select-menu-outer{background:#1b1230 !important;border:1px solid rgba(196,181,253,.35) !important;border-radius:12px !important;overflow:hidden}
      .Select-option{background:#1b1230 !important;color:#e9e3ff !important}
      .Select-option.is-focused{background:rgba(139,92,246,.35) !important}
      .Select-option.is-selected{background:rgba(244,114,182,.30) !important}
      /* dropdowns: newer Dash */
      .dash-dropdown,.dash-dropdown-trigger{background:#1b1230 !important;color:#f5f3ff !important;
        border:1px solid rgba(196,181,253,.35) !important;border-radius:12px !important}
      .dash-dropdown-value,.dash-dropdown-value-item{color:#f5f3ff !important}
      .dash-dropdown svg{color:#c4b5fd}
      .dash-dropdown-content{background:#1b1230 !important;color:#f5f3ff !important;
        border:1px solid rgba(196,181,253,.35) !important;border-radius:12px !important;z-index:9999}
      .dash-dropdown-search{background:#140c25 !important;color:#f5f3ff !important;border:1px solid rgba(196,181,253,.25) !important}
      .dash-dropdown-option{background:transparent !important;color:#e9e3ff !important}
      .dash-dropdown-option:hover,.dash-dropdown-option[data-highlighted]{background:rgba(139,92,246,.35) !important}
      .dash-dropdown-option[aria-selected="true"],.dash-dropdown-option[data-state="checked"]{background:rgba(244,114,182,.28) !important}

      .column-header--sort{color:#c4b5fd !important}

      @media (max-width:640px){
        .page{padding:22px 14px 40px}
        .hero h1{font-size:26px}
        .kpi-value{font-size:28px}
      }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>"""

GLOSSARY = [
    ("WAPE", "Total absolute forecast error divided by total actual sales. Lower is better."),
    ("RMSSE", "Error scaled by a 'yesterday = today' forecast. Below 1 means better than that naive rule."),
    ("80% interval", "The range (P10 to P90) expected to contain real sales about 8 times out of 10."),
    ("Coherence gap", "Largest mismatch between a store forecast and the sum of its items. 0 means all levels add up."),
    ("Order plan", "OR-Tools picks order quantities for the next 14 days that minimise expected holding plus "
                   "stockout cost under a per-store budget."),
]

app.layout = html.Div(className="page", children=[
    html.Div(className="hero", children=[
        html.Div("Project DS-02", className="eyebrow"),
        html.H1("Nationwide Demand Forecasting & Inventory Optimization"),
        html.P(f"M5 / Walmart · FOODS_3 subset: {n_items} items × {n_stores} stores · "
               "28-day hold-out forecast and 14-day order plan"),
        html.Div([html.Span(t, className="badge") for t in
                  ("Prophet", "LightGBM", "Temporal Fusion Transformer", "Darts", "OR-Tools", "Plotly Dash")]),
    ]),

    html.Div(className="kpi-grid", children=[
        kpi("Most accurate model (item level)", str(best["model"]),
            f"WAPE {best['WAPE_item_store']:.4f} · RMSSE {best['RMSSE']:.4f}"),
        kpi("Interval coverage", f"{coverage:.1%}", "of real sales fall inside LightGBM's 80% interval"),
        kpi("Fill rate · optimized", fill_val, fill_sub),
        kpi("Total cost vs equal-spend naive", cost_delta, "holding + stockout cost, lower is better"),
    ]),

    html.Div(className="card", children=[
        html.H2("Forecast explorer", className="section-title"),
        html.P("Choose a store and an item. The chart compares every model on the last 28 days, "
               "and the chips show totals and the recommended order.", className="section-sub"),
        html.Div(className="controls", children=[
            html.Div(className="control", children=[
                html.Label("Store", htmlFor="store"),
                dcc.Dropdown(id="store", options=stores, value=stores[0], clearable=False)]),
            html.Div(className="control", children=[
                html.Label("Item", htmlFor="item"),
                dcc.Dropdown(id="item", options=[], value=None, clearable=False)]),
        ]),
        html.Div(id="chips", className="chips"),
        html.H3(id="chart-title", className="chart-title"),
        dcc.Graph(id="chart", figure=EMPTY_FIG, style={"height": "480px"},
                  config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}),
    ]),

    card("Forecast accuracy", "Last 28 days, lower is better. The best value in each column is highlighted.",
         [metrics_table]),

    card("Hierarchical reconciliation (Darts)",
         "WAPE at each level of the hierarchy. A non-zero coherence gap means stores do not add up to their state.",
         [recon_table]),

    card("Inventory policy comparison", "Cost measured against what really sold during the 14-day plan.",
         [inventory_table,
          html.P("The plain 'forecast mean' policy has no budget limit. Compare Optimized with the "
                 "'same spend' row for a fair like-for-like result.", className="note")]),

    card("Order recommendations", "Selected store, 14-day plan. The highlighted row is the selected item. "
         "Click a column header to sort.", [orders_table]),

    card("How to read this dashboard", "Short definitions.",
         [html.Div(className="glossary", children=[
             html.Div([html.B(t), html.Span(d)], className="gloss") for t, d in GLOSSARY])]),

    html.Div("Built with Prophet · LightGBM · Darts · PyTorch Forecasting · OR-Tools · Plotly Dash",
             className="footer"),
])


@app.callback(Output("item", "options"), Output("item", "value"),
              Input("store", "value"), State("item", "value"))
def fill_items(store, current):
    return build_items(store, current)


@app.callback(Output("chart", "figure"), Output("chart-title", "children"), Output("chips", "children"),
              Output("orders", "data"), Output("orders", "style_data_conditional"),
              Input("store", "value"), Input("item", "value"))
def update(store, item):
    return build_view(store, item)


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)