from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "customer_scores.csv"
REF_DISCOUNT = 20.0   # % discount that the uplift scores correspond to

st.set_page_config(page_title="Retention ROI Simulator", layout="wide")
st.title("Retention offer ROI simulator")
st.caption("Uplift scores come from a SYNTHETIC treatment experiment on IBM Telco data "
           "(see README). Results demonstrate the method; they are not a forecast. Amounts in $.")

@st.cache_data
def load():
    return pd.read_csv(DATA)

df = load()
n = len(df)

with st.sidebar:
    st.header("Offer")
    discount = st.slider("Discount on monthly bill (%)", 5, 50, 20, step=5)
    months = st.slider("Discount duration (months)", 1, 12, 6)
    st.header("Targeting")
    strategy = st.selectbox("Who gets the offer?", ["Highest uplift", "Highest churn risk", "Random"])
    target_pct = st.slider("Share of customers targeted (%)", 5, 100, 30, step=5)

scale = min(discount / REF_DISCOUNT, 2.0)
saved_prob = np.minimum(df["uplift"].to_numpy() * scale, df["churn_prob"].to_numpy())
value = saved_prob * df["clv"].to_numpy()                         # expected $ retained per customer
cost = (discount / 100) * df["monthlycharges"].to_numpy() * months   # $ discount per targeted customer

orders = {"Highest uplift": np.argsort(-df["uplift"].to_numpy()),
          "Highest churn risk": np.argsort(-df["churn_prob"].to_numpy())}

def point(strat, frac):
    k = max(int(round(n * frac)), 1)
    if strat == "Random":
        return value.sum() * k / n, cost.sum() * k / n
    o = orders[strat][:k]
    return value[o].sum(), cost[o].sum()

v, c = point(strategy, target_pct / 100)
net = v - c
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers targeted", f"{int(round(n * target_pct / 100)):,}")
c2.metric("Expected savings", f"${v:,.0f}")
c3.metric("Offer cost", f"${c:,.0f}")
c4.metric("Net benefit", f"${net:,.0f}", f"ROI {net / c:.0%}" if c > 0 else None)

fracs = np.arange(0.05, 1.0001, 0.05)
fig = go.Figure()
for s in ["Highest uplift", "Highest churn risk", "Random"]:
    ys = [point(s, f)[0] - point(s, f)[1] for f in fracs]
    fig.add_trace(go.Scatter(x=fracs * 100, y=ys, mode="lines", name=s))
fig.add_trace(go.Scatter(x=[target_pct], y=[net], mode="markers", name="Your choice",
                         marker=dict(size=12, symbol="diamond")))
fig.update_layout(title="Net benefit vs share of customers targeted",
                  xaxis_title="% of customers targeted", yaxis_title="Net benefit ($)")
st.plotly_chart(fig)

if strategy != "Random":
    k = max(int(round(n * target_pct / 100)), 1)
    sel = df.iloc[orders[strategy][:k]].copy()
    sel["savings"], sel["cost"] = value[orders[strategy][:k]], cost[orders[strategy][:k]]
    st.subheader("Who is being targeted (by segment)")
    st.dataframe(sel.groupby("segment").agg(customers=("customer_id", "count"),
                 avg_uplift=("uplift", "mean"), savings=("savings", "sum"),
                 cost=("cost", "sum")).round(3))