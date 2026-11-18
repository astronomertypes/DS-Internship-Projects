"""Smoke tests for the dashboard: loads app/app.py and calls its logic directly (no browser needed)."""
import importlib.util
import sys
from pathlib import Path

import pytest
from dash.exceptions import PreventUpdate

ROOT = Path(__file__).resolve().parent.parent
APP_PATH = ROOT / "app" / "app.py"


@pytest.fixture(scope="module")
def mod():
    if not (ROOT / "app" / "data" / "forecasts.parquet").exists():
        pytest.skip("run python src/export_app_data.py first")
    spec = importlib.util.spec_from_file_location("dashboard_app", APP_PATH)
    m = importlib.util.module_from_spec(spec)
    sys.modules["dashboard_app"] = m
    spec.loader.exec_module(m)
    return m


def test_layout_contains_required_ids(mod):
    ids = {c.id for c in mod.app.layout._traverse() if getattr(c, "id", None)}
    assert {"store", "item", "chart", "chart-title", "chips", "orders"} <= ids


def test_items_for_store_and_selection_is_kept(mod):
    options, value = mod.build_items("CA_1", None)
    assert len(options) == mod.n_items and value == options[0]
    keep = options[5]
    assert mod.build_items("TX_1", keep)[1] == keep


def test_build_view_returns_all_five_outputs(mod):
    options, _ = mod.build_items("CA_1", None)
    fig, title, chips, data, cond = mod.build_view("CA_1", options[0])
    names = [t.name for t in fig.data]
    assert "Actual" in names and "Prophet" in names and "TFT" in names
    assert len(chips) == 4
    assert len(data) == mod.n_items
    assert data[0]["order_qty"] >= data[-1]["order_qty"]          # sorted by order size
    assert any(options[0] in str(c["if"].get("filter_query", "")) for c in cond)  # selected row highlighted


def test_unknown_selection_does_not_update(mod):
    with pytest.raises(PreventUpdate):
        mod.build_view("ZZ_9", "nope")