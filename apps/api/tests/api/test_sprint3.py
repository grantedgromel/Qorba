"""Sprint 3: period windowing + metrics catalog."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd


def _sample_csv(months: int = 60) -> bytes:
    rng = np.random.RandomState(7)
    dates = pd.date_range("2019-01-31", periods=months, freq="ME")
    values = rng.normal(0.008, 0.04, months)
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "return": values})
    return df.to_csv(index=False).encode()


def _build_analysis(client) -> str:
    """Register, ingest a 60-month CSV, confirm, build an analysis. Returns analysis_id."""
    client.post(
        "/api/v1/auth/register",
        json={"email": "p@example.com", "password": "correct horse battery staple"},
    )
    r = client.post(
        "/api/v1/ingest/csv",
        files={"file": ("fund.csv", io.BytesIO(_sample_csv(60)), "text/csv")},
    )
    draft = r.json()
    r = client.post(
        f"/api/v1/ingest/drafts/{draft['id']}/confirm",
        json={"scale": draft["detected_scale"], "edits": []},
    )
    fund_id = r.json()["id"]
    r = client.post(
        "/api/v1/analyses",
        json={
            "fund_id": fund_id,
            "metrics": {
                "metric_ids": [
                    "sharpe",
                    "ann_vol",
                    "ann_return_geo",
                    "max_dd",
                    "win_rate",
                    "sortino",
                ]
            },
            "rf_annual": 0.0,
            "mar_annual": 0.0,
            "omega_threshold": 0.0,
        },
    )
    return r.json()["id"]


def test_compute_default_period_is_all(client) -> None:
    aid = _build_analysis(client)
    r = client.post(f"/api/v1/analyses/{aid}/compute")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["period"] == "ALL"
    assert len(body["monthly_returns"]) == 60
    assert len(body["cumulative_growth"]) == 60


def test_compute_period_3m_returns_3_months(client) -> None:
    aid = _build_analysis(client)
    r = client.post(f"/api/v1/analyses/{aid}/compute?period=3M")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["period"] == "3M"
    assert len(body["monthly_returns"]) == 3
    assert len(body["cumulative_growth"]) == 3


def test_compute_period_1y_returns_12_months(client) -> None:
    aid = _build_analysis(client)
    r = client.post(f"/api/v1/analyses/{aid}/compute?period=1Y")
    assert r.status_code == 200
    assert len(r.json()["monthly_returns"]) == 12


def test_compute_period_changes_metric_values(client) -> None:
    """A period slice should produce different metric values than ALL."""
    aid = _build_analysis(client)
    full = client.post(f"/api/v1/analyses/{aid}/compute?period=ALL").json()
    one_y = client.post(f"/api/v1/analyses/{aid}/compute?period=1Y").json()
    # Different windows → different version_hash
    assert full["version_hash"] != one_y["version_hash"]


def test_compute_period_invalid_rejected(client) -> None:
    aid = _build_analysis(client)
    r = client.post(f"/api/v1/analyses/{aid}/compute?period=BOGUS")
    assert r.status_code == 422  # FastAPI rejects bad Literal values


def test_compute_metric_ids_override(client) -> None:
    aid = _build_analysis(client)
    r = client.post(
        f"/api/v1/analyses/{aid}/compute?metric_ids=sharpe,calmar,omega"
    )
    assert r.status_code == 200
    keys = set(r.json()["metrics"].keys())
    assert keys == {"sharpe", "calmar", "omega"}


def test_metrics_catalog_lists_groups(client) -> None:
    client.post(
        "/api/v1/auth/register",
        json={"email": "cat@example.com", "password": "correct horse battery staple"},
    )
    r = client.get("/api/v1/metrics/catalog")
    assert r.status_code == 200, r.text
    items = r.json()["items"]
    assert len(items) > 15
    groups = {item["group"] for item in items}
    assert {
        "Returns",
        "Risk",
        "Drawdown",
        "Risk-Adjusted",
        "Distributional",
        "Benchmark-Relative",
    } <= groups
    # At least the canonical defaults.
    defaults = {item["id"] for item in items if item["default"]}
    assert {"sharpe", "sortino", "ann_vol", "max_dd", "win_rate"} <= defaults


def test_metrics_catalog_marks_benchmark_metrics(client) -> None:
    client.post(
        "/api/v1/auth/register",
        json={"email": "b@example.com", "password": "correct horse battery staple"},
    )
    r = client.get("/api/v1/metrics/catalog")
    items = r.json()["items"]
    by_id = {item["id"]: item for item in items}
    assert by_id["beta"]["requires_benchmark"] is True
    assert by_id["alpha_ann"]["requires_benchmark"] is True
    assert by_id["sharpe"]["requires_benchmark"] is False
