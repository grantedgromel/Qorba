"""End-to-end Sprint-2 DoD: register → upload CSV → confirm draft → see Sharpe."""

from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd


def _sample_csv() -> bytes:
    rng = np.random.RandomState(42)
    dates = pd.date_range("2019-01-31", periods=48, freq="ME")
    returns = rng.normal(0.008, 0.04, 48)
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "return": returns})
    return df.to_csv(index=False).encode()


def test_csv_to_sharpe(client) -> None:
    # 1. Register
    r = client.post(
        "/api/v1/auth/register",
        json={"email": "lp@example.com", "password": "correct horse battery staple"},
    )
    assert r.status_code == 201, r.text

    # 2. Ingest CSV → draft
    r = client.post(
        "/api/v1/ingest/csv",
        files={"file": ("fund.csv", io.BytesIO(_sample_csv()), "text/csv")},
    )
    assert r.status_code == 200, r.text
    draft = r.json()
    assert draft["tier_used"] == 1
    assert len(draft["points"]) == 48
    assert draft["detected_scale"] in ("percent", "decimal")
    draft_id = draft["id"]

    # 3. Confirm the draft (no edits, accept the detected scale) → fund
    r = client.post(
        f"/api/v1/ingest/drafts/{draft_id}/confirm",
        json={"scale": draft["detected_scale"], "edits": []},
    )
    assert r.status_code == 201, r.text
    fund_id = r.json()["id"]

    # 4. Create analysis
    r = client.post(
        "/api/v1/analyses",
        json={
            "fund_id": fund_id,
            "metrics": {"metric_ids": ["sharpe", "ann_vol", "max_dd"]},
            "rf_annual": 0.0,
            "mar_annual": 0.0,
            "omega_threshold": 0.0,
        },
    )
    assert r.status_code == 201, r.text
    analysis_id = r.json()["id"]

    # 5. Compute
    r = client.post(f"/api/v1/analyses/{analysis_id}/compute")
    assert r.status_code == 200, r.text
    result = r.json()

    sharpe = result["metrics"]["sharpe"]
    assert sharpe["value"] is not None
    assert math.isfinite(float(sharpe["value"]))
    assert result["metrics"]["max_dd"]["value"] is not None
    assert result["metrics"]["ann_vol"]["value"] is not None


def test_paste_ingest(client) -> None:
    client.post(
        "/api/v1/auth/register",
        json={"email": "paste@example.com", "password": "correct horse battery staple"},
    )
    text = "date,return\n2024-01-31,0.012\n2024-02-29,-0.005\n2024-03-31,0.021\n"
    r = client.post("/api/v1/ingest/paste", json={"name": "Test Fund", "text": text})
    assert r.status_code == 200, r.text
    draft = r.json()
    assert draft["tier_used"] == 4
    assert len(draft["points"]) == 3


def test_benchmark_library(client) -> None:
    client.post(
        "/api/v1/auth/register",
        json={"email": "lib@example.com", "password": "correct horse battery staple"},
    )
    r = client.get("/api/v1/benchmarks/library")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provider_active"] == "static"
    codes = {item["code"] for item in body["items"]}
    assert {"SPX_TR", "RUT_TR", "ACWI", "HFRI_FW"} <= codes


def test_percent_scale_division(client) -> None:
    """A CSV with values like 1.23 (=1.23%) should land as 0.0123 in the fund."""
    client.post(
        "/api/v1/auth/register",
        json={"email": "pct@example.com", "password": "correct horse battery staple"},
    )
    rng = np.random.RandomState(7)
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    pct_values = rng.normal(0.8, 4.0, 24)  # in percent
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "return": pct_values})
    csv = df.to_csv(index=False).encode()

    r = client.post(
        "/api/v1/ingest/csv",
        files={"file": ("pct.csv", io.BytesIO(csv), "text/csv")},
    )
    assert r.status_code == 200, r.text
    draft = r.json()
    assert draft["detected_scale"] == "percent"

    r = client.post(
        f"/api/v1/ingest/drafts/{draft['id']}/confirm",
        json={"scale": "percent", "edits": []},
    )
    assert r.status_code == 201, r.text
    fund_id = r.json()["id"]

    r = client.post(
        "/api/v1/analyses",
        json={
            "fund_id": fund_id,
            "metrics": {"metric_ids": ["ann_vol"]},
            "rf_annual": 0.0,
            "mar_annual": 0.0,
            "omega_threshold": 0.0,
        },
    )
    analysis_id = r.json()["id"]
    r = client.post(f"/api/v1/analyses/{analysis_id}/compute")
    ann_vol = r.json()["metrics"]["ann_vol"]["value"]
    # The unscaled values had std~4.0 (percent), so decimal vol ~0.04 * sqrt(12) ~ 0.139
    # Far below 1.0, which is the smoking gun that scaling happened.
    assert 0.05 < ann_vol < 0.25, ann_vol
