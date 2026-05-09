"""End-to-end Sprint-1 DoD: register -> upload CSV -> create fund -> create
analysis -> compute -> see Sharpe."""

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

    # 2. Ingest CSV
    r = client.post(
        "/api/v1/ingest/csv",
        files={"file": ("fund.csv", io.BytesIO(_sample_csv()), "text/csv")},
    )
    assert r.status_code == 200, r.text
    extracted = r.json()
    assert extracted["tier_used"] == 1
    assert extracted["series"]["n_observations"] == 48
    assert extracted["detected_scale"] in ("percent", "decimal")

    # 3. Create fund
    r = client.post("/api/v1/funds", json=extracted)
    assert r.status_code == 201, r.text
    fund_id = r.json()["id"]

    # 4. Create analysis
    r = client.post(
        "/api/v1/analyses",
        json={"fund_id": fund_id, "metrics": {"metric_ids": ["sharpe", "ann_vol", "max_dd"]}},
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
    assert sharpe["formatted"].endswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"))
    assert result["metrics"]["max_dd"]["value"] is not None
    assert result["metrics"]["ann_vol"]["value"] is not None
