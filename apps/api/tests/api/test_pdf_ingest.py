"""PDF ingest cascade tests using a synthetic calendar tearsheet."""

from __future__ import annotations

import io

import numpy as np

from tests.fixtures.build_pdf import calendar_table_pdf


def _register(client) -> None:
    r = client.post(
        "/api/v1/auth/register",
        json={"email": "pdf@example.com", "password": "correct horse battery staple"},
    )
    assert r.status_code == 201, r.text


def _three_year_pdf() -> bytes:
    rng = np.random.RandomState(11)
    years = [2021, 2022, 2023]
    values = [list(rng.normal(0.6, 3.5, 12)) for _ in years]
    return calendar_table_pdf(years=years, values_pct=values)


def test_pdf_cascade_tier1_succeeds(client) -> None:
    _register(client)
    pdf = _three_year_pdf()

    r = client.post(
        "/api/v1/ingest/pdf",
        files={"file": ("synthetic.pdf", io.BytesIO(pdf), "application/pdf")},
    )
    assert r.status_code == 200, r.text
    draft = r.json()
    assert draft["tier_used"] in (1, 2)
    assert len(draft["points"]) == 36
    # Synthetic values are in percent
    assert draft["detected_scale"] == "percent"
    assert draft["confidence"] >= 0.6


def test_pdf_cascade_empty_pdf_422(client) -> None:
    _register(client)
    import pymupdf

    doc = pymupdf.open()
    doc.new_page()
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()

    r = client.post(
        "/api/v1/ingest/pdf",
        files={"file": ("blank.pdf", io.BytesIO(buf.getvalue()), "application/pdf")},
    )
    assert r.status_code == 422


def test_pdf_full_round_trip_to_sharpe(client) -> None:
    _register(client)
    pdf = _three_year_pdf()

    r = client.post(
        "/api/v1/ingest/pdf",
        files={"file": ("synthetic.pdf", io.BytesIO(pdf), "application/pdf")},
    )
    draft = r.json()

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
            "metrics": {"metric_ids": ["sharpe", "ann_vol", "max_dd"]},
            "rf_annual": 0.0,
            "mar_annual": 0.0,
            "omega_threshold": 0.0,
        },
    )
    analysis_id = r.json()["id"]
    r = client.post(f"/api/v1/analyses/{analysis_id}/compute")
    assert r.status_code == 200, r.text
    metrics = r.json()["metrics"]
    assert metrics["sharpe"]["value"] is not None
