"""Caissa browser-bridge proxy tests.

The proxy is stateless: browser holds the token in sessionStorage and
sends it as X-Caissa-Token. We mock the upstream Caissa API with respx.
"""

from __future__ import annotations

import base64
import json
import time

import httpx
import respx


def _fake_jwt(*, exp_in: int = 360, tenant: str = "287", scopes: list[str] | None = None) -> str:
    """Build a minimal unsigned JWT-shaped string for status decoding."""
    header = base64.urlsafe_b64encode(b'{"alg":"none","typ":"JWT"}').rstrip(b"=").decode()
    payload = {
        "iss": "https://platform-login.caissallc.com",
        "exp": int(time.time()) + exp_in,
        "tenant_id": tenant,
        "email": "lp@example.com",
        "scope": scopes or ["read", "write"],
    }
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    sig = "x" * 16
    return f"{header}.{body}.{sig}"


def _register(client) -> None:
    r = client.post(
        "/api/v1/auth/register",
        json={"email": "caissa@example.com", "password": "correct horse battery staple"},
    )
    assert r.status_code == 201, r.text


def test_status_no_token(client) -> None:
    _register(client)
    r = client.get("/api/v1/integrations/caissa/status")
    assert r.status_code == 200
    assert r.json()["connected"] is False


def test_status_with_token_decodes_metadata(client) -> None:
    _register(client)
    token = _fake_jwt()
    r = client.get(
        "/api/v1/integrations/caissa/status",
        headers={"X-Caissa-Token": token},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["connected"] is True
    assert body["expires_in_seconds"] is not None
    assert 300 < body["expires_in_seconds"] <= 360
    assert body["tenant_id"] == "287"
    assert body["user_email"] == "lp@example.com"
    assert "read" in body["scopes"]


@respx.mock
def test_probe_passthrough(client) -> None:
    _register(client)
    token = _fake_jwt()
    respx.get("https://client-api.caissallc.com/api/v1/foo").mock(
        return_value=httpx.Response(200, json={"hello": "world"})
    )
    r = client.get(
        "/api/v1/integrations/caissa/probe",
        params={"path": "/api/v1/foo"},
        headers={"X-Caissa-Token": token},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == 200
    assert body["body"] == {"hello": "world"}


@respx.mock
def test_list_benchmarks_tolerant_shape(client) -> None:
    _register(client)
    token = _fake_jwt()
    # Mock returns shape {"items": [...]}, parser should pull out the list.
    respx.get("https://client-api.caissallc.com/api/v1/benchmarks").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {"code": "SPX_TR", "name": "S&P 500"},
                    {"code": "ACWI", "name": "MSCI ACWI"},
                ]
            },
        )
    )
    r = client.get(
        "/api/v1/integrations/caissa/list-benchmarks",
        headers={"X-Caissa-Token": token},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    codes = {item["code"] for item in body["items"]}
    assert codes == {"SPX_TR", "ACWI"}


@respx.mock
def test_import_benchmark_persists(client) -> None:
    _register(client)
    token = _fake_jwt()
    respx.get(
        "https://client-api.caissallc.com/api/v1/benchmarks/SPX_TR/returns"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {"period": "2024-01-31", "value": 0.012},
                    {"period": "2024-02-29", "value": -0.005},
                    {"period": "2024-03-31", "value": 0.021},
                ]
            },
        )
    )
    r = client.post(
        "/api/v1/integrations/caissa/import-benchmark",
        json={"code": "SPX_TR", "name": "S&P 500 TR"},
        headers={"X-Caissa-Token": token},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["code"] == "SPX_TR"
    assert body["name"] == "S&P 500 TR"
    assert body["provider"] == "caissa"
    assert body["n_observations"] == 3

    # And it shows up in the library
    r = client.get("/api/v1/benchmarks/library")
    assert r.status_code == 200
    found = any(item["code"] == "SPX_TR" and item["provider"] == "user" for item in r.json()["items"])
    assert found


def test_proxy_requires_token(client) -> None:
    _register(client)
    r = client.get("/api/v1/integrations/caissa/list-benchmarks")
    assert r.status_code == 400


@respx.mock
def test_import_benchmark_handles_unknown_shape(client) -> None:
    _register(client)
    token = _fake_jwt()
    respx.get(
        "https://client-api.caissallc.com/api/v1/benchmarks/MYSTERY/returns"
    ).mock(
        return_value=httpx.Response(200, json={"some": "weird shape", "no": "items"})
    )
    r = client.post(
        "/api/v1/integrations/caissa/import-benchmark",
        json={"code": "MYSTERY"},
        headers={"X-Caissa-Token": token},
    )
    assert r.status_code == 422


@respx.mock
def test_proxy_handles_401(client) -> None:
    _register(client)
    token = _fake_jwt()
    respx.get("https://client-api.caissallc.com/api/v1/benchmarks").mock(
        return_value=httpx.Response(401, json={"detail": "expired"})
    )
    r = client.get(
        "/api/v1/integrations/caissa/list-benchmarks",
        headers={"X-Caissa-Token": token},
    )
    assert r.status_code == 401
