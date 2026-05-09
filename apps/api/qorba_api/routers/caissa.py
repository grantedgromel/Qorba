"""Caissa proxy endpoints.

The browser holds an access token (the JWT extracted from the Caissa
Swagger UI's implicit-flow login) in sessionStorage and sends it as the
`X-Caissa-Token` header on each request. We never persist the token
server-side — it expires in ~6 minutes and tabs close.

Three endpoints:
  GET  /integrations/caissa/probe?path=...  raw GET-passthrough so the user
                                            can verify connection / discover
                                            endpoint paths from the Swagger UI.
  GET  /integrations/caissa/list-benchmarks  proxy + a tolerant parse of
                                             whatever benchmark-list shape
                                             Caissa returns.
  POST /integrations/caissa/import-benchmark  pull a benchmark's monthly
                                              returns and persist as a row
                                              in our `benchmarks` table.

Endpoint paths inside Caissa are not yet confirmed — they're parameterised
so the user can try alternates from the Swagger UI without a code change.
"""

from __future__ import annotations

import base64
import json
import time
from datetime import date

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from qorba_api.auth import get_current_user
from qorba_api.db.models import Benchmark, User
from qorba_api.db.session import get_db
from qorba_api.schemas.benchmarks import BenchmarkOut
from qorba_api.settings import Settings, get_settings

router = APIRouter(prefix="/integrations/caissa", tags=["caissa"])


# ── Helpers ──────────────────────────────────────────────────────────


def _require_token(x_caissa_token: str | None) -> str:
    if not x_caissa_token:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Caissa is not connected. Paste a Caissa access token at /settings/caissa.",
        )
    return x_caissa_token


def _decode_unverified_jwt_payload(token: str) -> dict:
    """Decode the JWT body without verifying the signature.

    We trust it because the user pasted it themselves and the call to
    Caissa with this exact token will be the actual auth check.
    """
    try:
        _, body, _ = token.split(".")
    except ValueError:
        return {}
    pad = "=" * (-len(body) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(body + pad))
    except (ValueError, json.JSONDecodeError):
        return {}


class TokenStatus(BaseModel):
    connected: bool
    expires_in_seconds: int | None
    tenant_id: str | None
    user_email: str | None
    scopes: list[str]


@router.get("/status", response_model=TokenStatus)
def status_(
    x_caissa_token: str | None = Header(default=None, alias="X-Caissa-Token"),
    _user: User = Depends(get_current_user),
) -> TokenStatus:
    """Decode the pasted token (no network call) and report what's in it."""
    if not x_caissa_token:
        return TokenStatus(
            connected=False,
            expires_in_seconds=None,
            tenant_id=None,
            user_email=None,
            scopes=[],
        )
    payload = _decode_unverified_jwt_payload(x_caissa_token)
    exp = int(payload.get("exp", 0))
    remaining = max(0, exp - int(time.time())) if exp else None
    scopes = payload.get("scope")
    if isinstance(scopes, str):
        scopes_list = scopes.split()
    elif isinstance(scopes, list):
        scopes_list = [str(s) for s in scopes]
    else:
        scopes_list = []
    return TokenStatus(
        connected=remaining is not None and remaining > 0,
        expires_in_seconds=remaining,
        tenant_id=str(payload.get("tenant_id")) if payload.get("tenant_id") else None,
        user_email=payload.get("email") or payload.get("preferred_username"),
        scopes=scopes_list,
    )


def _caissa_get(path: str, token: str, settings: Settings, params: dict | None = None) -> httpx.Response:
    base = settings.caissa_base_url.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    url = f"{base}{path}"
    with httpx.Client(timeout=20.0) as cli:
        return cli.get(
            url,
            params=params,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
        )


@router.get("/probe")
def probe(
    path: str = "/api/v1",
    x_caissa_token: str | None = Header(default=None, alias="X-Caissa-Token"),
    settings: Settings = Depends(get_settings),
    _user: User = Depends(get_current_user),
):
    """Raw GET-proxy. Useful for verifying the token works and discovering
    endpoint paths via Caissa's own Swagger UI.

    Returns Caissa's response body unchanged (for inspection in the UI).
    """
    token = _require_token(x_caissa_token)
    try:
        r = _caissa_get(path, token, settings)
    except httpx.HTTPError as err:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, f"Could not reach Caissa: {err}"
        ) from err
    body: object
    try:
        body = r.json()
    except ValueError:
        body = r.text
    return {
        "status": r.status_code,
        "url": str(r.request.url),
        "body": body,
    }


# ── Benchmark list / import ──────────────────────────────────────────


# Caissa endpoint paths are PROVISIONAL. The user can override at request
# time via ?path= so we don't need a code change to try alternates.
DEFAULT_LIST_PATH = "/v1/benchmarks"
DEFAULT_RETURNS_PATH_TEMPLATE = "/v1/benchmarks/{code}/returns"


def _coerce_benchmark_list(payload: object) -> list[dict]:
    """Tolerantly extract a list of {code, name, ...} dicts from whatever
    shape Caissa returns. Handles top-level list, {items: [...]}, etc."""
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    if isinstance(payload, dict):
        for key in ("items", "data", "results", "benchmarks"):
            inner = payload.get(key)
            if isinstance(inner, list):
                return [p for p in inner if isinstance(p, dict)]
    return []


@router.get("/list-benchmarks")
def list_benchmarks(
    path: str = DEFAULT_LIST_PATH,
    x_caissa_token: str | None = Header(default=None, alias="X-Caissa-Token"),
    settings: Settings = Depends(get_settings),
    _user: User = Depends(get_current_user),
):
    token = _require_token(x_caissa_token)
    r = _caissa_get(path, token, settings)
    if r.status_code == 401:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Caissa token expired or invalid.")
    if r.status_code != 200:
        return {
            "status": r.status_code,
            "ok": False,
            "raw": r.text[:2000],
            "items": [],
        }
    try:
        body = r.json()
    except ValueError:
        return {"status": r.status_code, "ok": False, "raw": r.text[:2000], "items": []}
    items = _coerce_benchmark_list(body)
    return {
        "status": r.status_code,
        "ok": True,
        "items": items,
        "raw_shape": "list" if isinstance(body, list) else "object",
    }


class CaissaImportRequest(BaseModel):
    code: str
    name: str | None = None
    returns_path: str | None = None  # override DEFAULT_RETURNS_PATH_TEMPLATE


def _parse_returns_payload(payload: object) -> list[tuple[date, float]]:
    """Pull (period, value) tuples out of the Caissa response. We try a few
    common shapes; if none match we return [] and the caller can surface
    a helpful error."""

    candidates: list[object] = []
    if isinstance(payload, list):
        candidates = payload
    elif isinstance(payload, dict):
        for key in ("items", "data", "results", "returns", "monthlyReturns"):
            inner = payload.get(key)
            if isinstance(inner, list):
                candidates = inner
                break

    out: list[tuple[date, float]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        period_raw = (
            item.get("period")
            or item.get("date")
            or item.get("asOf")
            or item.get("month")
        )
        value_raw = (
            item.get("value")
            if "value" in item
            else item.get("return") if "return" in item
            else item.get("monthlyReturn") if "monthlyReturn" in item
            else None
        )
        if period_raw is None or value_raw is None:
            continue
        try:
            period = date.fromisoformat(str(period_raw)[:10])
            value = float(value_raw)
        except (ValueError, TypeError):
            continue
        out.append((period, value))
    out.sort()
    return out


@router.post("/import-benchmark", response_model=BenchmarkOut, status_code=status.HTTP_201_CREATED)
def import_benchmark(
    body: CaissaImportRequest,
    x_caissa_token: str | None = Header(default=None, alias="X-Caissa-Token"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    token = _require_token(x_caissa_token)
    template = body.returns_path or DEFAULT_RETURNS_PATH_TEMPLATE
    path = template.format(code=body.code)
    r = _caissa_get(path, token, settings)
    if r.status_code == 401:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Caissa token expired or invalid.")
    if r.status_code != 200:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"Caissa returned {r.status_code} for {path}: {r.text[:300]}",
        )
    try:
        payload = r.json()
    except ValueError as err:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, "Caissa returned non-JSON for the returns endpoint."
        ) from err
    points = _parse_returns_payload(payload)
    if not points:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Could not parse a (date, value) series from Caissa's response. "
            "Try /probe to inspect the raw shape and override returns_path.",
        )

    name = body.name or body.code
    series = {
        "id": str(__import__("uuid").uuid4()),
        "name": name,
        "points": [{"period": d.isoformat(), "value": v} for d, v in points],
        "inception": points[0][0].isoformat(),
        "last_observation": points[-1][0].isoformat(),
        "n_observations": len(points),
        "source": "library",
        "checksum": _checksum_points(points),
    }

    bench = Benchmark(
        user_id=user.id,
        name=name,
        code=body.code,
        provider="caissa",
        is_user_uploaded=False,
        series=series,
    )
    db.add(bench)
    db.commit()
    db.refresh(bench)
    return BenchmarkOut(
        id=bench.id,
        name=bench.name,
        code=bench.code,
        provider=bench.provider,
        inception=points[0][0],
        last_observation=points[-1][0],
        n_observations=len(points),
    )


def _checksum_points(points: list[tuple[date, float]]) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(",".join(f"{d.isoformat()}:{round(v, 12)}" for d, v in points).encode())
    return h.hexdigest()
