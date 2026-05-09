"""Caissa benchmark provider — OAuth2 (authorization_code + refresh_token).

This module is *scaffolded but not active*. It will start working once
Caissa registers an OAuth client_id (+ optional client_secret) for our
backend. Until then:

  - `is_configured()` returns False, so the benchmarks router falls back
    to the StaticProvider.
  - The token-exchange and API-call methods are implemented but never
    reached during normal operation.

OAuth context (decoded from a real Caissa Swagger access token):
  - issuer:        https://platform-login.caissallc.com
  - audience:      api
  - scopes:        read, write
  - resources:     fine-grained permissions (e.g. benchmark_read), inherited
                   from the authenticated user's account.

Auth flow we use:
  1. User clicks "Connect Caissa" -> we redirect to {auth_url}/connect/authorize
     with response_type=code, scope=read, redirect_uri=our callback.
  2. Caissa redirects back with ?code=... -> we POST to /connect/token to
     exchange for access_token + refresh_token.
  3. We persist the refresh_token in `users.caissa_refresh_token` (encrypted).
  4. On each API call, we use the access_token; on 401, refresh and retry once.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import httpx

from qorba_api.core.benchmarks.provider import (
    BenchmarkInfo,
    BenchmarkProvider,
    BenchmarkSeries,
)


@dataclass(frozen=True)
class CaissaConfig:
    client_id: str | None
    client_secret: str | None
    base_url: str
    auth_url: str
    token_url: str
    redirect_uri: str

    def is_configured(self) -> bool:
        return bool(self.client_id)


class CaissaProvider(BenchmarkProvider):
    name = "caissa"

    def __init__(self, config: CaissaConfig, refresh_token: str | None = None) -> None:
        self.config = config
        self.refresh_token = refresh_token
        self._access_token: str | None = None

    # ── Public protocol ─────────────────────────────────────────────────
    def list_benchmarks(self) -> list[BenchmarkInfo]:
        # Endpoint path is provisional pending real spec inspection.
        data = self._get("/v1/benchmarks")
        return [
            BenchmarkInfo(
                code=str(item["code"]),
                name=str(item["name"]),
                category=str(item.get("category", "unknown")),
                provider="caissa",
            )
            for item in data.get("items", data if isinstance(data, list) else [])
        ]

    def fetch_returns(
        self,
        code: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> BenchmarkSeries:
        params: dict[str, str] = {}
        if start:
            params["start"] = start.isoformat()
        if end:
            params["end"] = end.isoformat()
        _ = self._get(f"/v1/benchmarks/{code}/returns", params=params)
        # Shape unknown until first real call. Punt parsing into Sprint 2.5.
        raise NotImplementedError(
            "CaissaProvider response parsing is unwritten — wire up after the first "
            "successful exchange against client-api.caissallc.com.",
        )

    # ── HTTP plumbing (private) ─────────────────────────────────────────
    def _ensure_token(self) -> str:
        if self._access_token:
            return self._access_token
        if not self.refresh_token:
            raise RuntimeError(
                "Caissa is not connected: no refresh token. The user needs to "
                "complete the /auth/caissa/connect flow first.",
            )
        with httpx.Client(timeout=15.0) as cli:
            r = cli.post(
                self.config.token_url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self.refresh_token,
                    "client_id": self.config.client_id,
                    **({"client_secret": self.config.client_secret} if self.config.client_secret else {}),
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            r.raise_for_status()
            payload = r.json()
            self._access_token = payload["access_token"]
            new_refresh = payload.get("refresh_token")
            if new_refresh:
                self.refresh_token = new_refresh
            return self._access_token

    def _get(self, path: str, params: dict[str, str] | None = None) -> dict:
        token = self._ensure_token()
        url = self.config.base_url.rstrip("/") + path
        with httpx.Client(timeout=15.0) as cli:
            r = cli.get(
                url,
                params=params,
                headers={"Authorization": f"Bearer {token}"},
            )
            if r.status_code == 401:
                self._access_token = None  # force refresh
                token = self._ensure_token()
                r = cli.get(
                    url,
                    params=params,
                    headers={"Authorization": f"Bearer {token}"},
                )
            r.raise_for_status()
            return r.json()
