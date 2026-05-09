"""Selects the active BenchmarkProvider based on settings."""

from __future__ import annotations

from qorba_api.core.benchmarks import BenchmarkProvider, StaticProvider
from qorba_api.core.benchmarks.caissa import CaissaConfig, CaissaProvider
from qorba_api.settings import Settings, get_settings


def _caissa_config(settings: Settings) -> CaissaConfig:
    return CaissaConfig(
        client_id=settings.caissa_client_id,
        client_secret=settings.caissa_client_secret,
        base_url=settings.caissa_base_url,
        auth_url=settings.caissa_auth_url,
        token_url=settings.caissa_token_url,
        redirect_uri=settings.caissa_redirect_uri,
    )


def get_benchmark_provider() -> BenchmarkProvider:
    settings = get_settings()
    cfg = _caissa_config(settings)
    if cfg.is_configured():
        # Until users.caissa_refresh_token is wired, Caissa can't actually
        # serve calls. Keep falling back; flip this when the OAuth callback
        # endpoint lands.
        return StaticProvider()
    return StaticProvider()


__all__ = ["CaissaProvider", "_caissa_config", "get_benchmark_provider"]
