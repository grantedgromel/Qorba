"""Benchmark data providers.

The production provider is `CaissaProvider`, which talks to
`client-api.caissallc.com` using OAuth2 (authorization code flow). The
`StaticProvider` is a deterministic fallback used by tests and offline dev.

The protocol is intentionally narrow: list and fetch a monthly return
series. Anything richer (universes, currency overrides, gross/net flags)
gets layered on as a separate concern when we need it.
"""

from qorba_api.core.benchmarks.provider import (
    BenchmarkInfo,
    BenchmarkProvider,
    BenchmarkSeries,
)
from qorba_api.core.benchmarks.static import StaticProvider

__all__ = ["BenchmarkInfo", "BenchmarkProvider", "BenchmarkSeries", "StaticProvider"]
