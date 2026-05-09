"""FastAPI app factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from alembic import command
from alembic.config import Config as AlembicConfig
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from qorba_api.routers import analyses, auth, benchmarks, funds, health, ingest
from qorba_api.settings import get_settings


def _run_migrations() -> None:
    api_root = Path(__file__).resolve().parent.parent
    cfg = AlembicConfig(str(api_root / "alembic.ini"))
    cfg.set_main_option("script_location", str(api_root / "qorba_api/db/migrations"))
    cfg.set_main_option("sqlalchemy.url", get_settings().database_url)
    command.upgrade(cfg, "head")


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    if get_settings().run_migrations_on_startup:
        _run_migrations()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Qorba API",
        version="0.1.0",
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.web_origin],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api_v1 = "/api/v1"
    app.include_router(health.router, prefix=api_v1)
    app.include_router(auth.router, prefix=api_v1)
    app.include_router(ingest.router, prefix=api_v1)
    app.include_router(funds.router, prefix=api_v1)
    app.include_router(benchmarks.router, prefix=api_v1)
    app.include_router(analyses.router, prefix=api_v1)

    return app


app = create_app()
