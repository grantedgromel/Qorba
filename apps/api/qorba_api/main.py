"""FastAPI app factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from qorba_api.db.models import Base
from qorba_api.db.session import engine
from qorba_api.routers import analyses, auth, funds, health, ingest
from qorba_api.settings import get_settings


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Sprint 1 convenience: create tables if missing. Sprint 2 switches to Alembic.
    Base.metadata.create_all(bind=engine)
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
    app.include_router(analyses.router, prefix=api_v1)

    return app


app = create_app()
