"""Shared test fixtures: in-memory SQLite + a TestClient that uses it."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Force test settings before importing the app.
os.environ["QORBA_ENV"] = "test"
os.environ["QORBA_DATABASE_URL"] = "sqlite+pysqlite:///:memory:"
os.environ["QORBA_SESSION_SECRET"] = "test-secret-32-bytes-aaaaaaaaaaaaa"
os.environ["QORBA_ALLOW_REGISTRATION"] = "true"

from qorba_api.db.models import Base
from qorba_api.db.session import get_db
from qorba_api.main import create_app


@pytest.fixture
def engine_and_session():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
    yield engine, Session
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
def client(engine_and_session) -> Iterator[TestClient]:
    _, Session = engine_and_session
    app = create_app()

    def _get_db_override():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _get_db_override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
