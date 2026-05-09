"""Metric catalog — surfaces the full registry to the frontend palette."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from qorba_api.auth import get_current_user
from qorba_api.db.models import User
from qorba_api.schemas.analyses import MetricCatalog, MetricCatalogEntry
from qorba_api.services.compute import metrics_catalog

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/catalog", response_model=MetricCatalog)
def catalog(_user: User = Depends(get_current_user)) -> MetricCatalog:
    return MetricCatalog(
        items=[MetricCatalogEntry(**entry) for entry in metrics_catalog()],
    )
