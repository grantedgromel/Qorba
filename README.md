# Qorba

Public-manager analytics for institutional allocators.

> Sprint 1 status: monorepo bootstrap. CSV upload → Sharpe round-trip works
> end-to-end. PDF ingestion, peer groups, exports, charts and the metric
> picker arrive in Sprints 2-6. The full plan lives on the planning branch
> at `docs/qorba-v2-plan.md`.

## Stack

| Layer | Choice |
| --- | --- |
| Frontend | Next.js 15 App Router, React 19, Tailwind v3, shadcn primitives |
| Backend | FastAPI, Pydantic v2, SQLAlchemy 2 |
| DB / cache / queue | Postgres 16, Redis 7 |
| Charts | Visx (Sprint 3+) |
| Contracts | OpenAPI → `openapi-typescript` codegen, drift-checked in CI |
| Hosting | Self-hostable Docker stack via `docker compose up` |

## Repo layout

```
apps/
  api/      FastAPI service. Travers Chapter 6 analytics in core/analytics/.
  web/      Next.js frontend.
packages/
  shared/   Generated TS types (do not edit by hand).
  tsconfig/ Shared tsconfig presets.
infra/      Dockerfiles + Caddyfile for self-host.
scripts/    codegen.sh, golden PDF fixtures.
```

## Local dev

### Docker (recommended)

```bash
cp .env.example .env   # then edit secrets
docker compose up --build
# Web:  http://localhost:3000
# API:  http://localhost:8000/api/v1/health
```

### Without Docker

```bash
# API
cd apps/api
uv venv .venv
uv pip install -e ".[dev]"
.venv/bin/uvicorn qorba_api.main:app --reload

# Web (separate terminal)
pnpm install
pnpm --filter @qorba/web dev
```

You'll need a Postgres on `localhost:5432` (or override `QORBA_DATABASE_URL`).

## Tests

```bash
# API
cd apps/api && .venv/bin/pytest -q

# Web typecheck and build
pnpm --filter @qorba/web typecheck
pnpm --filter @qorba/web build
```

## Codegen

API contracts are the source of truth. After changing a Pydantic schema or
adding an endpoint, regenerate the TS types:

```bash
pnpm codegen           # writes packages/shared/src/openapi.d.ts
pnpm codegen:check     # CI uses this; fails on drift
```

## Decisions of record

The "what and why" is in the planning doc. Single-user with login; Caissa as
the benchmark data source via the user's key; Tier 3 PDF parsing via
Anthropic with per-upload + monthly USD caps; pct-vs-decimal is always
confirmed by the user in the correction UI.
