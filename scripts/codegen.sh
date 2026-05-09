#!/usr/bin/env bash
# Generate TS types from the FastAPI OpenAPI schema.
#
#   scripts/codegen.sh           — regenerate, write to packages/shared/src/openapi.d.ts
#   scripts/codegen.sh --check   — fail if the generated file differs from committed
#
# Pulls the schema by importing the API directly (no running server needed).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_FILE="$REPO_ROOT/packages/shared/src/openapi.d.ts"
TMP_SCHEMA="$(mktemp -t qorba-openapi.XXXXXX.json)"
TMP_OUT="$(mktemp -t qorba-openapi.XXXXXX.d.ts)"
trap 'rm -f "$TMP_SCHEMA" "$TMP_OUT"' EXIT

PY="$REPO_ROOT/apps/api/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "API venv missing — run 'uv venv && uv pip install -e \".[dev]\"' inside apps/api first." >&2
    exit 2
fi

(cd "$REPO_ROOT/apps/api" && "$PY" -c "
import json, os
os.environ.setdefault('QORBA_DATABASE_URL', 'sqlite+pysqlite:///:memory:')
os.environ.setdefault('QORBA_SESSION_SECRET', 'codegen-secret-32-bytes-long-xxxx')
from qorba_api.main import create_app
print(json.dumps(create_app().openapi()))
") > "$TMP_SCHEMA"

(cd "$REPO_ROOT" && pnpm dlx openapi-typescript@7.4.4 "$TMP_SCHEMA" -o "$TMP_OUT" >/dev/null)

if [[ "${1:-}" == "--check" ]]; then
    if ! diff -q "$TMP_OUT" "$OUT_FILE" >/dev/null 2>&1; then
        echo "Generated TS types differ from committed $OUT_FILE." >&2
        echo "Run 'pnpm codegen' and commit the result." >&2
        diff -u "$OUT_FILE" "$TMP_OUT" || true
        exit 1
    fi
    echo "OpenAPI types are up to date."
else
    mv "$TMP_OUT" "$OUT_FILE"
    echo "Wrote $OUT_FILE"
fi
