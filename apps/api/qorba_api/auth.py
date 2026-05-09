"""Password hashing + signed session cookies.

Single-user-with-login (Section 8 decision #1). Argon2 hashing, HMAC-signed
cookie carrying (user_id, expiry). No JWTs, no third-party identity provider.
"""

from __future__ import annotations

import hmac
import json
import time
import uuid
from base64 import urlsafe_b64decode, urlsafe_b64encode
from hashlib import sha256

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Cookie, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from qorba_api.db.models import User
from qorba_api.db.session import get_db
from qorba_api.settings import Settings, get_settings

_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        _hasher.verify(password_hash, password)
        return True
    except VerifyMismatchError:
        return False


def _b64encode(raw: bytes) -> str:
    return urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return urlsafe_b64decode(s + pad)


def _sign(payload: bytes, secret: str) -> str:
    return _b64encode(hmac.new(secret.encode(), payload, sha256).digest())


def issue_session_cookie(response: Response, user_id: uuid.UUID, settings: Settings) -> None:
    expires_at = int(time.time()) + settings.session_max_age_seconds
    payload = json.dumps({"uid": str(user_id), "exp": expires_at}).encode()
    body = _b64encode(payload)
    sig = _sign(payload, settings.session_secret)
    response.set_cookie(
        settings.session_cookie_name,
        f"{body}.{sig}",
        max_age=settings.session_max_age_seconds,
        httponly=True,
        samesite="lax",
        secure=settings.env == "prod",
        path="/",
    )


def clear_session_cookie(response: Response, settings: Settings) -> None:
    response.delete_cookie(settings.session_cookie_name, path="/")


def _decode_session(cookie_value: str, secret: str) -> dict | None:
    try:
        body_b64, sig = cookie_value.split(".", 1)
    except ValueError:
        return None
    payload = _b64decode(body_b64)
    expected = _sign(payload, secret)
    if not hmac.compare_digest(expected, sig):
        return None
    data = json.loads(payload)
    if int(data.get("exp", 0)) < int(time.time()):
        return None
    return data


def get_current_user(
    qorba_session: str | None = Cookie(default=None, alias="qorba_session"),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> User:
    if not qorba_session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    data = _decode_session(qorba_session, settings.session_secret)
    if not data:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")
    user = db.get(User, uuid.UUID(data["uid"]))
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user
