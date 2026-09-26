"""Authentication stub: a fixed token table, no cryptography.

Tokens are synthetic benchmark values; nothing here is production auth.
"""

from __future__ import annotations

AUTH_TOKENS: dict[str, str] = {
    "tok-admin-001": "admin",
    "tok-writer-002": "writer",
    "tok-reader-003": "reader",
}

# Which roles may perform which actions.
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "admin": {"read", "write", "delete"},
    "writer": {"read", "write"},
    "reader": {"read"},
}


class AuthError(PermissionError):
    """Raised when a token is missing, unknown or lacks the action."""


def principal_for(token: str | None) -> str | None:
    """Map a bearer token to its role, or None when unknown."""
    if not token:
        return None
    return AUTH_TOKENS.get(token.strip())


def is_authorized(token: str | None, action: str) -> bool:
    """True when the token's role may perform the action."""
    role = principal_for(token)
    if role is None:
        return False
    return action in ROLE_PERMISSIONS.get(role, set())


def require_token(token: str | None, action: str) -> str:
    """Return the role when authorized; raise AuthError otherwise."""
    role = principal_for(token)
    if role is None:
        raise AuthError("missing or unknown token")
    if action not in ROLE_PERMISSIONS.get(role, set()):
        raise AuthError(f"role {role!r} may not {action!r}")
    return role
