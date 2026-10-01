"""User configuration, authentication credential store, and run history manager.

Manages persistent client-side configuration stored in ~/.mintok/:
- credentials.json: authentication token and hosted API endpoint
- config.json: local default preferences (default model, budget, timeouts)
- history.jsonl: local append-only log of execution runs and credit spend
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_ENDPOINT = "https://api.mintok.ai/v1"
MINTOK_HOME = Path(os.getenv("MINTOK_HOME", Path.home() / ".mintok"))
CREDENTIALS_FILE = MINTOK_HOME / "credentials.json"
CONFIG_FILE = MINTOK_HOME / "config.json"
HISTORY_FILE = MINTOK_HOME / "history.jsonl"

DEFAULT_CONFIG: dict[str, Any] = {
    "default_model": "mintok-pro",
    "token_budget": 50000,
    "max_turns": 15,
    "telemetry": True,
    "auto_verify": True,
    "endpoint": DEFAULT_ENDPOINT,
}


def ensure_mintok_dir() -> Path:
    """Ensure ~/.mintok directory exists with secure permissions."""
    MINTOK_HOME.mkdir(parents=True, exist_ok=True)
    try:
        MINTOK_HOME.chmod(0o700)
    except Exception:
        pass
    return MINTOK_HOME


def save_credentials(api_key: str, endpoint: str = DEFAULT_ENDPOINT, email: str | None = None) -> None:
    """Save user authentication credentials securely."""
    ensure_mintok_dir()
    data = {
        "api_key": api_key,
        "endpoint": endpoint.rstrip("/"),
        "email": email or "",
    }
    CREDENTIALS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        CREDENTIALS_FILE.chmod(0o600)
    except Exception:
        pass


def load_credentials() -> dict[str, str]:
    """Load credentials from environment variable or ~/.mintok/credentials.json."""
    env_key = os.getenv("MINTOK_API_KEY")
    env_endpoint = os.getenv("MINTOK_API_ENDPOINT", DEFAULT_ENDPOINT)
    if env_key:
        return {
            "api_key": env_key,
            "endpoint": env_endpoint.rstrip("/"),
            "email": os.getenv("MINTOK_USER_EMAIL", ""),
        }

    if CREDENTIALS_FILE.exists():
        try:
            data = json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8"))
            return {
                "api_key": data.get("api_key", ""),
                "endpoint": data.get("endpoint", DEFAULT_ENDPOINT).rstrip("/"),
                "email": data.get("email", ""),
            }
        except Exception:
            return {"api_key": "", "endpoint": DEFAULT_ENDPOINT, "email": ""}

    return {"api_key": "", "endpoint": DEFAULT_ENDPOINT, "email": ""}


def clear_credentials() -> bool:
    """Remove stored credentials on logout."""
    if CREDENTIALS_FILE.exists():
        CREDENTIALS_FILE.unlink()
        return True
    return False


def load_all_config() -> dict[str, Any]:
    """Load client configuration merged with defaults."""
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            cfg.update(data)
        except Exception:
            pass
    return cfg


def get_config(key: str, default: Any = None) -> Any:
    """Get single configuration parameter."""
    cfg = load_all_config()
    return cfg.get(key, default)


def set_config(key: str, value: Any) -> None:
    """Set and persist a configuration parameter."""
    ensure_mintok_dir()
    cfg = load_all_config()
    cfg[key] = value
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def record_run_history(run_data: dict[str, Any]) -> None:
    """Append execution run telemetry record to local history.jsonl."""
    ensure_mintok_dir()
    with HISTORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(run_data) + "\n")


def get_run_history(limit: int = 10) -> list[dict[str, Any]]:
    """Retrieve the most recent run records from local history."""
    if not HISTORY_FILE.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        lines = HISTORY_FILE.read_text(encoding="utf-8").splitlines()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
                if len(records) >= limit:
                    break
            except Exception:
                continue
    except Exception:
        pass
    return records
