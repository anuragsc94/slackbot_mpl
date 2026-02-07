from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class BotProfile:
    name: str
    slack_bot_token: str
    slack_app_token: str
    bq_project_id: Optional[str] = None


def _require_str(d: Dict[str, Any], key: str, *, context: str) -> str:
    v = d.get(key)
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"Missing/invalid `{key}` for {context}")
    return v.strip()


def _resolve_token(spec: Dict[str, Any], *, token_key: str, env_key: str, context: str) -> str:
    """
    Resolve a token either from:
      - direct value: `token_key` (not recommended for real deployments)
      - env var reference: `env_key` -> read from os.environ
    """
    direct = spec.get(token_key)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    env_name = spec.get(env_key)
    if isinstance(env_name, str) and env_name.strip():
        val = os.getenv(env_name.strip())
        if not val:
            raise ValueError(f"Env var `{env_name.strip()}` not set for {context}")
        return val.strip()

    raise ValueError(f"Provide `{token_key}` or `{env_key}` for {context}")


def load_bot_profiles(path: str) -> List[BotProfile]:
    """
    Load bot profiles from a YAML config file.

    Supported format:

      bots:
        <bot_name>:
          slack_bot_token_env: SLACK_BOT_TOKEN_...
          slack_app_token_env: SLACK_APP_TOKEN_...
          bq_project_id: my-gcp-project

    For local/dev you may also set `slack_bot_token` / `slack_app_token` directly.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError("Bot config must be a YAML mapping")

    bots = raw.get("bots")
    if not isinstance(bots, dict) or not bots:
        raise ValueError("Bot config must contain a non-empty `bots:` mapping")

    out: List[BotProfile] = []
    for name, spec in bots.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Bot name keys under `bots:` must be non-empty strings")
        if not isinstance(spec, dict):
            raise ValueError(f"`bots.{name}` must be a mapping")

        ctx = f"bot `{name}`"
        slack_bot_token = _resolve_token(
            spec, token_key="slack_bot_token", env_key="slack_bot_token_env", context=ctx
        )
        slack_app_token = _resolve_token(
            spec, token_key="slack_app_token", env_key="slack_app_token_env", context=ctx
        )
        bq_project_id = spec.get("bq_project_id")
        if bq_project_id is not None and (not isinstance(bq_project_id, str) or not bq_project_id.strip()):
            raise ValueError(f"Invalid `bq_project_id` for {ctx}")

        out.append(
            BotProfile(
                name=name.strip(),
                slack_bot_token=slack_bot_token,
                slack_app_token=slack_app_token,
                bq_project_id=bq_project_id.strip() if isinstance(bq_project_id, str) else None,
            )
        )

    # stable ordering for predictable logs
    out.sort(key=lambda p: p.name)
    return out

