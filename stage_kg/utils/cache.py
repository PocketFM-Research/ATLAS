"""
Disk-based cache for intermediate pipeline outputs.

Cache keys include a SCHEMA_VERSION so that any change to the extraction
schema or prompts automatically invalidates existing cached results — preventing
the schema-drift bug where old (wrong) relation types are silently reused.

Bump SCHEMA_VERSION whenever you change:
  - relation type names or the valid-triples set (schema.py)
  - prompt templates (prompts/*.py)
  - extraction logic that changes the shape of output dicts
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Bump this whenever schema or prompts change.
SCHEMA_VERSION = "v3"   # v1=initial, v2=schema fix (owns->possesses etc), v3=reflection+embedding


class Cache:
    """Simple JSON file cache organized under a root directory."""

    def __init__(self, cache_dir: Path, schema_version: str = SCHEMA_VERSION):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._schema_version = schema_version

    def _key_to_path(self, key: str) -> Path:
        versioned = f"{self._schema_version}:{key}"
        safe = hashlib.md5(versioned.encode()).hexdigest()
        return self.cache_dir / f"{safe}.json"

    def get(self, key: str) -> Optional[Any]:
        path = self._key_to_path(key)
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                # Double-check the stored version matches (belt-and-suspenders)
                if payload.get("schema_version") != self._schema_version:
                    logger.debug(
                        "Cache version mismatch for key=%s (stored=%s, current=%s) — ignoring",
                        key, payload.get("schema_version"), self._schema_version,
                    )
                    return None
                logger.debug("Cache HIT: %s", key)
                return payload["value"]
            except Exception as e:
                logger.warning("Cache read error for key=%s: %s", key, e)
        return None

    def set(self, key: str, value: Any) -> None:
        path = self._key_to_path(key)
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(
                    {"schema_version": self._schema_version, "key": key, "value": value},
                    f,
                    ensure_ascii=False,
                )
            logger.debug("Cache SET: %s", key)
        except Exception as e:
            logger.warning("Cache write error for key=%s: %s", key, e)

    def exists(self, key: str) -> bool:
        return self._key_to_path(key).exists() and self.get(key) is not None

    @staticmethod
    def make_key(*parts: str) -> str:
        return ":".join(str(p) for p in parts)
