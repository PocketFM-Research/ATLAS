"""
Disk-based cache for intermediate pipeline outputs.

Keyed by (movie_id, stage, scene_id) so runs can resume without re-calling LLMs.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class Cache:
    """Simple JSON file cache organized under a root directory."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        safe = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe}.json"

    def get(self, key: str) -> Optional[Any]:
        path = self._key_to_path(key)
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                logger.debug("Cache HIT: %s", key)
                return payload["value"]
            except Exception as e:
                logger.warning("Cache read error for key=%s: %s", key, e)
        return None

    def set(self, key: str, value: Any) -> None:
        path = self._key_to_path(key)
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump({"key": key, "value": value}, f, ensure_ascii=False)
            logger.debug("Cache SET: %s", key)
        except Exception as e:
            logger.warning("Cache write error for key=%s: %s", key, e)

    def exists(self, key: str) -> bool:
        return self._key_to_path(key).exists()

    @staticmethod
    def make_key(*parts: str) -> str:
        return ":".join(str(p) for p in parts)
