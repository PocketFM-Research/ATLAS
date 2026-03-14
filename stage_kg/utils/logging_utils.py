"""Logging configuration and prompt/response logging."""

import json
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    verbose: bool = False,
) -> None:
    """Configure root logger with console and optional file handler."""
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    handlers = [logging.StreamHandler()]
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(
            log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8",
        )
        handlers.append(fh)

    logging.basicConfig(
        level=logging.DEBUG if verbose else level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
    )


class PromptLogger:
    """
    Logs all prompts and raw LLM responses to a JSONL file for reproducibility.
    """

    def __init__(self, log_dir: Path, movie_id: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / f"prompts_{movie_id}.jsonl"

    def log(
        self,
        stage: str,
        scene_id: str,
        prompt: str,
        response: str,
        model: str = "",
    ) -> None:
        entry = {
            "ts": datetime.utcnow().isoformat(),
            "stage": stage,
            "scene_id": scene_id,
            "model": model,
            "prompt": prompt,
            "response": response,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
