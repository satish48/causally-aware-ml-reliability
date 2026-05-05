from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

_CONFIG_SIGNATURE: tuple[str, bool, str] | None = None


class JsonFormatter(logging.Formatter):
    def __init__(self, service_name: str) -> None:
        super().__init__()
        self._service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": self._service_name,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(*, level: str, json_format: bool, service_name: str) -> None:
    global _CONFIG_SIGNATURE
    signature = (level.upper(), json_format, service_name)
    if _CONFIG_SIGNATURE == signature:
        return

    root = logging.getLogger()
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(JsonFormatter(service_name=service_name))
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )

    root.addHandler(handler)
    root.setLevel(getattr(logging, signature[0], logging.INFO))

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(root.level)

    _CONFIG_SIGNATURE = signature
