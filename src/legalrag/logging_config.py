"""Structured (JSON) logging: one JSON object per log line."""

import contextvars
import json
import logging
import os
import sys
from datetime import datetime, timezone

# Holds the id of the request currently being handled, so every log line
# written while serving it carries the same request_id.
request_id_var = contextvars.ContextVar("request_id", default="-")

# Extra fields we allow on a log record (passed via logger.info(..., extra={...})).
_EXTRA_FIELDS = (
    "method",
    "path",
    "status",
    "latency_ms",
    "question_chars",
    "retrieved_articles",
    "retrieval_ms",
    "llm_ms",
)


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        for field in _EXTRA_FIELDS:
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging():
    """Send all logs to stdout as JSON. Safe to call more than once."""
    root = logging.getLogger()
    if getattr(root, "_legalrag_configured", False):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    # Keep HTTP-client libraries quiet so they never print request details.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    root._legalrag_configured = True
