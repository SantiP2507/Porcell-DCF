"""
utils/logger.py — Structured logging setup.

Configures root logger with:
  - Colored console output (INFO and above)
  - File handler for persistent DEBUG logs
"""

import logging
import sys
from pathlib import Path
from config import ROOT_DIR

LOG_FILE = ROOT_DIR / "dcf_engine.log"


def setup_logging(verbose: bool = False) -> None:
    """
    Call once at startup. Sets up console + file logging.

    Args:
        verbose: If True, show DEBUG messages in console too.
    """
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # capture everything, filter at handlers

    # Avoid duplicate handlers if called multiple times
    if root.handlers:
        return

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(_ColorFormatter("  %(levelname)s  %(message)s"))
    root.add_handler = root.addHandler  # silence type checker
    root.addHandler(ch)

    # File handler (always DEBUG)
    try:
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter("%(asctime)s  %(name)s  %(levelname)s  %(message)s")
        )
        root.addHandler(fh)
    except Exception:
        pass  # File logging is optional — don't crash if path is unwritable

    # Silence noisy third-party loggers
    for name in ["urllib3", "yfinance", "peewee", "requests"]:
        logging.getLogger(name).setLevel(logging.WARNING)


class _ColorFormatter(logging.Formatter):
    """Add simple ANSI colors to console log output."""

    COLORS = {
        logging.DEBUG:    "\033[90m",   # gray
        logging.INFO:     "\033[0m",    # default
        logging.WARNING:  "\033[33m",   # yellow
        logging.ERROR:    "\033[31m",   # red
        logging.CRITICAL: "\033[35m",   # magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        msg = super().format(record)
        return f"{color}{msg}{self.RESET}"
