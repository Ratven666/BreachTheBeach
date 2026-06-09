import sys
from pathlib import Path

from loguru import logger

def setup_logger(log_path: str | Path = "logs/coastline_extractor.log") -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}:{function}:{line}</cyan> - "
               "<level>{message}</level>",
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    logger.add(
        str(log_path),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="10 days",
        compression="zip",
        backtrace=True,
        diagnose=True,
        encoding="utf-8",
    )
