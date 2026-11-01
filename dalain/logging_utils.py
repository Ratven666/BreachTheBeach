from contextlib import contextmanager
import sys
import time

from loguru import logger


def configure_logging(cfg, directory):
    logger.remove()

    logger.add(
        sys.stderr,
        level=cfg.console_log_level.upper(),
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        ),
        colorize=sys.stderr.isatty(),
        backtrace=True,
        diagnose=False,
    )

    logger.add(
        directory / "calculation.log",
        level=cfg.file_log_level.upper(),
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | {name}:{function}:{line} | "
            "{message}"
        ),
        rotation=cfg.log_rotation,
        encoding="utf-8",
        backtrace=True,
        diagnose=False,
        enqueue=False,
    )


@contextmanager
def stage(name):
    started = time.perf_counter()
    logger.info("НАЧАЛО | {}", name)

    try:
        yield
    except BaseException:
        logger.error(
            "ОШИБКА | {} | прошло {:.2f} с",
            name,
            time.perf_counter() - started,
        )
        raise
    else:
        logger.success(
            "ГОТОВО | {} | {:.2f} с",
            name,
            time.perf_counter() - started,
        )


class Progress:
    def __init__(self, name, total, interval=5.0, every=10):
        self.name = name
        self.total = total
        self.interval = interval
        self.every = every
        self.started = time.perf_counter()
        self.last = self.started

    def update(self, completed, extra=""):
        now = time.perf_counter()

        if (
            completed != self.total
            and completed % self.every
            and now - self.last < self.interval
        ):
            return

        elapsed = now - self.started
        speed = completed / elapsed if elapsed > 0 else 0
        remaining = (
            (self.total - completed) / speed
            if speed > 0 else 0
        )

        logger.info(
            "{} | {}/{} ({:.1f}%) | {:.1f} с | "
            "ETA {:.1f} с | {}",
            self.name,
            completed,
            self.total,
            100 * completed / self.total if self.total else 100,
            elapsed,
            remaining,
            extra,
        )

        self.last = now