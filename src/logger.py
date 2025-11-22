import codecs
import glob
import gzip
import json
import logging
import logging.handlers
import os
import time
from datetime import datetime
from typing import Annotated

import structlog
from fastapi import Depends, Request
from structlog.processors import CallsiteParameter

from app.config import settings


class TimedCompressedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """
    Extended version of TimedRotatingFileHandler that compress logs on rollover.
    """

    def doRollover(self):
        """
        do a rollover; in this case, a date/time stamp is appended to the filename
        when the rollover happens.  However, you want the file to be named for the
        start of the interval, not the current time.  If there is a backup count,
        then we have to get a list of matching filenames, sort them and remove
        the one with the oldest suffix.
        """
        self.stream.close()
        t = self.rolloverAt - self.interval
        timeTuple = time.localtime(t)
        dfn = self.baseFilename + "." + time.strftime(self.suffix, timeTuple)
        if os.path.exists(dfn):
            os.remove(dfn)
        os.rename(self.baseFilename, dfn)
        if self.backupCount > 0:
            # find the oldest log file and delete it
            s = glob.glob(self.baseFilename + ".20*")
            if len(s) > self.backupCount:
                s.sort()
                os.remove(s[0])
        self.stream = codecs.open(self.baseFilename, "w", "utf-8")
        self.rolloverAt = self.rolloverAt + self.interval
        if os.path.exists(dfn + ".gz"):
            os.remove(dfn + ".gz")
        with open(dfn, "rb") as f_in, gzip.open(dfn + ".gz", "wb") as f_out:
            f_out.writelines(f_in)
        os.remove(dfn)


class UnstructuredLoggingFormatter(logging.Formatter):
    def format(self, record):
        try:
            log_data = json.loads(record.getMessage())
        except json.JSONDecodeError:
            log_data = {"message": record.getMessage()}

        log_message_parts = []
        log_level = log_data.get("level", record.levelname).upper()
        log_message_parts.append(log_level)

        request_start = log_data.get("request_start")
        request_ended = log_data.get("request_ended")
        if request_start and request_ended:
            try:
                start_time = datetime.fromisoformat(request_start)
                end_time = datetime.fromisoformat(request_ended)
                duration = end_time - start_time
                duration_ms = int(duration.total_seconds() * 1000)
                log_message_parts.append(f"{duration_ms} ms")
            except (ValueError, TypeError) as e:
                log_message_parts.append(f"unable to compute time {e}")

        excluded_keys = {"request_start", "request_ended", "level"}
        for key, value in log_data.items():
            if key not in excluded_keys and value:
                log_message_parts.append(f"{value}")

        if record.exc_info:
            log_message_parts.append(self.formatException(record.exc_info))

        return " | ".join(log_message_parts)


def setup_logging(log_file: str):
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler()

    backup_count = 30
    console_handler.setFormatter(UnstructuredLoggingFormatter())

    file_handler = TimedCompressedRotatingFileHandler(
        log_file, when="midnight", interval=1, backupCount=backup_count
    )
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    handlers = [file_handler]
    if settings.ENV == "local":
        handlers.append(console_handler)

    logging.basicConfig(
        level=settings.log_level,
        format="%(message)s",
        handlers=handlers,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.format_exc_info,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.CallsiteParameterAdder(
                parameters=[CallsiteParameter.FUNC_NAME]
            ),
            structlog.stdlib.filter_by_level,
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logging.getLogger("boto").setLevel(logging.CRITICAL)
    logging.getLogger("uvicorn.error").disabled = True
    logging.getLogger("uvicorn.access").disabled = True
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("tqdm").setLevel(logging.ERROR)
    logging.getLogger("googlemaps").setLevel(logging.ERROR)


LoggerType = structlog.stdlib.BoundLogger


def get_logger(request: Request) -> structlog.stdlib.BoundLogger:
    """
    FastAPI dependency to get the request-bound logger.
    """
    try:
        return request.state.logger
    except AttributeError:
        return structlog.get_logger()


Logger = Annotated[structlog.stdlib.BoundLogger, Depends(get_logger)]
