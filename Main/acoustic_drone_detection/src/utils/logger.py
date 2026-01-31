"""
Advanced Logging Module

Provides comprehensive logging functionality with:
- Multiple output handlers (console, file, rotating)
- Structured logging format
- Performance logging
- Log level filtering
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional
import threading
import json


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors."""
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        # Format the message
        formatted = super().format(record)

        # Add color if terminal supports it
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            return f"{color}{formatted}{reset}"
        return formatted


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename',
                           'funcName', 'levelname', 'levelno', 'lineno',
                           'module', 'msecs', 'pathname', 'process',
                           'processName', 'relativeCreated', 'stack_info',
                           'exc_info', 'exc_text', 'thread', 'threadName',
                           'message', 'asctime']:
                log_data[key] = value

        return json.dumps(log_data)


class DroneDetectionLogger:
    """
    Advanced logger for the drone detection system.

    Features:
    - Console output with colors
    - File output with rotation
    - JSON structured logging option
    - Performance timing
    - Thread-safe operations
    """

    _instances: dict = {}
    _lock = threading.Lock()

    def __new__(cls, name: str = "DroneDetection", *args, **kwargs):
        """Create or return existing logger instance."""
        with cls._lock:
            if name not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[name] = instance
            return cls._instances[name]

    def __init__(
        self,
        name: str = "DroneDetection",
        level: int = logging.INFO,
        log_dir: Optional[str] = None,
        enable_console: bool = True,
        enable_file: bool = True,
        enable_json: bool = False,
        max_file_size: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5
    ):
        """
        Initialize the logger.

        Args:
            name: Logger name.
            level: Logging level.
            log_dir: Directory for log files.
            enable_console: Enable console output.
            enable_file: Enable file output.
            enable_json: Use JSON format for file logs.
            max_file_size: Maximum log file size before rotation.
            backup_count: Number of backup files to keep.
        """
        if hasattr(self, '_initialized') and self._initialized:
            return

        self._name = name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.handlers = []  # Clear existing handlers

        self._log_dir = log_dir or "logs"
        self._enable_console = enable_console
        self._enable_file = enable_file
        self._enable_json = enable_json
        self._max_file_size = max_file_size
        self._backup_count = backup_count

        self._setup_handlers()
        self._initialized = True

    def _setup_handlers(self) -> None:
        """Set up logging handlers."""
        # Console handler
        if self._enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self._logger.level)

            console_format = ColoredFormatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_format)
            self._logger.addHandler(console_handler)

        # File handler
        if self._enable_file:
            os.makedirs(self._log_dir, exist_ok=True)

            # Standard log file
            log_file = os.path.join(
                self._log_dir,
                f"{self._name}_{datetime.now().strftime('%Y%m%d')}.log"
            )

            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=self._max_file_size,
                backupCount=self._backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(self._logger.level)

            if self._enable_json:
                file_handler.setFormatter(JSONFormatter())
            else:
                file_format = logging.Formatter(
                    '%(asctime)s | %(levelname)-8s | %(name)s | '
                    '%(module)s:%(funcName)s:%(lineno)d | %(message)s'
                )
                file_handler.setFormatter(file_format)

            self._logger.addHandler(file_handler)

    @property
    def logger(self) -> logging.Logger:
        """Get the underlying logger instance."""
        return self._logger

    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message."""
        self._logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log an info message."""
        self._logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message."""
        self._logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log an error message."""
        self._logger.error(message, extra=kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log a critical message."""
        self._logger.critical(message, extra=kwargs)

    def exception(self, message: str, **kwargs) -> None:
        """Log an exception with traceback."""
        self._logger.exception(message, extra=kwargs)

    def detection(
        self,
        azimuth: float,
        elevation: float,
        distance: float,
        confidence: float,
        classification: str,
        threat_level: str
    ) -> None:
        """
        Log a drone detection event.

        Args:
            azimuth: Detection azimuth in degrees.
            elevation: Detection elevation in degrees.
            distance: Estimated distance in meters.
            confidence: Detection confidence (0-1).
            classification: Drone classification.
            threat_level: Threat level assessment.
        """
        self._logger.info(
            f"DETECTION | {classification} | "
            f"Az: {azimuth:.1f}deg | El: {elevation:.1f}deg | "
            f"Dist: {distance:.1f}m | Conf: {confidence:.1%} | "
            f"Threat: {threat_level}",
            extra={
                'event_type': 'detection',
                'azimuth': azimuth,
                'elevation': elevation,
                'distance': distance,
                'confidence': confidence,
                'classification': classification,
                'threat_level': threat_level
            }
        )

    def alert(self, threat_level: str, message: str) -> None:
        """
        Log an alert event.

        Args:
            threat_level: Alert threat level.
            message: Alert message.
        """
        level = logging.WARNING if threat_level == 'low' else \
                logging.ERROR if threat_level == 'medium' else logging.CRITICAL

        self._logger.log(
            level,
            f"ALERT [{threat_level.upper()}] | {message}",
            extra={'event_type': 'alert', 'threat_level': threat_level}
        )

    def performance(self, operation: str, duration_ms: float, **metrics) -> None:
        """
        Log a performance metric.

        Args:
            operation: Operation name.
            duration_ms: Duration in milliseconds.
            **metrics: Additional metrics to log.
        """
        self._logger.debug(
            f"PERF | {operation} | {duration_ms:.2f}ms",
            extra={'event_type': 'performance', 'operation': operation,
                   'duration_ms': duration_ms, **metrics}
        )

    def set_level(self, level: int) -> None:
        """Set the logging level."""
        self._logger.setLevel(level)
        for handler in self._logger.handlers:
            handler.setLevel(level)


class PerformanceTimer:
    """Context manager for timing operations."""

    def __init__(self, logger: DroneDetectionLogger, operation: str):
        """
        Initialize the timer.

        Args:
            logger: Logger instance.
            operation: Operation name being timed.
        """
        self.logger = logger
        self.operation = operation
        self.start_time = None

    def __enter__(self):
        """Start the timer."""
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the timer and log the duration."""
        duration = (datetime.now() - self.start_time).total_seconds() * 1000
        self.logger.performance(self.operation, duration)
        return False


# Global logger instance
_global_logger: Optional[DroneDetectionLogger] = None


def get_logger(name: str = "DroneDetection") -> DroneDetectionLogger:
    """
    Get or create a logger instance.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = DroneDetectionLogger(name)
    return _global_logger


def setup_logging(
    level: int = logging.INFO,
    log_dir: str = "logs",
    enable_json: bool = False
) -> DroneDetectionLogger:
    """
    Set up the global logging system.

    Args:
        level: Logging level.
        log_dir: Directory for log files.
        enable_json: Enable JSON formatting.

    Returns:
        Configured logger instance.
    """
    global _global_logger
    _global_logger = DroneDetectionLogger(
        name="DroneDetection",
        level=level,
        log_dir=log_dir,
        enable_json=enable_json
    )
    return _global_logger
