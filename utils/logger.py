# =============================================================================
#  KR6 Pick & Place — 結構化 JSON 日誌系統
#  統一日誌格式，支援 cycle_id 追蹤、每日輪替、JSON 結構化輸出
# =============================================================================
from __future__ import annotations

import json
import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    """
    將 log record 格式化為單行 JSON，方便 ELK / Grafana Loki 解析。

    輸出格式：
    {"timestamp": "...", "level": "INFO", "module": "yolo_worker",
     "cycle_id": 42, "message": "偵測完成", "data": {"label": "classA"}}
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "module": record.name,
            "cycle_id": getattr(record, "cycle_id", None),
            "message": record.getMessage(),
        }

        # 附帶結構化資料
        data = getattr(record, "data", None)
        if data is not None:
            log_entry["data"] = data

        # 例外資訊
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """開發階段用的可讀格式"""

    FMT = "%(asctime)s [%(levelname)-5s] %(name)s | %(message)s"

    def __init__(self):
        super().__init__(fmt=self.FMT, datefmt="%Y-%m-%d %H:%M:%S")


def setup_logger(
    name: str,
    log_dir: str = "logs",
    level: str = "DEBUG",
    retention_days: int = 30,
    console_level: str = "INFO",
    use_json: bool = True,
) -> logging.Logger:
    """
    建立並回傳一個已配置的 Logger。

    Args:
        name:           模組名稱（如 "yolo_worker", "coordinator"）
        log_dir:        日誌輸出目錄
        level:          檔案日誌的最低等級
        retention_days: 日誌檔保留天數
        console_level:  Console 輸出的最低等級
        use_json:       True=JSON 格式, False=可讀文字格式

    Returns:
        已配置的 logging.Logger

    使用範例:
        logger = setup_logger("yolo_worker")
        logger.info("偵測完成", extra={"cycle_id": 42, "data": {"label": "classA"}})
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))

    # 避免重複加 handler（多次呼叫 setup_logger 時）
    if logger.handlers:
        return logger

    formatter = JsonFormatter() if use_json else TextFormatter()

    # ---- 檔案 Handler：每日輪替 ----
    file_handler = TimedRotatingFileHandler(
        filename=str(Path(log_dir) / f"{name}.jsonl"),
        when="midnight",
        interval=1,
        backupCount=retention_days,
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # ---- Console Handler ----
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 防止日誌向上傳播到 root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    取得已存在的 Logger（不重新設定 handler）。
    若尚未呼叫 setup_logger，回傳預設 logger。
    """
    return logging.getLogger(name)
