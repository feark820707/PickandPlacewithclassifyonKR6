# =============================================================================
#  KR6 Pick & Place — 健康監控（心跳機制）
#  各模組定期回報心跳，超時則觸發告警 / 狀態降級
# =============================================================================
from __future__ import annotations

import logging
import time
import threading
from enum import Enum, auto
from typing import Callable

logger = logging.getLogger(__name__)


class ModuleStatus(Enum):
    OK = auto()
    WARN = auto()
    TIMEOUT = auto()
    UNKNOWN = auto()


class HealthMonitor:
    """
    健康監控器：各模組定期呼叫 beat()，監控器定期 check_all()。

    用法:
        monitor = HealthMonitor(timeout_sec=5.0)

        # 在各模組主迴圈中：
        monitor.beat("yolo_worker")
        monitor.beat("camera")

        # 在 Coordinator 中定期檢查：
        status = monitor.check_all()
        # {"yolo_worker": "OK", "camera": "TIMEOUT", ...}
    """

    def __init__(
        self,
        timeout_sec: float = 5.0,
        on_timeout: Callable[[str], None] | None = None,
    ):
        """
        Args:
            timeout_sec: 超時閾值（秒），模組未心跳超過此值 → TIMEOUT
            on_timeout:  超時回呼函式，接收模組名稱
        """
        self._heartbeats: dict[str, float] = {}
        self._metadata: dict[str, dict] = {}
        self._timeout = timeout_sec
        self._on_timeout = on_timeout
        self._lock = threading.Lock()

    def register(self, module_name: str) -> None:
        """註冊模組（可選，beat 時自動註冊）"""
        with self._lock:
            if module_name not in self._heartbeats:
                self._heartbeats[module_name] = time.time()
                self._metadata[module_name] = {}

    def beat(self, module_name: str, **metadata) -> None:
        """
        模組心跳回報。

        Args:
            module_name: 模組名稱
            **metadata:  額外資訊（如 fps=28.5, queue_size=3）
        """
        with self._lock:
            self._heartbeats[module_name] = time.time()
            if metadata:
                self._metadata[module_name] = metadata

    def check(self, module_name: str) -> ModuleStatus:
        """檢查單一模組的健康狀態"""
        with self._lock:
            if module_name not in self._heartbeats:
                return ModuleStatus.UNKNOWN
            elapsed = time.time() - self._heartbeats[module_name]
            if elapsed > self._timeout:
                return ModuleStatus.TIMEOUT
            elif elapsed > self._timeout * 0.8:
                return ModuleStatus.WARN
            return ModuleStatus.OK

    def check_all(self) -> dict[str, ModuleStatus]:
        """
        檢查所有已註冊模組的健康狀態。

        Returns:
            {module_name: ModuleStatus} 的 dict
        """
        now = time.time()
        statuses: dict[str, ModuleStatus] = {}

        with self._lock:
            modules = list(self._heartbeats.items())

        for name, last_beat in modules:
            elapsed = now - last_beat
            if elapsed > self._timeout:
                statuses[name] = ModuleStatus.TIMEOUT
                if self._on_timeout:
                    try:
                        self._on_timeout(name)
                    except Exception as e:
                        logger.error("心跳超時回呼失敗 (%s): %s", name, e)
            elif elapsed > self._timeout * 0.8:
                statuses[name] = ModuleStatus.WARN
            else:
                statuses[name] = ModuleStatus.OK

        return statuses

    def get_metadata(self, module_name: str) -> dict:
        """取得模組最近一次回報的 metadata"""
        with self._lock:
            return self._metadata.get(module_name, {}).copy()

    def summary(self) -> dict:
        """
        取得完整健康摘要，含心跳資訊。

        Returns:
            {
                "modules": {
                    "yolo_worker": {"status": "OK", "last_beat_ago_sec": 1.2, "fps": 28},
                    ...
                },
                "healthy": True/False
            }
        """
        now = time.time()
        modules_info = {}
        all_ok = True

        with self._lock:
            for name, last_beat in self._heartbeats.items():
                elapsed = now - last_beat
                if elapsed > self._timeout:
                    status = "TIMEOUT"
                    all_ok = False
                elif elapsed > self._timeout * 0.8:
                    status = "WARN"
                else:
                    status = "OK"

                info = {
                    "status": status,
                    "last_beat_ago_sec": round(elapsed, 2),
                }
                info.update(self._metadata.get(name, {}))
                modules_info[name] = info

        return {
            "modules": modules_info,
            "healthy": all_ok,
        }
