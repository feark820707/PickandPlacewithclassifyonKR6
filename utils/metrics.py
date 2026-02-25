# =============================================================================
#  KR6 Pick & Place — 效能指標收集
#  追蹤每個 cycle 各階段耗時，計算吞吐量，供 UI 與日誌使用
# =============================================================================
from __future__ import annotations

import time
import threading
from collections import deque
from typing import Any


class CycleMetrics:
    """
    收集每個 Pick & Place 循環的效能指標。

    用法:
        metrics = CycleMetrics(window_size=100)

        # 在各階段量測耗時
        with metrics.measure("capture"):
            frame = camera.get_frame()

        with metrics.measure("yolo"):
            results = model(frame)

        # 記錄完整 cycle
        metrics.complete_cycle()

        # 取得摘要
        print(metrics.summary())
    """

    STAGES = ("capture", "yolo", "geometry", "opcua", "total")

    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: 滑動窗口大小（保留最近 N 個 cycle 的資料）
        """
        self._window_size = window_size
        self._lock = threading.Lock()

        # 各階段耗時（ms）
        self._timings: dict[str, deque[float]] = {
            stage: deque(maxlen=window_size) for stage in self.STAGES
        }

        # 計數器
        self.cycle_count: int = 0
        self.error_count: int = 0
        self.skip_count: int = 0

        # 當前 cycle 的計時器
        self._cycle_start: float | None = None
        self._stage_starts: dict[str, float] = {}

    # ---- 計時 Context Manager ----
    class _Timer:
        """Context manager for measuring stage duration"""

        def __init__(self, metrics: "CycleMetrics", stage: str):
            self._metrics = metrics
            self._stage = stage
            self._start: float = 0.0

        def __enter__(self):
            self._start = time.perf_counter()
            return self

        def __exit__(self, *exc_info):
            elapsed_ms = (time.perf_counter() - self._start) * 1000
            self._metrics.record(self._stage, elapsed_ms)

    def measure(self, stage: str) -> "_Timer":
        """
        Context manager 形式量測某階段耗時。

        Usage:
            with metrics.measure("yolo"):
                results = model(frame)
        """
        return self._Timer(self, stage)

    # ---- 手動記錄 ----
    def record(self, stage: str, elapsed_ms: float) -> None:
        """手動記錄某階段耗時（ms）"""
        with self._lock:
            if stage not in self._timings:
                self._timings[stage] = deque(maxlen=self._window_size)
            self._timings[stage].append(elapsed_ms)

    def start_cycle(self) -> None:
        """標記 cycle 開始"""
        self._cycle_start = time.perf_counter()

    def complete_cycle(self) -> None:
        """標記 cycle 完成，自動記錄 total 耗時"""
        with self._lock:
            self.cycle_count += 1
            if self._cycle_start is not None:
                total_ms = (time.perf_counter() - self._cycle_start) * 1000
                self._timings["total"].append(total_ms)
                self._cycle_start = None

    def record_error(self) -> None:
        """記錄錯誤計數"""
        with self._lock:
            self.error_count += 1

    def record_skip(self) -> None:
        """記錄跳過計數（如空幀、無偵測）"""
        with self._lock:
            self.skip_count += 1

    # ---- 統計查詢 ----
    def summary(self) -> dict[str, Any]:
        """
        取得效能摘要。

        Returns:
            {
                "cycles": 142,
                "errors": 2,
                "skips": 5,
                "avg_capture_ms": 12.3,
                "avg_yolo_ms": 25.1,
                "avg_geometry_ms": 3.2,
                "avg_opcua_ms": 8.5,
                "avg_total_ms": 55.6,
                "throughput_picks_per_sec": 2.1,
                "p95_total_ms": 72.3,
            }
        """
        with self._lock:
            result: dict[str, Any] = {
                "cycles": self.cycle_count,
                "errors": self.error_count,
                "skips": self.skip_count,
            }

            for stage in self.STAGES:
                data = self._timings.get(stage, deque())
                if data:
                    avg = sum(data) / len(data)
                    result[f"avg_{stage}_ms"] = round(avg, 1)
                else:
                    result[f"avg_{stage}_ms"] = 0.0

            # 吞吐量
            total_data = self._timings.get("total", deque())
            if total_data:
                total_sec = sum(total_data) / 1000
                result["throughput_picks_per_sec"] = round(
                    len(total_data) / max(total_sec, 0.001), 2
                )

                # P95
                sorted_total = sorted(total_data)
                p95_idx = int(len(sorted_total) * 0.95)
                result["p95_total_ms"] = round(sorted_total[min(p95_idx, len(sorted_total) - 1)], 1)
            else:
                result["throughput_picks_per_sec"] = 0.0
                result["p95_total_ms"] = 0.0

            return result

    def stage_avg(self, stage: str) -> float:
        """取得單一階段的平均耗時 (ms)"""
        with self._lock:
            data = self._timings.get(stage, deque())
            return sum(data) / len(data) if data else 0.0

    def reset(self) -> None:
        """重置所有指標"""
        with self._lock:
            for d in self._timings.values():
                d.clear()
            self.cycle_count = 0
            self.error_count = 0
            self.skip_count = 0
            self._cycle_start = None
