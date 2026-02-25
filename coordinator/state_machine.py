# =============================================================================
#  KR6 Pick & Place — 系統狀態機
#  有限狀態機管理系統生命週期，所有狀態轉換皆寫入日誌
# =============================================================================
from __future__ import annotations

import logging
import threading
import time
from enum import Enum, auto
from typing import Callable, Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  State & Error Level Enums
# ---------------------------------------------------------------------------
class SystemState(Enum):
    INITIALIZING = auto()  # 啟動中：載入設定、連線相機/PLC
    READY = auto()         # 就緒：等待 PLC cmd 或手動觸發
    RUNNING = auto()       # 執行中：pick & place 循環
    ERROR = auto()         # 錯誤：依嚴重等級處理
    SAFE_STOP = auto()     # 安全停機：真空關閉、手臂歸位


class ErrorLevel(Enum):
    WARN = auto()      # 警告：跳過此物件，繼續下一個
    RETRY = auto()     # 重試：重新擷取/推論（最多 N 次）
    CRITICAL = auto()  # 嚴重：進入 SAFE_STOP


# ---------------------------------------------------------------------------
#  Valid Transitions
# ---------------------------------------------------------------------------
# 定義合法的狀態轉換
VALID_TRANSITIONS: dict[SystemState, set[SystemState]] = {
    SystemState.INITIALIZING: {SystemState.READY, SystemState.ERROR, SystemState.SAFE_STOP},
    SystemState.READY:        {SystemState.RUNNING, SystemState.ERROR, SystemState.SAFE_STOP},
    SystemState.RUNNING:      {SystemState.READY, SystemState.ERROR, SystemState.SAFE_STOP},
    SystemState.ERROR:        {SystemState.RUNNING, SystemState.READY, SystemState.SAFE_STOP},
    SystemState.SAFE_STOP:    {SystemState.INITIALIZING},  # 只能重新初始化
}


# ---------------------------------------------------------------------------
#  State Machine
# ---------------------------------------------------------------------------
class StateMachine:
    """
    系統狀態機，管理 Pick & Place 系統的完整生命週期。

    狀態轉換：
        INITIALIZING → READY → RUNNING → READY (正常循環)
                                       → ERROR (異常)
                                         → RUNNING (恢復)
                                         → SAFE_STOP (不可恢復)
        SAFE_STOP → INITIALIZING (重啟)

    用法:
        sm = StateMachine()
        sm.on_enter(SystemState.ERROR, handle_error)
        sm.on_exit(SystemState.RUNNING, cleanup_running)

        sm.transition_to(SystemState.READY)
        sm.transition_to(SystemState.RUNNING)
    """

    def __init__(self, initial_state: SystemState = SystemState.INITIALIZING):
        self._state = initial_state
        self._lock = threading.RLock()
        self._state_since: float = time.time()
        self._transition_count: int = 0
        self._history: list[tuple[float, SystemState, SystemState]] = []

        # 回呼函式
        self._on_enter: dict[SystemState, list[Callable]] = {s: [] for s in SystemState}
        self._on_exit: dict[SystemState, list[Callable]] = {s: [] for s in SystemState}
        self._on_any_transition: list[Callable] = []

        # 錯誤追蹤
        self._error_level: ErrorLevel | None = None
        self._error_message: str = ""
        self._consecutive_errors: int = 0

    # ---- 狀態查詢 ----
    @property
    def state(self) -> SystemState:
        """取得目前狀態"""
        with self._lock:
            return self._state

    @property
    def state_duration(self) -> float:
        """取得目前狀態持續時間（秒）"""
        with self._lock:
            return time.time() - self._state_since

    @property
    def error_level(self) -> ErrorLevel | None:
        """取得目前錯誤等級（僅 ERROR 狀態有效）"""
        with self._lock:
            return self._error_level

    @property
    def error_message(self) -> str:
        with self._lock:
            return self._error_message

    @property
    def consecutive_errors(self) -> int:
        with self._lock:
            return self._consecutive_errors

    # ---- 狀態轉換 ----
    def transition_to(
        self,
        new_state: SystemState,
        error_level: ErrorLevel | None = None,
        error_message: str = "",
        cycle_id: int | None = None,
    ) -> bool:
        """
        嘗試轉換到新狀態。

        Args:
            new_state:     目標狀態
            error_level:   錯誤等級（僅轉換到 ERROR 時需要）
            error_message: 錯誤訊息
            cycle_id:      當前 cycle ID（用於日誌追蹤）

        Returns:
            True=轉換成功, False=轉換不合法

        Raises:
            ValueError: 無效的狀態轉換
        """
        with self._lock:
            old_state = self._state

            # 相同狀態不轉換
            if old_state == new_state:
                return True

            # 檢查合法轉換
            if new_state not in VALID_TRANSITIONS.get(old_state, set()):
                logger.error(
                    "非法狀態轉換: %s → %s",
                    old_state.name, new_state.name,
                    extra={"cycle_id": cycle_id},
                )
                return False

            # 執行 on_exit 回呼
            for callback in self._on_exit[old_state]:
                try:
                    callback(old_state, new_state)
                except Exception as e:
                    logger.error("on_exit 回呼失敗 (%s): %s", old_state.name, e)

            # 更新狀態
            self._state = new_state
            self._state_since = time.time()
            self._transition_count += 1
            self._history.append((time.time(), old_state, new_state))

            # 保留最近 1000 筆歷史
            if len(self._history) > 1000:
                self._history = self._history[-500:]

            # 錯誤追蹤
            if new_state == SystemState.ERROR:
                self._error_level = error_level
                self._error_message = error_message
                self._consecutive_errors += 1
            elif new_state in (SystemState.READY, SystemState.RUNNING):
                self._consecutive_errors = 0
                self._error_level = None
                self._error_message = ""

            # 日誌
            log_data = {
                "from": old_state.name,
                "to": new_state.name,
                "transition_count": self._transition_count,
            }
            if error_level:
                log_data["error_level"] = error_level.name
                log_data["error_message"] = error_message

            logger.info(
                "狀態轉換: %s → %s",
                old_state.name, new_state.name,
                extra={"cycle_id": cycle_id, "data": log_data},
            )

            # 執行 on_enter 回呼
            for callback in self._on_enter[new_state]:
                try:
                    callback(old_state, new_state)
                except Exception as e:
                    logger.error("on_enter 回呼失敗 (%s): %s", new_state.name, e)

            # 執行全局轉換回呼
            for callback in self._on_any_transition:
                try:
                    callback(old_state, new_state)
                except Exception as e:
                    logger.error("on_any_transition 回呼失敗: %s", e)

            return True

    # ---- 回呼註冊 ----
    def on_enter(self, state: SystemState, callback: Callable) -> None:
        """註冊進入某狀態時的回呼"""
        self._on_enter[state].append(callback)

    def on_exit(self, state: SystemState, callback: Callable) -> None:
        """註冊離開某狀態時的回呼"""
        self._on_exit[state].append(callback)

    def on_any_transition(self, callback: Callable) -> None:
        """註冊任何狀態轉換時的回呼"""
        self._on_any_transition.append(callback)

    # ---- 便捷方法 ----
    def enter_error(
        self,
        level: ErrorLevel,
        message: str,
        cycle_id: int | None = None,
    ) -> bool:
        """進入 ERROR 狀態的便捷方法"""
        return self.transition_to(
            SystemState.ERROR,
            error_level=level,
            error_message=message,
            cycle_id=cycle_id,
        )

    def enter_safe_stop(self, reason: str = "", cycle_id: int | None = None) -> bool:
        """進入 SAFE_STOP 狀態的便捷方法"""
        logger.critical(
            "進入安全停機: %s", reason,
            extra={"cycle_id": cycle_id},
        )
        return self.transition_to(
            SystemState.SAFE_STOP,
            error_message=reason,
            cycle_id=cycle_id,
        )

    def is_operational(self) -> bool:
        """系統是否處於可運作狀態（READY 或 RUNNING）"""
        return self._state in (SystemState.READY, SystemState.RUNNING)

    # ---- 狀態查詢 ----
    def summary(self) -> dict[str, Any]:
        """取得狀態機摘要"""
        with self._lock:
            return {
                "state": self._state.name,
                "state_duration_sec": round(time.time() - self._state_since, 2),
                "transition_count": self._transition_count,
                "consecutive_errors": self._consecutive_errors,
                "error_level": self._error_level.name if self._error_level else None,
                "error_message": self._error_message or None,
                "recent_transitions": [
                    {
                        "time": t,
                        "from": f.name,
                        "to": to.name,
                    }
                    for t, f, to in self._history[-10:]
                ],
            }
