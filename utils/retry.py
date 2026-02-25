# =============================================================================
#  KR6 Pick & Place — 通用重試裝飾器 + Circuit Breaker
#  用於 OPC-UA 連線、PLC 寫入等可能暫時失敗的操作
# =============================================================================
from __future__ import annotations

import functools
import logging
import time
import threading
from enum import Enum, auto
from typing import Callable, Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Retry Decorator
# ---------------------------------------------------------------------------
def retry(
    max_retries: int = 3,
    delay: float = 2.0,
    backoff_factor: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_fail: Callable[[Exception], None] | None = None,
):
    """
    通用重試裝飾器，支援指數退避。

    Args:
        max_retries:    最大重試次數
        delay:          初始延遲（秒）
        backoff_factor: 延遲倍增因子（1.0=固定延遲, 1.5=指數退避）
        exceptions:     要捕捉的例外類別
        on_fail:        所有重試失敗後的回呼函式

    Usage:
        @retry(max_retries=3, delay=2.0, on_fail=lambda e: enter_safe_stop())
        def write_to_plc(data):
            ...
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logger.warning(
                        "%s 第 %d/%d 次失敗: %s",
                        func.__name__, attempt, max_retries, e,
                    )
                    if attempt < max_retries:
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(
                            "%s 已達最大重試次數 (%d)，放棄。",
                            func.__name__, max_retries,
                        )
                        if on_fail:
                            on_fail(last_exception)
                        raise

        return wrapper
    return decorator


# ---------------------------------------------------------------------------
#  Circuit Breaker
# ---------------------------------------------------------------------------
class CircuitState(Enum):
    CLOSED = auto()     # 正常：所有請求通過
    OPEN = auto()       # 斷路：所有請求直接失敗
    HALF_OPEN = auto()  # 半開：允許一次探測請求


class CircuitBreakerError(Exception):
    """Circuit Breaker 處於 OPEN 狀態時拋出"""
    pass


class CircuitBreaker:
    """
    Circuit Breaker 模式 — 防止連續失敗導致系統雪崩。

    當連續失敗達到閾值時，自動斷路（OPEN），停止嘗試一段時間。
    冷卻期後進入 HALF_OPEN，嘗試一次：
      - 成功 → 恢復 CLOSED
      - 失敗 → 繼續 OPEN

    Usage:
        cb = CircuitBreaker(fail_threshold=3, reset_timeout=10.0)

        @cb
        def call_plc():
            ...

        # 或手動使用：
        try:
            cb.call(call_plc, arg1, arg2)
        except CircuitBreakerError:
            logger.error("PLC 斷路，等待恢復")
    """

    def __init__(
        self,
        fail_threshold: int = 3,
        reset_timeout: float = 10.0,
        name: str = "default",
    ):
        self.fail_threshold = fail_threshold
        self.reset_timeout = reset_timeout
        self.name = name

        self._state = CircuitState.CLOSED
        self._fail_count = 0
        self._last_fail_time: float = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if (
                self._state == CircuitState.OPEN
                and time.time() - self._last_fail_time >= self.reset_timeout
            ):
                self._state = CircuitState.HALF_OPEN
                logger.info(
                    "CircuitBreaker[%s] OPEN → HALF_OPEN (嘗試恢復)",
                    self.name,
                )
            return self._state

    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """透過 circuit breaker 呼叫函式"""
        current_state = self.state

        if current_state == CircuitState.OPEN:
            raise CircuitBreakerError(
                f"CircuitBreaker[{self.name}] 處於 OPEN 狀態，"
                f"需等待 {self.reset_timeout}s 冷卻"
            )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        with self._lock:
            self._fail_count = 0
            if self._state == CircuitState.HALF_OPEN:
                logger.info(
                    "CircuitBreaker[%s] HALF_OPEN → CLOSED (恢復正常)",
                    self.name,
                )
            self._state = CircuitState.CLOSED

    def _on_failure(self) -> None:
        with self._lock:
            self._fail_count += 1
            self._last_fail_time = time.time()

            if self._fail_count >= self.fail_threshold:
                self._state = CircuitState.OPEN
                logger.error(
                    "CircuitBreaker[%s] → OPEN (連續失敗 %d 次)",
                    self.name, self._fail_count,
                )

    def reset(self) -> None:
        """手動重置 circuit breaker"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._fail_count = 0
            logger.info("CircuitBreaker[%s] 手動重置為 CLOSED", self.name)

    def __call__(self, func: Callable) -> Callable:
        """作為裝飾器使用"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
