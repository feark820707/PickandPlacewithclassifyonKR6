# =============================================================================
#  Unit Tests — 系統狀態機
# =============================================================================
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from coordinator.state_machine import (
    ErrorLevel,
    StateMachine,
    SystemState,
    VALID_TRANSITIONS,
)


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sm():
    """建立預設狀態機（INITIALIZING）"""
    return StateMachine()


@pytest.fixture
def sm_ready():
    """建立已進入 READY 的狀態機"""
    machine = StateMachine()
    machine.transition_to(SystemState.READY)
    return machine


@pytest.fixture
def sm_running(sm_ready):
    """建立已進入 RUNNING 的狀態機"""
    sm_ready.transition_to(SystemState.RUNNING)
    return sm_ready


# ---------------------------------------------------------------------------
#  Tests: 基本狀態查詢
# ---------------------------------------------------------------------------
class TestInitialState:
    def test_initial_state(self, sm):
        assert sm.state == SystemState.INITIALIZING

    def test_is_not_operational_at_init(self, sm):
        assert not sm.is_operational()


# ---------------------------------------------------------------------------
#  Tests: 合法轉換
# ---------------------------------------------------------------------------
class TestValidTransitions:
    def test_init_to_ready(self, sm):
        result = sm.transition_to(SystemState.READY)
        assert result is True
        assert sm.state == SystemState.READY

    def test_ready_to_running(self, sm_ready):
        result = sm_ready.transition_to(SystemState.RUNNING)
        assert result is True
        assert sm_ready.state == SystemState.RUNNING

    def test_running_to_ready(self, sm_running):
        """正常完成 → READY"""
        result = sm_running.transition_to(SystemState.READY)
        assert result is True
        assert sm_running.state == SystemState.READY

    def test_running_to_error(self, sm_running):
        result = sm_running.enter_error(ErrorLevel.RETRY, "test error")
        assert result is True
        assert sm_running.state == SystemState.ERROR

    def test_error_to_running(self, sm_running):
        """錯誤恢復 → RUNNING"""
        sm_running.enter_error(ErrorLevel.RETRY, "recoverable")
        result = sm_running.transition_to(SystemState.RUNNING)
        assert result is True
        assert sm_running.state == SystemState.RUNNING

    def test_error_to_safe_stop(self, sm_running):
        """嚴重錯誤 → SAFE_STOP"""
        sm_running.enter_error(ErrorLevel.CRITICAL, "fatal")
        result = sm_running.enter_safe_stop("unrecoverable")
        assert result is True
        assert sm_running.state == SystemState.SAFE_STOP

    def test_safe_stop_to_init(self):
        """SAFE_STOP → INITIALIZING（重啟）"""
        sm = StateMachine()
        sm.transition_to(SystemState.READY)
        sm.transition_to(SystemState.SAFE_STOP)
        result = sm.transition_to(SystemState.INITIALIZING)
        assert result is True

    def test_same_state_returns_true(self, sm_ready):
        """相同狀態 → True（不轉換）"""
        result = sm_ready.transition_to(SystemState.READY)
        assert result is True


# ---------------------------------------------------------------------------
#  Tests: 非法轉換
# ---------------------------------------------------------------------------
class TestInvalidTransitions:
    def test_init_to_running(self, sm):
        """INITIALIZING → RUNNING 不合法（需先 READY）"""
        result = sm.transition_to(SystemState.RUNNING)
        assert result is False
        assert sm.state == SystemState.INITIALIZING

    def test_ready_to_init(self, sm_ready):
        """READY → INITIALIZING 不合法"""
        result = sm_ready.transition_to(SystemState.INITIALIZING)
        assert result is False

    def test_safe_stop_to_ready(self):
        """SAFE_STOP → READY 不合法（只能重啟到 INIT）"""
        sm = StateMachine()
        sm.transition_to(SystemState.READY)
        sm.transition_to(SystemState.SAFE_STOP)
        result = sm.transition_to(SystemState.READY)
        assert result is False


# ---------------------------------------------------------------------------
#  Tests: 錯誤追蹤
# ---------------------------------------------------------------------------
class TestErrorTracking:
    def test_error_level_recorded(self, sm_running):
        sm_running.enter_error(ErrorLevel.WARN, "skip object")
        assert sm_running.error_level == ErrorLevel.WARN
        assert sm_running.error_message == "skip object"

    def test_consecutive_errors(self, sm_running):
        """連續錯誤計數 — 回到 RUNNING 會重置，需連續 ERROR→ERROR 才累加"""
        sm_running.enter_error(ErrorLevel.RETRY, "err1")
        assert sm_running.consecutive_errors == 1

        # 回到 RUNNING 時 consecutive_errors 歸零（因為恢復成功）
        sm_running.transition_to(SystemState.RUNNING)
        assert sm_running.consecutive_errors == 0

        # 再次進入 ERROR
        sm_running.enter_error(ErrorLevel.RETRY, "err2")
        assert sm_running.consecutive_errors == 1

    def test_errors_reset_on_success(self, sm_running):
        """回到正常後，連續錯誤計數歸零"""
        sm_running.enter_error(ErrorLevel.RETRY, "err")
        sm_running.transition_to(SystemState.RUNNING)
        sm_running.transition_to(SystemState.READY)
        assert sm_running.consecutive_errors == 0
        assert sm_running.error_level is None


# ---------------------------------------------------------------------------
#  Tests: 回呼機制
# ---------------------------------------------------------------------------
class TestCallbacks:
    def test_on_enter_callback(self, sm):
        entered = []
        sm.on_enter(SystemState.READY, lambda old, new: entered.append(new))
        sm.transition_to(SystemState.READY)
        assert entered == [SystemState.READY]

    def test_on_exit_callback(self, sm_ready):
        exited = []
        sm_ready.on_exit(SystemState.READY, lambda old, new: exited.append(old))
        sm_ready.transition_to(SystemState.RUNNING)
        assert exited == [SystemState.READY]

    def test_on_any_transition(self, sm):
        transitions = []
        sm.on_any_transition(lambda old, new: transitions.append((old, new)))
        sm.transition_to(SystemState.READY)
        sm.transition_to(SystemState.RUNNING)
        assert len(transitions) == 2
        assert transitions[0] == (SystemState.INITIALIZING, SystemState.READY)
        assert transitions[1] == (SystemState.READY, SystemState.RUNNING)

    def test_callback_error_does_not_break(self, sm):
        """回呼拋出例外不應影響狀態轉換"""
        sm.on_enter(SystemState.READY, lambda old, new: 1 / 0)
        result = sm.transition_to(SystemState.READY)
        assert result is True
        assert sm.state == SystemState.READY


# ---------------------------------------------------------------------------
#  Tests: 便捷方法與摘要
# ---------------------------------------------------------------------------
class TestConvenience:
    def test_is_operational(self, sm):
        assert not sm.is_operational()
        sm.transition_to(SystemState.READY)
        assert sm.is_operational()
        sm.transition_to(SystemState.RUNNING)
        assert sm.is_operational()
        sm.enter_error(ErrorLevel.CRITICAL, "test")
        assert not sm.is_operational()

    def test_state_duration(self, sm_ready):
        import time
        time.sleep(0.05)
        assert sm_ready.state_duration >= 0.04

    def test_summary(self, sm_running):
        sm_running.enter_error(ErrorLevel.RETRY, "test error")
        summary = sm_running.summary()
        assert summary["state"] == "ERROR"
        assert summary["error_level"] == "RETRY"
        assert summary["consecutive_errors"] == 1
        assert len(summary["recent_transitions"]) >= 1


# ---------------------------------------------------------------------------
#  Tests: 所有合法轉換路徑
# ---------------------------------------------------------------------------
class TestAllValidPaths:
    @pytest.mark.parametrize(
        "from_state,to_states",
        list(VALID_TRANSITIONS.items()),
    )
    def test_valid_paths(self, from_state, to_states):
        """確認 VALID_TRANSITIONS 表中每條路徑都能走通"""
        for to_state in to_states:
            sm = StateMachine(initial_state=from_state)
            result = sm.transition_to(to_state)
            assert result is True, f"{from_state.name} → {to_state.name} should be valid"
