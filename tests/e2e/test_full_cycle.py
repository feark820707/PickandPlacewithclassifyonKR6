# =============================================================================
#  End-to-End Tests — 完整 Pick & Place 流程
#  Mock 環境下驗證：Config → Camera → YOLO → Geometry → OPC-UA 全管線
#
#  ⚠️  實體測試需要：D435/Cognex + PLC + KR6
#      此檔案僅在 Mock 環境下驗證端對端邏輯正確性
# =============================================================================
import sys
import threading
import time
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from camera.base import CameraBase
from communication.opcua_bridge import DB1Data
from config import AppConfig, load_config
from coordinator.state_machine import StateMachine, SystemState, ErrorLevel
from vision.geometry_worker import compute_pick_pose, PickPose
from vision.yolo_worker import OBBDetection, YOLOResult


# ---------------------------------------------------------------------------
#  Mock Camera
# ---------------------------------------------------------------------------
class FakeCamera(CameraBase):
    """E2E 測試用 Mock Camera"""

    def __init__(self, depth_mode: str = "3D"):
        super().__init__(depth_mode=depth_mode)
        self._frame_count = 0

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def _capture(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        self._frame_count += 1
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        # 模擬物件在畫面中央
        rgb[200:280, 280:360] = 128

        depth = np.full((480, 640), 795.0, dtype=np.float32)
        if self._depth_mode == "2D":
            depth = None
        return rgb, depth


# ---------------------------------------------------------------------------
#  Mock OPC-UA Bridge
# ---------------------------------------------------------------------------
class FakeOPCUA:
    """E2E 測試用 Mock OPC-UA"""

    def __init__(self):
        self._connected = False
        self._commands: list[DB1Data] = []
        self._cmd = 0

    def connect(self):
        self._connected = True

    def disconnect(self):
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def write_pick_command(self, data: DB1Data, cycle_id: int = 0):
        self._commands.append(data)
        self._cmd = 1
        # 模擬 PLC 自動完成
        self._cmd = 2

    def read_cmd(self) -> int:
        return self._cmd

    def wait_for_done(self, timeout: float = 10.0, poll_interval: float = 0.01, cycle_id: int = 0) -> bool:
        if self._cmd == 2:
            self._cmd = 0
            return True
        return False

    def reset_cmd(self):
        self._cmd = 0


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------
class TestEndToEndPipeline:
    """端對端管線測試（Mock 環境）"""

    @pytest.fixture
    def config_3d(self):
        """載入 3D 設定"""
        cfg = load_config(site="A")
        return AppConfig(cfg)

    @pytest.fixture
    def config_2d(self):
        """載入 2D 設定"""
        cfg = load_config(site="B")
        return AppConfig(cfg)

    @pytest.fixture
    def identity_H(self):
        return np.eye(3)

    def test_full_3d_cycle(self, config_3d, identity_H):
        """完整 3D 管線: Camera → YOLO detection → Geometry → DB1Data"""
        # 1. Camera capture
        cam = FakeCamera(depth_mode="3D")
        cam.connect()
        rgb, depth = cam.get_frame()

        assert rgb.shape == (480, 640, 3)
        assert depth is not None

        # 2. 模擬 YOLO 偵測
        det = OBBDetection(
            label="classA",
            cx=320.0, cy=240.0,
            width=80.0, height=60.0,
            theta_deg=15.0,
            confidence=0.92,
            place_pos={"x": 400, "y": 100, "z": -50},
        )

        # 3. Geometry 計算
        pose = compute_pick_pose(
            cx=det.cx, cy=det.cy,
            theta_deg=det.theta_deg,
            label=det.label,
            place_pos=det.place_pos,
            H=identity_H,
            depth_or_none=depth,
            depth_mode="3D",
            offset_x=config_3d.offset_x,
            offset_y=config_3d.offset_y,
            camera_height_mm=config_3d.camera_height_mm,
            suction_length_mm=config_3d.suction_length_mm,
            safety_margin_mm=config_3d.safety_margin_mm,
        )

        assert isinstance(pose, PickPose)
        assert pose.rz == 15.0
        assert pose.label == "classA"

        # 4. 組裝 DB1Data
        db1 = DB1Data(
            pick_x=pose.pick_x,
            pick_y=pose.pick_y,
            pick_z=pose.pick_z,
            rx=pose.rx,
            ry=pose.ry,
            rz=pose.rz,
            place_x=det.place_pos["x"],
            place_y=det.place_pos["y"],
            place_z=det.place_pos["z"],
            cmd=1,
        )

        assert db1.cmd == 1
        assert db1.rz == 15.0

        # 5. 寫入 Mock PLC
        plc = FakeOPCUA()
        plc.connect()
        plc.write_pick_command(db1, cycle_id=1)
        done = plc.wait_for_done()
        assert done is True
        assert len(plc._commands) == 1

        cam.disconnect()

    def test_full_2d_cycle(self, config_2d, identity_H):
        """完整 2D 管線: Camera → YOLO detection → Geometry → DB1Data"""
        cam = FakeCamera(depth_mode="2D")
        cam.connect()
        rgb, depth = cam.get_frame()

        assert depth is None

        det = OBBDetection(
            label="classA",
            cx=320.0, cy=240.0,
            width=80.0, height=60.0,
            theta_deg=0.0,
            confidence=0.88,
            place_pos={"x": 400, "y": 200, "z": -50},
        )

        pose = compute_pick_pose(
            cx=det.cx, cy=det.cy,
            theta_deg=det.theta_deg,
            label=det.label,
            place_pos=det.place_pos,
            H=identity_H,
            depth_or_none=depth,
            depth_mode="2D",
            offset_x=0.0,
            offset_y=0.0,
            worktable_z_mm=config_2d.worktable_z_mm,
            suction_length_mm=config_2d.suction_length_mm,
            safety_margin_mm=config_2d.safety_margin_mm,
            thickness_map=config_2d.thickness_map,
        )

        assert isinstance(pose, PickPose)
        assert pose.rz == 0.0
        assert pose.label == "classA"

        db1 = DB1Data(
            pick_x=pose.pick_x,
            pick_y=pose.pick_y,
            pick_z=pose.pick_z,
            rz=pose.rz,
            place_x=det.place_pos["x"],
            place_y=det.place_pos["y"],
            place_z=det.place_pos["z"],
        )

        plc = FakeOPCUA()
        plc.connect()
        plc.write_pick_command(db1)
        done = plc.wait_for_done()
        assert done

        cam.disconnect()


class TestStateMachineFullLifecycle:
    """狀態機完整生命週期"""

    def test_normal_lifecycle(self):
        """正常: INIT → READY → RUNNING → READY → ... → SAFE_STOP"""
        sm = StateMachine()

        assert sm.state == SystemState.INITIALIZING

        sm.transition_to(SystemState.READY)
        assert sm.state == SystemState.READY

        # 多次 RUNNING ↔ READY 循環
        for _ in range(5):
            sm.transition_to(SystemState.RUNNING)
            assert sm.state == SystemState.RUNNING
            assert sm.is_operational()

            sm.transition_to(SystemState.READY)
            assert sm.state == SystemState.READY

        # 安全停機
        sm.enter_safe_stop("正常關機")
        assert sm.state == SystemState.SAFE_STOP
        assert not sm.is_operational()

    def test_error_recovery_lifecycle(self):
        """錯誤恢復: INIT → READY → RUNNING → ERROR → RUNNING → READY"""
        sm = StateMachine()
        sm.transition_to(SystemState.READY)
        sm.transition_to(SystemState.RUNNING)

        # 發生錯誤
        sm.enter_error(ErrorLevel.RETRY, "YOLO 推論逾時")
        assert sm.state == SystemState.ERROR
        assert sm.error_level == ErrorLevel.RETRY

        # 恢復
        sm.transition_to(SystemState.RUNNING)
        assert sm.state == SystemState.RUNNING
        assert sm.consecutive_errors == 0

        # 正常完成
        sm.transition_to(SystemState.READY)
        assert sm.state == SystemState.READY

    def test_critical_error_to_safe_stop(self):
        """嚴重錯誤: ERROR (CRITICAL) → SAFE_STOP"""
        sm = StateMachine()
        sm.transition_to(SystemState.READY)
        sm.transition_to(SystemState.RUNNING)

        sm.enter_error(ErrorLevel.CRITICAL, "相機斷線")
        assert sm.state == SystemState.ERROR
        assert sm.error_level == ErrorLevel.CRITICAL

        sm.enter_safe_stop("嚴重錯誤，進入安全停機")
        assert sm.state == SystemState.SAFE_STOP

    def test_restart_from_safe_stop(self):
        """重啟: SAFE_STOP → INITIALIZING → READY"""
        sm = StateMachine()
        sm.transition_to(SystemState.READY)
        sm.transition_to(SystemState.SAFE_STOP)

        # 只能回到 INITIALIZING
        sm.transition_to(SystemState.INITIALIZING)
        assert sm.state == SystemState.INITIALIZING

        sm.transition_to(SystemState.READY)
        assert sm.state == SystemState.READY


class TestMultiDetectionCycle:
    """多物件偵測 E2E"""

    def test_pick_highest_confidence(self):
        """多個偵測結果 → 取最高信心度"""
        detections = [
            OBBDetection("classA", 100, 200, 60, 40, 10.0, 0.72, {"x": 400, "y": 100, "z": -50}),
            OBBDetection("classB", 320, 240, 80, 60, 15.0, 0.95, {"x": 400, "y": 200, "z": -50}),
            OBBDetection("classA", 500, 300, 50, 50, 5.0, 0.81, {"x": 400, "y": 300, "z": -50}),
        ]

        best = max(detections, key=lambda d: d.confidence)
        assert best.label == "classB"
        assert best.confidence == 0.95
        assert best.theta_deg == 15.0

        # 計算 PickPose
        H = np.eye(3)
        depth = np.full((480, 640), 795.0, dtype=np.float32)

        pose = compute_pick_pose(
            cx=best.cx, cy=best.cy,
            theta_deg=best.theta_deg,
            label=best.label,
            place_pos=best.place_pos,
            H=H,
            depth_or_none=depth,
            depth_mode="3D",
            offset_x=5.0,
            offset_y=3.0,
            camera_height_mm=800.0,
            suction_length_mm=80.0,
            safety_margin_mm=5.0,
        )

        assert pose.label == "classB"
        assert pose.rz == 15.0
