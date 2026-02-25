# =============================================================================
#  Integration Tests — Mock 相機 → 幾何計算管線
#  使用 Mock 相機驗證從擷取到 PickPose 的完整流程
# =============================================================================
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from camera.base import CameraBase
from vision.geometry_worker import (
    compute_pick_pose,
    PickPose,
    GeometryPool,
)


# ---------------------------------------------------------------------------
#  Mock Camera
# ---------------------------------------------------------------------------
class MockCamera(CameraBase):
    """測試用 Mock 相機，回傳固定影像"""

    def __init__(
        self,
        depth_mode: str = "3D",
        width: int = 640,
        height: int = 480,
        depth_value: float = 795.0,
    ):
        super().__init__(depth_mode=depth_mode)
        self._width = width
        self._height = height
        self._depth_value = depth_value

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def _capture(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        rgb = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        # 畫一個白色矩形當作物件
        rgb[200:280, 280:360] = 255

        depth = np.full(
            (self._height, self._width), self._depth_value, dtype=np.float32,
        )
        return rgb, depth


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------
class TestMockCameraPipeline:
    @pytest.fixture
    def mock_cam_3d(self):
        cam = MockCamera(depth_mode="3D", depth_value=795.0)
        cam.connect()
        yield cam
        cam.disconnect()

    @pytest.fixture
    def mock_cam_2d(self):
        cam = MockCamera(depth_mode="2D")
        cam.connect()
        yield cam
        cam.disconnect()

    @pytest.fixture
    def identity_H(self):
        return np.eye(3)

    def test_mock_camera_3d_returns_depth(self, mock_cam_3d):
        """3D Mock 相機回傳 RGB + Depth"""
        rgb, depth = mock_cam_3d.get_frame()
        assert rgb.shape == (480, 640, 3)
        assert depth is not None
        assert depth.shape == (480, 640)

    def test_mock_camera_2d_no_depth(self, mock_cam_2d):
        """2D Mock 相機回傳 RGB + None"""
        rgb, depth = mock_cam_2d.get_frame()
        assert rgb.shape == (480, 640, 3)
        assert depth is None

    def test_full_pipeline_3d(self, mock_cam_3d, identity_H):
        """完整 3D 管線: 相機 → 幾何計算 → PickPose"""
        rgb, depth = mock_cam_3d.get_frame()

        pose = compute_pick_pose(
            cx=320, cy=240,
            theta_deg=15.0,
            label="classA",
            place_pos={"x": 400, "y": 100, "z": -50},
            H=identity_H,
            depth_or_none=depth,
            depth_mode="3D",
            offset_x=5.0,
            offset_y=3.0,
            camera_height_mm=800.0,
            suction_length_mm=80.0,
            safety_margin_mm=5.0,
        )

        assert isinstance(pose, PickPose)
        assert pose.rz == 15.0
        assert pose.rx == 0.0
        assert pose.ry == 0.0
        assert pose.pick_z == pytest.approx(-80.0)
        assert pose.label == "classA"
        # 有偏移補償 → pick_x ≠ 320
        assert pose.pick_x != pytest.approx(320.0)

    def test_full_pipeline_2d(self, mock_cam_2d, identity_H):
        """完整 2D 管線: 相機 → 幾何計算 → PickPose"""
        rgb, depth = mock_cam_2d.get_frame()

        pose = compute_pick_pose(
            cx=320, cy=240,
            theta_deg=0.0,
            label="classB",
            place_pos={"x": 400, "y": 200, "z": -50},
            H=identity_H,
            depth_or_none=depth,  # None
            depth_mode="2D",
            offset_x=0.0,
            offset_y=0.0,
            worktable_z_mm=0.0,
            suction_length_mm=80.0,
            safety_margin_mm=5.0,
            thickness_map={"classB": 10.0},
        )

        assert isinstance(pose, PickPose)
        assert pose.rz == 0.0
        assert pose.pick_z == pytest.approx(-75.0)  # 0 - 80 - 5 + 10
        assert pose.pick_x == pytest.approx(320.0)   # 無偏移
        assert pose.pick_y == pytest.approx(240.0)

    def test_camera_context_manager(self):
        """Context Manager 測試"""
        with MockCamera(depth_mode="2D") as cam:
            assert cam.is_connected
            rgb, depth = cam.get_frame()
            assert rgb is not None
        assert not cam.is_connected

    def test_camera_not_connected_raises(self):
        """未連線時取幀 → RuntimeError"""
        cam = MockCamera()
        with pytest.raises(RuntimeError, match="未連線"):
            cam.get_frame()


class TestGeometryPool:
    def test_pool_submit(self):
        """GeometryPool 提交任務"""
        H = np.eye(3)
        depth = np.full((480, 640), 795.0, dtype=np.float32)

        with GeometryPool(max_workers=2) as pool:
            future = pool.submit(
                compute_pick_pose,
                cx=320, cy=240, theta_deg=0.0,
                label="classA",
                place_pos={"x": 400, "y": 100, "z": -50},
                H=H,
                depth_or_none=depth,
                depth_mode="3D",
                offset_x=0.0, offset_y=0.0,
                camera_height_mm=800.0,
                suction_length_mm=80.0,
                safety_margin_mm=5.0,
            )
            pose = future.result(timeout=5.0)
            assert isinstance(pose, PickPose)

    def test_pool_not_started_raises(self):
        """未啟動 Pool 就提交 → RuntimeError"""
        pool = GeometryPool()
        with pytest.raises(RuntimeError, match="未啟動"):
            pool.submit(lambda: None)
