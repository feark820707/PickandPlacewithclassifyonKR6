# =============================================================================
#  Unit Tests — 幾何計算（Z 補償、偏移補償、像素轉換）
# =============================================================================
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vision.geometry_worker import (
    calc_z_2d,
    calc_z_3d,
    compensate_offset,
    compute_pick_pose,
    extract_surface_depth,
    pixel_to_robot,
    PickPose,
)


# ---------------------------------------------------------------------------
#  Tests: calc_z_3d
# ---------------------------------------------------------------------------
class TestCalcZ3D:
    def test_standard_case(self):
        """標準情況：相機 800mm, 表面 795mm, 吸盤 80mm, 安全 5mm"""
        z = calc_z_3d(
            surface_depth_mm=795,
            camera_height_mm=800,
            suction_length_mm=80,
            safety_margin_mm=5,
        )
        assert z == pytest.approx(-80.0)

    def test_thick_object(self):
        """厚物件：表面 780mm"""
        z = calc_z_3d(
            surface_depth_mm=780,
            camera_height_mm=800,
            suction_length_mm=80,
            safety_margin_mm=5,
        )
        assert z == pytest.approx(-65.0)

    def test_thin_object(self):
        """薄物件：表面 798mm"""
        z = calc_z_3d(
            surface_depth_mm=798,
            camera_height_mm=800,
            suction_length_mm=80,
            safety_margin_mm=5,
        )
        assert z == pytest.approx(-83.0)


# ---------------------------------------------------------------------------
#  Tests: calc_z_2d
# ---------------------------------------------------------------------------
class TestCalcZ2D:
    def test_classA(self):
        """classA 厚 5mm"""
        z = calc_z_2d(
            label="classA",
            thickness_map={"classA": 5.0, "classB": 10.0, "classC": 20.0},
            worktable_z_mm=0.0,
            suction_length_mm=80.0,
            safety_margin_mm=5.0,
        )
        # 0 - 80 - 5 + 5 = -80
        assert z == pytest.approx(-80.0)

    def test_classC(self):
        """classC 厚 20mm"""
        z = calc_z_2d(
            label="classC",
            thickness_map={"classA": 5.0, "classB": 10.0, "classC": 20.0},
            worktable_z_mm=0.0,
            suction_length_mm=80.0,
            safety_margin_mm=5.0,
        )
        # 0 - 80 - 5 + 20 = -65
        assert z == pytest.approx(-65.0)

    def test_unknown_label(self):
        """未知類別 → 厚度 0"""
        z = calc_z_2d(
            label="unknown",
            thickness_map={"classA": 5.0},
            worktable_z_mm=0.0,
            suction_length_mm=80.0,
            safety_margin_mm=5.0,
        )
        # 0 - 80 - 5 + 0 = -85
        assert z == pytest.approx(-85.0)


# ---------------------------------------------------------------------------
#  Tests: compensate_offset
# ---------------------------------------------------------------------------
class TestCompensateOffset:
    def test_zero_angle_zero_offset(self):
        """θ=0, offset=(0,0) → 無補償"""
        x, y = compensate_offset(100, 200, theta_deg=0, offset_x=0, offset_y=0)
        assert x == pytest.approx(100.0)
        assert y == pytest.approx(200.0)

    def test_zero_angle_with_offset(self):
        """θ=0 時只做平移"""
        x, y = compensate_offset(100, 200, theta_deg=0, offset_x=5, offset_y=3)
        assert x == pytest.approx(105.0)
        assert y == pytest.approx(203.0)

    def test_90_degrees(self):
        """θ=90° 時偏移向量旋轉 90°"""
        x, y = compensate_offset(
            100, 200, theta_deg=90, offset_x=5.0, offset_y=3.0,
        )
        # comp_x = 100 + 5*cos(90) - 3*sin(90) = 100 + 0 - 3 = 97
        # comp_y = 200 + 5*sin(90) + 3*cos(90) = 200 + 5 + 0 = 205
        assert x == pytest.approx(97.0, abs=0.01)
        assert y == pytest.approx(205.0, abs=0.01)

    def test_180_degrees(self):
        """θ=180° 偏移完全反轉"""
        x, y = compensate_offset(
            100, 200, theta_deg=180, offset_x=5.0, offset_y=3.0,
        )
        # comp_x = 100 + 5*cos(180) - 3*sin(180) = 100 - 5 - 0 = 95
        # comp_y = 200 + 5*sin(180) + 3*cos(180) = 200 + 0 - 3 = 197
        assert x == pytest.approx(95.0, abs=0.01)
        assert y == pytest.approx(197.0, abs=0.01)

    def test_45_degrees(self):
        """θ=45° 驗證"""
        x, y = compensate_offset(
            0, 0, theta_deg=45, offset_x=10.0, offset_y=0.0,
        )
        # comp_x = 0 + 10*cos(45) - 0 = 7.071
        # comp_y = 0 + 10*sin(45) + 0 = 7.071
        assert x == pytest.approx(10 * math.cos(math.radians(45)), abs=0.01)
        assert y == pytest.approx(10 * math.sin(math.radians(45)), abs=0.01)

    def test_negative_angle(self):
        """負角度"""
        x, y = compensate_offset(
            0, 0, theta_deg=-90, offset_x=5.0, offset_y=3.0,
        )
        # comp_x = 0 + 5*cos(-90) - 3*sin(-90) = 0 + 0 + 3 = 3
        # comp_y = 0 + 5*sin(-90) + 3*cos(-90) = 0 - 5 + 0 = -5
        assert x == pytest.approx(3.0, abs=0.01)
        assert y == pytest.approx(-5.0, abs=0.01)


# ---------------------------------------------------------------------------
#  Tests: pixel_to_robot
# ---------------------------------------------------------------------------
class TestPixelToRobot:
    def test_identity_homography(self):
        """單位矩陣 → 像素 = 機器人座標"""
        H = np.eye(3)
        x, y = pixel_to_robot(100, 200, H)
        assert x == pytest.approx(100.0)
        assert y == pytest.approx(200.0)

    def test_scaling_homography(self):
        """縮放矩陣"""
        H = np.array([
            [2.0, 0, 0],
            [0, 3.0, 0],
            [0, 0, 1],
        ])
        x, y = pixel_to_robot(10, 20, H)
        assert x == pytest.approx(20.0)
        assert y == pytest.approx(60.0)

    def test_translation_homography(self):
        """平移矩陣"""
        H = np.array([
            [1, 0, 50],
            [0, 1, -30],
            [0, 0, 1],
        ], dtype=np.float64)
        x, y = pixel_to_robot(100, 200, H)
        assert x == pytest.approx(150.0)
        assert y == pytest.approx(170.0)


# ---------------------------------------------------------------------------
#  Tests: extract_surface_depth
# ---------------------------------------------------------------------------
class TestExtractSurfaceDepth:
    def test_uniform_depth(self):
        """均勻深度圖"""
        depth = np.full((480, 640), 795.0, dtype=np.float32)
        z = extract_surface_depth(depth, cx=320, cy=240, roi_radius=10)
        assert z == pytest.approx(795.0)

    def test_handles_zeros(self):
        """含零值的深度圖（過濾無效值）"""
        depth = np.full((480, 640), 795.0, dtype=np.float32)
        depth[235:240, 315:320] = 0  # 部分無效
        z = extract_surface_depth(depth, cx=320, cy=240, roi_radius=10)
        assert z == pytest.approx(795.0)

    def test_insufficient_valid_points(self):
        """有效點不足 → 拋出例外"""
        depth = np.zeros((480, 640), dtype=np.float32)
        with pytest.raises(ValueError, match="深度值不足"):
            extract_surface_depth(depth, cx=320, cy=240, roi_radius=5)

    def test_edge_roi(self):
        """中心點在影像邊緣"""
        depth = np.full((480, 640), 800.0, dtype=np.float32)
        z = extract_surface_depth(depth, cx=5, cy=5, roi_radius=10)
        assert z == pytest.approx(800.0)


# ---------------------------------------------------------------------------
#  Tests: compute_pick_pose（整合測試）
# ---------------------------------------------------------------------------
class TestComputePickPose:
    @pytest.fixture
    def identity_H(self):
        return np.eye(3)

    def test_3d_mode(self, identity_H):
        """3D 模式完整計算"""
        depth = np.full((480, 640), 795.0, dtype=np.float32)
        pose = compute_pick_pose(
            cx=100, cy=200, theta_deg=0,
            label="classA",
            place_pos={"x": 400, "y": 100, "z": -50},
            H=identity_H,
            depth_or_none=depth,
            depth_mode="3D",
            offset_x=0, offset_y=0,
            camera_height_mm=800,
            suction_length_mm=80,
            safety_margin_mm=5,
        )
        assert isinstance(pose, PickPose)
        assert pose.pick_x == pytest.approx(100.0)
        assert pose.pick_y == pytest.approx(200.0)
        assert pose.pick_z == pytest.approx(-80.0)
        assert pose.rz == 0.0
        assert pose.rx == 0.0
        assert pose.ry == 0.0

    def test_2d_mode(self, identity_H):
        """2D 模式完整計算"""
        pose = compute_pick_pose(
            cx=100, cy=200, theta_deg=15.0,
            label="classC",
            place_pos={"x": 400, "y": 300, "z": -50},
            H=identity_H,
            depth_or_none=None,
            depth_mode="2D",
            offset_x=5.0, offset_y=3.0,
            worktable_z_mm=0,
            suction_length_mm=80,
            safety_margin_mm=5,
            thickness_map={"classC": 20.0},
        )
        assert pose.pick_z == pytest.approx(-65.0)
        assert pose.rz == 15.0
        assert pose.label == "classC"
        # 有偏移補償
        assert pose.pick_x != pytest.approx(100.0)
        assert pose.pick_y != pytest.approx(200.0)

    def test_3d_no_depth_raises(self, identity_H):
        """3D 模式無深度 → 報錯"""
        with pytest.raises(ValueError, match="3D"):
            compute_pick_pose(
                cx=100, cy=200, theta_deg=0,
                label="classA",
                place_pos={},
                H=identity_H,
                depth_or_none=None,
                depth_mode="3D",
                offset_x=0, offset_y=0,
            )
