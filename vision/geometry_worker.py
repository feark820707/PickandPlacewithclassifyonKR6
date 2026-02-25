# =============================================================================
#  KR6 Pick & Place — 幾何計算 Worker
#  CPU ProcessPool：OBB → 抓取點 + Rz + XY 偏移補償 + Z(3D/2D)
# =============================================================================
from __future__ import annotations

import logging
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Data Classes
# ---------------------------------------------------------------------------
@dataclass
class PickPose:
    """完整抓取姿態"""
    pick_x: float       # 機器人 X (mm)，含偏移補償
    pick_y: float       # 機器人 Y (mm)，含偏移補償
    pick_z: float       # 機器人 Z (mm)，3D 動態或 2D 查表
    rx: float = 0.0     # 固定 0（吸盤朝下）
    ry: float = 0.0     # 固定 0（吸盤朝下）
    rz: float = 0.0     # 物件旋轉角 θ（度）
    label: str = ""
    place_pos: dict | None = None


# ---------------------------------------------------------------------------
#  Z 軸計算
# ---------------------------------------------------------------------------
def calc_z_3d(
    surface_depth_mm: float,
    camera_height_mm: float,
    suction_length_mm: float,
    safety_margin_mm: float,
) -> float:
    """
    3D 模式 Z 軸計算：即時深度量測。

    公式：robot_z = camera_height - surface_depth - suction_length - safety_margin

    Args:
        surface_depth_mm:  物件表面深度（mm，相機到物件表面的距離）
        camera_height_mm:  相機安裝高度（mm）
        suction_length_mm: 吸盤長度（mm）
        safety_margin_mm:  安全餘量（mm）

    Returns:
        robot_z (mm)
    """
    return camera_height_mm - surface_depth_mm - suction_length_mm - safety_margin_mm


def calc_z_2d(
    label: str,
    thickness_map: dict[str, float],
    worktable_z_mm: float,
    suction_length_mm: float,
    safety_margin_mm: float,
) -> float:
    """
    2D 模式 Z 軸計算：查表固定 Z。

    公式：robot_z = worktable_z - suction_length - safety_margin + thickness

    Args:
        label:             物件類別
        thickness_map:     各類別厚度 dict
        worktable_z_mm:    工作台面 Z 值（機器人座標系）
        suction_length_mm: 吸盤長度（mm）
        safety_margin_mm:  安全餘量（mm）

    Returns:
        robot_z (mm)
    """
    thickness = thickness_map.get(label, 0.0)
    return worktable_z_mm - suction_length_mm - safety_margin_mm + thickness


# ---------------------------------------------------------------------------
#  XY 偏移補償
# ---------------------------------------------------------------------------
def compensate_offset(
    pick_x_raw: float,
    pick_y_raw: float,
    theta_deg: float,
    offset_x: float,
    offset_y: float,
) -> tuple[float, float]:
    """
    鏡頭−吸盤不同軸 XY 偏移補償（旋轉矩陣）。

    當機器人末端旋轉 Rz=θ 後，偏移向量也跟著旋轉：
        comp_x = raw_x + OFFSET_X·cos(θ) - OFFSET_Y·sin(θ)
        comp_y = raw_y + OFFSET_X·sin(θ) + OFFSET_Y·cos(θ)

    特殊情況：
      - θ=0, offset=(0,0) → 無補償
      - θ=0, offset=(5,3) → 平移
      - θ=90°, offset=(5,3) → comp_x = raw_x - 3, comp_y = raw_y + 5

    Args:
        pick_x_raw: H 矩陣轉換後的原始 X
        pick_y_raw: H 矩陣轉換後的原始 Y
        theta_deg:  物件旋轉角度（度）= Rz
        offset_x:   鏡頭→吸盤 X 偏移 (mm)
        offset_y:   鏡頭→吸盤 Y 偏移 (mm)

    Returns:
        (compensated_x, compensated_y)
    """
    if offset_x == 0.0 and offset_y == 0.0:
        return pick_x_raw, pick_y_raw

    theta_rad = math.radians(theta_deg)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)

    comp_x = pick_x_raw + offset_x * cos_t - offset_y * sin_t
    comp_y = pick_y_raw + offset_x * sin_t + offset_y * cos_t

    return comp_x, comp_y


# ---------------------------------------------------------------------------
#  表面深度擷取
# ---------------------------------------------------------------------------
def extract_surface_depth(
    depth_map: np.ndarray,
    cx: float,
    cy: float,
    roi_radius: int = 10,
) -> float:
    """
    從深度圖中擷取物件表面深度值。

    取中心點周圍 ROI 區域的中位數作為表面深度。

    Args:
        depth_map: shape (H, W), float32, 單位 mm
        cx, cy: 物件中心像素座標
        roi_radius: ROI 半徑（像素）

    Returns:
        表面深度 (mm)

    Raises:
        ValueError: 深度值無效
    """
    h, w = depth_map.shape[:2]
    ix, iy = int(round(cx)), int(round(cy))

    # 限制 ROI 在影像範圍內
    x1 = max(0, ix - roi_radius)
    x2 = min(w, ix + roi_radius)
    y1 = max(0, iy - roi_radius)
    y2 = min(h, iy + roi_radius)

    roi = depth_map[y1:y2, x1:x2]
    valid = roi[(roi > 0) & (~np.isnan(roi))]

    if len(valid) < 5:
        raise ValueError(
            f"深度值不足: cx={cx}, cy={cy}, valid_points={len(valid)}"
        )

    return float(np.median(valid))


# ---------------------------------------------------------------------------
#  像素 → 機器人座標
# ---------------------------------------------------------------------------
def pixel_to_robot(
    px_x: float,
    px_y: float,
    H: np.ndarray,
) -> tuple[float, float]:
    """像素座標 → 機器人基座標"""
    px = np.array([px_x, px_y, 1.0])
    rob = H @ px
    return float(rob[0] / rob[2]), float(rob[1] / rob[2])


# ---------------------------------------------------------------------------
#  完整計算入口
# ---------------------------------------------------------------------------
def compute_pick_pose(
    cx: float,
    cy: float,
    theta_deg: float,
    label: str,
    place_pos: dict,
    H: np.ndarray,
    depth_or_none: Optional[np.ndarray],
    depth_mode: str,
    offset_x: float,
    offset_y: float,
    # 3D params
    camera_height_mm: float = 800.0,
    suction_length_mm: float = 80.0,
    safety_margin_mm: float = 5.0,
    # 2D params
    worktable_z_mm: float = 0.0,
    thickness_map: dict[str, float] | None = None,
) -> PickPose:
    """
    完整計算抓取姿態：XY(含偏移補償) + Z(3D/2D) + Rz(=θ)。

    Args:
        cx, cy:         OBB 中心像素座標
        theta_deg:      物件旋轉角（度）
        label:          物件類別
        place_pos:      放置位置 dict {x, y, z}
        H:              Homography 矩陣 (3×3)
        depth_or_none:  深度圖或 None
        depth_mode:     "3D" | "2D"
        offset_x, offset_y: 鏡頭−吸盤偏移 (mm)
        camera_height_mm:   相機高度 (3D 用)
        suction_length_mm:  吸盤長度
        safety_margin_mm:   安全餘量
        worktable_z_mm:     工作台 Z (2D 用)
        thickness_map:      厚度查表 (2D 用)

    Returns:
        PickPose
    """
    # 1. 像素 → 機器人座標
    pick_x_raw, pick_y_raw = pixel_to_robot(cx, cy, H)

    # 2. Z 軸
    if depth_mode == "3D":
        if depth_or_none is None:
            raise ValueError("3D 模式需要深度資料，但收到 None")
        surface_z = extract_surface_depth(depth_or_none, cx, cy)
        pick_z = calc_z_3d(
            surface_z, camera_height_mm, suction_length_mm, safety_margin_mm,
        )
    else:
        pick_z = calc_z_2d(
            label, thickness_map or {},
            worktable_z_mm, suction_length_mm, safety_margin_mm,
        )

    # 3. Rz = θ
    rz = theta_deg

    # 4. 偏移補償
    pick_x, pick_y = compensate_offset(
        pick_x_raw, pick_y_raw, theta_deg, offset_x, offset_y,
    )

    return PickPose(
        pick_x=pick_x,
        pick_y=pick_y,
        pick_z=pick_z,
        rx=0.0,
        ry=0.0,
        rz=rz,
        label=label,
        place_pos=place_pos,
    )


# ---------------------------------------------------------------------------
#  Geometry Pool（ProcessPoolExecutor 包裝器）
# ---------------------------------------------------------------------------
class GeometryPool:
    """
    幾何計算 Process Pool。

    使用 ProcessPoolExecutor 在獨立 CPU 核心上並行計算。

    用法:
        pool = GeometryPool(max_workers=4)
        pool.start()

        future = pool.submit(compute_pick_pose, **params)
        pose = future.result()

        pool.shutdown()
    """

    def __init__(self, max_workers: int = 4):
        self._max_workers = max_workers
        self._executor: Optional[ProcessPoolExecutor] = None

    def start(self) -> None:
        """啟動 Process Pool"""
        self._executor = ProcessPoolExecutor(max_workers=self._max_workers)
        logger.info("GeometryPool 已啟動 (workers=%d)", self._max_workers)

    def submit(self, fn, *args, **kwargs):
        """提交計算任務"""
        if self._executor is None:
            raise RuntimeError("GeometryPool 未啟動，請先呼叫 start()")
        return self._executor.submit(fn, *args, **kwargs)

    def shutdown(self, wait: bool = True) -> None:
        """關閉 Process Pool"""
        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None
            logger.info("GeometryPool 已關閉")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc_info):
        self.shutdown()
