# =============================================================================
#  KR6 Pick & Place — Intel RealSense D435 相機驅動
#  USB 3.0 連線，支援 RGB + Depth 同步擷取
# =============================================================================
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

from camera.base import CameraBase, register_camera

if TYPE_CHECKING:
    from config import AppConfig

logger = logging.getLogger(__name__)


def _d435_kwargs(cfg: "AppConfig") -> dict:
    """從 AppConfig 建構 D435Camera 初始化參數"""
    sec = cfg.raw.get("d435", {})
    return {
        "width": sec.get("width", 1280),
        "height": sec.get("height", 720),
        "fps": sec.get("fps", 30),
    }


@register_camera(
    "d435",
    supported_depth_modes={"2D", "3D"},
    required_config_keys=("d435.width", "d435.height"),
    dependencies=("pyrealsense2",),
    description="Intel RealSense D435 (USB 3.0, RGB + Depth)",
    factory_kwargs_builder=_d435_kwargs,
)


class D435Camera(CameraBase):
    """
    Intel RealSense D435 相機。

    - USB 3.0 連線
    - RGB 1280×720 + Depth 同步
    - 3D 模式：回傳 aligned depth map (float32, mm)
    - 2D 模式：忽略深度，depth=None

    Usage:
        cam = D435Camera(width=1280, height=720, fps=30, depth_mode="3D")
        with cam:
            rgb, depth = cam.get_frame()
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        depth_mode: str = "3D",
    ):
        super().__init__(depth_mode=depth_mode)
        self._width = width
        self._height = height
        self._fps = fps
        self._pipeline = None
        self._align = None
        self._profile = None

    def connect(self) -> None:
        """初始化 D435 pipeline 並開始串流"""
        try:
            import pyrealsense2 as rs
        except ImportError:
            raise ImportError(
                "pyrealsense2 未安裝。請執行: pip install pyrealsense2"
            )

        logger.info(
            "D435 連線中... (解析度=%dx%d, FPS=%d, depth_mode=%s)",
            self._width, self._height, self._fps, self._depth_mode,
        )

        self._pipeline = rs.pipeline()
        config = rs.config()

        # 啟用 RGB 串流
        config.enable_stream(
            rs.stream.color,
            self._width, self._height,
            rs.format.bgr8,
            self._fps,
        )

        # 啟用 Depth 串流（即使 2D 模式也啟用，但 get_frame 中忽略）
        config.enable_stream(
            rs.stream.depth,
            self._width, self._height,
            rs.format.z16,
            self._fps,
        )

        # 開始串流
        self._profile = self._pipeline.start(config)

        # Align depth to color
        self._align = rs.align(rs.stream.color)

        # 取得深度比例
        depth_sensor = self._profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()  # 通常 0.001 (m)

        self._connected = True
        logger.info("D435 連線成功 (depth_scale=%.4f)", self._depth_scale)

    def disconnect(self) -> None:
        """停止串流並釋放資源"""
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception as e:
                logger.warning("D435 停止串流時發生錯誤: %s", e)
            self._pipeline = None

        self._align = None
        self._profile = None
        self._connected = False
        logger.info("D435 已斷線")

    def _capture(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        擷取一幀 RGB + Depth。

        Returns:
            (rgb, depth):
              - rgb:   shape (H, W, 3), dtype uint8, BGR
              - depth: shape (H, W), dtype float32, 單位 mm
        """
        import pyrealsense2 as rs

        # 等待下一幀
        frames = self._pipeline.wait_for_frames(timeout_ms=5000)

        # 對齊深度到 RGB
        aligned = self._align.process(frames)

        # RGB
        color_frame = aligned.get_color_frame()
        if not color_frame:
            raise RuntimeError("D435 無法取得 RGB 幀")
        rgb = np.asanyarray(color_frame.get_data())

        # Depth（轉為 mm float32）
        depth_frame = aligned.get_depth_frame()
        if depth_frame:
            depth = (
                np.asanyarray(depth_frame.get_data()).astype(np.float32)
                * self._depth_scale
                * 1000.0  # 轉為 mm
            )
        else:
            depth = None

        return rgb, depth

    @property
    def resolution(self) -> tuple[int, int]:
        """回傳 (width, height)"""
        return self._width, self._height
