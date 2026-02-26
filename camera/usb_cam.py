# =============================================================================
#  KR6 Pick & Place — 通用 USB 攝影機驅動（cv2.VideoCapture）
#
#  支援任何 OpenCV 可識別的裝置：
#    - 一般 USB 網路攝影機 (index 0, 1, 2…)
#    - RTSP / HTTP 串流 URL
#    - 本地影片檔 (.mp4 / .avi 等)
#
#  Register name: "usb"
#
#  Config 範例（site_B.yaml）：
#    camera_type: usb
#    usb:
#      index: 0          # 裝置編號（預設 0）
#      width: 1280       # 解析度（選填，0 = 不設定）
#      height: 720
#      fps: 30           # 幀率（選填，0 = 不設定）
#
#  --source 用法：
#    --source usb        → index=0
#    --source usb:1      → index=1
#    --source usb:2      → index=2
# =============================================================================
from __future__ import annotations

import logging
from typing import Optional

import sys

import cv2
import numpy as np

from camera.base import CameraBase, register_camera

logger = logging.getLogger(__name__)


@register_camera(
    "usb",
    supported_depth_modes={"2D"},
    description="通用 USB 攝影機 / RTSP 串流（cv2.VideoCapture）",
)
class UsbCamera(CameraBase):
    """
    通用 USB 攝影機驅動。

    Args:
        index:      裝置編號（int）或串流 URL（str）。預設 0。
        width:      設定擷取解析度寬（0 = 使用裝置預設）。
        height:     設定擷取解析度高（0 = 使用裝置預設）。
        fps:        設定幀率（0 = 使用裝置預設）。
        depth_mode: 固定 "2D"（USB 攝影機無深度）。
    """

    def __init__(
        self,
        index: int | str = 0,
        width: int = 0,
        height: int = 0,
        fps: int = 0,
        depth_mode: str = "2D",
    ):
        super().__init__(depth_mode="2D")
        # index 可能是整數字串 "0", "1"，或 URL
        if isinstance(index, str) and index.isdigit():
            index = int(index)
        self._index  = index
        self._width  = width
        self._height = height
        self._fps    = fps
        self._cap: Optional[cv2.VideoCapture] = None

    # ------------------------------------------------------------------
    #  CameraBase 介面
    # ------------------------------------------------------------------
    def connect(self) -> None:
        # Windows 上使用 DSHOW 後端，裝置編號與系統一致
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(self._index, backend)
        if not self._cap.isOpened():
            raise ConnectionError(
                f"無法開啟 USB 攝影機 (index={self._index})。"
                "請確認裝置已連接，或嘗試其他 index（--source usb:1）。"
            )

        # 套用解析度 / 幀率設定
        if self._width > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        if self._height > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if self._fps > 0:
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        # 讀回實際值
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "USB 攝影機已開啟 index=%s  解析度=%dx%d  fps=%.1f",
            self._index, w, h, f,
        )
        self._connected = True

    def disconnect(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._connected = False
        logger.info("USB 攝影機已關閉")

    def _capture(self) -> tuple[np.ndarray, None]:
        if self._cap is None:
            raise RuntimeError("VideoCapture 未初始化")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError(
                f"USB 攝影機讀取失敗 (index={self._index})。"
                "裝置可能已拔除。"
            )
        return frame, None

    # ------------------------------------------------------------------
    #  屬性
    # ------------------------------------------------------------------
    @property
    def resolution(self) -> tuple[int, int]:
        """回傳 (width, height)，未連線時回傳 (0, 0)。"""
        if self._cap is None:
            return (0, 0)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)
