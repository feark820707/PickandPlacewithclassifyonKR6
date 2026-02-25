# =============================================================================
#  KR6 Pick & Place — 相機抽象基底類別
#  統一介面 get_frame() → (rgb, depth_or_none)
# =============================================================================
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class CameraBase(ABC):
    """
    相機抽象基底類別。

    所有相機實作必須繼承此類別並實作：
      - connect()    → 初始化連線
      - disconnect() → 釋放資源
      - get_frame()  → (rgb_ndarray, depth_ndarray_or_None)

    規則：
      - DEPTH_MODE="3D"：depth 必須為有效的 ndarray
      - DEPTH_MODE="2D"：depth 為 None（即使硬體有深度能力也忽略）
    """

    def __init__(self, depth_mode: str = "3D"):
        """
        Args:
            depth_mode: "3D" → 回傳深度圖, "2D" → depth 固定回傳 None
        """
        self._depth_mode = depth_mode
        self._connected = False

    @property
    def depth_mode(self) -> str:
        return self._depth_mode

    @property
    def is_connected(self) -> bool:
        return self._connected

    @abstractmethod
    def connect(self) -> None:
        """初始化相機連線。連線失敗應拋出 ConnectionError。"""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """釋放相機資源。"""
        ...

    @abstractmethod
    def _capture(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        內部擷取方法，由子類別實作。

        Returns:
            (rgb, depth_or_none):
              - rgb:   np.ndarray, shape (H, W, 3), dtype uint8, BGR
              - depth: np.ndarray, shape (H, W), dtype float32 (mm)
                       或 None（若硬體無深度能力）
        """
        ...

    def get_frame(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        取得一幀影像。

        Returns:
            (rgb, depth_or_none):
              - 3D 模式：(rgb, depth)
              - 2D 模式：(rgb, None)

        Raises:
            RuntimeError: 未連線時呼叫
        """
        if not self._connected:
            raise RuntimeError("相機未連線，請先呼叫 connect()")

        rgb, depth = self._capture()

        # 2D 模式強制忽略深度
        if self._depth_mode == "2D":
            depth = None

        return rgb, depth

    # ---- Context Manager ----
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc_info):
        self.disconnect()
