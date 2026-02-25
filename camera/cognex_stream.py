# =============================================================================
#  KR6 Pick & Place — Cognex IS8505MP GigE Vision 相機驅動
#  Ethernet RJ45 連線，僅 RGB（無深度硬體）
# =============================================================================
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from camera.base import CameraBase

logger = logging.getLogger(__name__)


class CognexCamera(CameraBase):
    """
    Cognex IS8505MP-363-50 相機（GigE Vision）。

    - GigE Ethernet 連線
    - 僅 RGB 高解析度影像（內建光源/鏡頭）
    - 無深度硬體 → depth 永遠為 None
    - 僅支援 DEPTH_MODE="2D"

    需求：
      - harvesters (pip install harvesters)
      - Cognex GigE Vision Driver（提供 .cti GenTL producer）

    Usage:
        cam = CognexCamera(
            ip="192.168.0.50",
            cti_path="C:/Program Files/Cognex/.../CognexGigEVision.cti",
            depth_mode="2D",
        )
        with cam:
            rgb, depth = cam.get_frame()  # depth=None
    """

    def __init__(
        self,
        ip: str = "192.168.0.50",
        port: int = 3000,
        cti_path: str = "",
        depth_mode: str = "2D",
    ):
        if depth_mode == "3D":
            raise ValueError(
                "Cognex IS8505MP 無深度硬體，不支援 DEPTH_MODE='3D'"
            )
        super().__init__(depth_mode=depth_mode)
        self._ip = ip
        self._port = port
        self._cti_path = cti_path
        self._harvester = None
        self._acquirer = None

    def connect(self) -> None:
        """透過 GigE Vision 連線到 Cognex 相機"""
        try:
            from harvesters.core import Harvester
        except ImportError:
            raise ImportError(
                "harvesters 未安裝。請執行: pip install harvesters\n"
                "同時需安裝 Cognex GigE Vision Driver（.cti 檔案）"
            )

        logger.info(
            "Cognex 連線中... (IP=%s, CTI=%s)", self._ip, self._cti_path,
        )

        self._harvester = Harvester()

        # 載入 GenTL producer (.cti)
        if self._cti_path:
            self._harvester.add_file(self._cti_path)
        else:
            logger.warning("未指定 CTI 路徑，嘗試使用系統預設 GenTL producer")

        self._harvester.update()

        # 尋找目標相機
        device_list = self._harvester.device_info_list
        if not device_list:
            raise ConnectionError("找不到任何 GigE Vision 相機")

        # 嘗試依 IP 匹配
        target_idx = 0
        for idx, dev_info in enumerate(device_list):
            # GenTL 的 device_info 中可能包含 IP 資訊
            if hasattr(dev_info, "property_dict"):
                props = dev_info.property_dict
                if self._ip in str(props):
                    target_idx = idx
                    break

        logger.info(
            "找到 %d 台相機，選擇第 %d 台",
            len(device_list), target_idx,
        )

        # 建立影像擷取器
        self._acquirer = self._harvester.create(target_idx)
        self._acquirer.start()

        self._connected = True
        logger.info("Cognex 連線成功 (IP=%s)", self._ip)

    def disconnect(self) -> None:
        """釋放 GigE Vision 連線"""
        if self._acquirer is not None:
            try:
                self._acquirer.stop()
                self._acquirer.destroy()
            except Exception as e:
                logger.warning("Cognex acquirer 停止時發生錯誤: %s", e)
            self._acquirer = None

        if self._harvester is not None:
            try:
                self._harvester.reset()
            except Exception as e:
                logger.warning("Cognex harvester 重置時發生錯誤: %s", e)
            self._harvester = None

        self._connected = False
        logger.info("Cognex 已斷線")

    def _capture(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        擷取一幀 RGB 影像。

        Returns:
            (rgb, None):
              - rgb: np.ndarray, shape (H, W, 3), dtype uint8, BGR
              - depth: 永遠為 None（Cognex 無深度硬體）
        """
        with self._acquirer.fetch(timeout=5.0) as buffer:
            component = buffer.payload.components[0]

            # 取得影像尺寸與資料
            width = component.width
            height = component.height
            data = component.data

            # 根據像素格式轉換
            if component.num_components_per_pixel == 1:
                # Mono → 轉 BGR
                img = data.reshape(height, width)
                import cv2
                rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif component.num_components_per_pixel == 3:
                # RGB → BGR
                rgb = data.reshape(height, width, 3)
                if self._is_rgb_format(component):
                    import cv2
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            else:
                rgb = data.reshape(height, width, -1)[:, :, :3]

        return rgb, None  # Cognex 無深度

    @staticmethod
    def _is_rgb_format(component) -> bool:
        """判斷像素格式是否為 RGB（需轉 BGR）"""
        pixel_format = getattr(component, "pixel_format", "")
        return "RGB" in str(pixel_format).upper()

    @property
    def ip(self) -> str:
        return self._ip
