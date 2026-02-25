# =============================================================================
#  KR6 Pick & Place — 相機工廠函式
#  依 CAMERA_TYPE 設定建立對應的 CameraBase 實例
# =============================================================================
from __future__ import annotations

import logging

from camera.base import CameraBase
from config import AppConfig

logger = logging.getLogger(__name__)


def create_camera(cfg: AppConfig) -> CameraBase:
    """
    工廠函式：依 CAMERA_TYPE 建立對應的相機實例。

    Args:
        cfg: 已驗證的 AppConfig 物件

    Returns:
        CameraBase 子類別實例

    Raises:
        ValueError: 不支援的 CAMERA_TYPE
    """
    camera_type = cfg.camera_type
    depth_mode = cfg.depth_mode

    logger.info(
        "建立相機: type=%s, depth_mode=%s", camera_type, depth_mode,
    )

    if camera_type == "d435":
        from camera.d435_stream import D435Camera

        return D435Camera(
            width=cfg.d435_width,
            height=cfg.d435_height,
            fps=cfg.d435_fps,
            depth_mode=depth_mode,
        )

    elif camera_type == "cognex":
        from camera.cognex_stream import CognexCamera

        return CognexCamera(
            ip=cfg.cognex_ip,
            port=cfg.cognex_port,
            cti_path=cfg.cognex_cti,
            depth_mode=depth_mode,
        )

    else:
        raise ValueError(f"不支援的 CAMERA_TYPE: '{camera_type}'")
