# =============================================================================
#  KR6 Pick & Place — 相機工廠函式
#  依 CAMERA_TYPE 設定從 CAMERA_REGISTRY 建立對應的 CameraBase 實例
#
#  擴充方式：新增相機時只需在驅動模組使用 @register_camera(...)
#  裝飾器即可，工廠函式會自動識別——無需修改此檔案。
# =============================================================================
from __future__ import annotations

import logging

from camera.base import CAMERA_REGISTRY, CameraBase
from config import AppConfig

logger = logging.getLogger(__name__)

# 確保內建相機插件已載入（觸發 @register_camera 裝飾器）
import camera  # noqa: F401


def create_camera(cfg: AppConfig) -> CameraBase:
    """
    工廠函式：依 CAMERA_TYPE 從 registry 建立對應的相機實例。

    流程：
      1. 查詢 CAMERA_REGISTRY[camera_type]
      2. 若有 factory_kwargs_builder → 呼叫建構參數
      3. 否則將 YAML 中 camera_type section 展開為 kwargs
      4. 注入 depth_mode 並建立實例

    Args:
        cfg: 已驗證的 AppConfig 物件

    Returns:
        CameraBase 子類別實例

    Raises:
        ValueError: 不支援的 CAMERA_TYPE（未註冊）
    """
    camera_type = cfg.camera_type
    depth_mode = cfg.depth_mode

    logger.info(
        "建立相機: type=%s, depth_mode=%s", camera_type, depth_mode,
    )

    # 查詢 registry
    info = CAMERA_REGISTRY.get(camera_type)
    if info is None:
        registered = list(CAMERA_REGISTRY.keys()) or ["(無)"]
        raise ValueError(
            f"不支援的 CAMERA_TYPE: '{camera_type}'。"
            f"已註冊: {registered}"
        )

    # 建構 kwargs
    if info.factory_kwargs_builder is not None:
        kwargs = info.factory_kwargs_builder(cfg)
    else:
        # 預設：從 YAML 的 camera_type section 取得所有參數
        kwargs = dict(cfg.raw.get(camera_type, {}))

    # 注入 depth_mode（所有 CameraBase 子類別都需要）
    kwargs["depth_mode"] = depth_mode

    logger.debug("相機初始化參數: %s", kwargs)

    return info.camera_class(**kwargs)

