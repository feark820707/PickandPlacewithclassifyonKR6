# =============================================================================
#  KR6 Pick & Place — 相機抽象基底類別 + 插件註冊機制
#  統一介面 get_frame() → (rgb, depth_or_none)
#
#  擴充方式（第三方相機）：
#    1. 繼承 CameraBase
#    2. 使用 @register_camera(...) 裝飾器註冊
#    3. 在 configs/ 中新增對應 section
#    → 系統自動識別，無需修改 factory / config 驗證
# =============================================================================
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Camera Plugin Registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CameraInfo:
    """
    相機插件元資料 — 描述一款相機的能力與需求。

    Attributes:
        camera_class:         CameraBase 子類別
        supported_depth_modes: 支援的深度模式集合, e.g. {"2D"} or {"2D","3D"}
        required_config_keys:  YAML 中必須存在的 dotted-key 清單
        dependencies:          Python 套件名稱清單（dry-run 時檢查）
        description:           人類可讀描述
        factory_kwargs_builder: 可選的工廠參數建構函式
                               簽名: (AppConfig) → dict[str, Any]
                               若未提供，factory 會將整個 camera section 傳入
    """
    camera_class: Type["CameraBase"]
    supported_depth_modes: frozenset[str] = frozenset({"2D", "3D"})
    required_config_keys: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    description: str = ""
    factory_kwargs_builder: Optional[Callable[..., dict[str, Any]]] = None


# 全域註冊表
CAMERA_REGISTRY: dict[str, CameraInfo] = {}


def register_camera(
    name: str,
    *,
    supported_depth_modes: set[str] | frozenset[str] = frozenset({"2D", "3D"}),
    required_config_keys: tuple[str, ...] | list[str] = (),
    dependencies: tuple[str, ...] | list[str] = (),
    description: str = "",
    factory_kwargs_builder: Optional[Callable[..., dict[str, Any]]] = None,
) -> Callable[[Type["CameraBase"]], Type["CameraBase"]]:
    """
    相機插件註冊裝飾器。

    用法::

        @register_camera(
            "my_camera",
            supported_depth_modes={"2D"},
            required_config_keys=("my_camera.ip",),
            dependencies=("my_camera_sdk",),
            description="My Custom Camera",
        )
        class MyCam(CameraBase):
            ...

    Args:
        name: 在 YAML camera_type 中使用的唯一名稱
        supported_depth_modes: 此相機支援的 depth_mode 集合
        required_config_keys: YAML 中必須存在的 dotted-key
        dependencies: Python import 名稱（dry-run 檢查用）
        description: 人類可讀描述
        factory_kwargs_builder: 工廠參數建構函式 (AppConfig → dict)
    """
    def decorator(cls: Type["CameraBase"]) -> Type["CameraBase"]:
        if name in CAMERA_REGISTRY:
            logger.warning("相機 '%s' 已註冊，將被 %s 覆蓋", name, cls.__name__)
        CAMERA_REGISTRY[name] = CameraInfo(
            camera_class=cls,
            supported_depth_modes=frozenset(supported_depth_modes),
            required_config_keys=tuple(required_config_keys),
            dependencies=tuple(dependencies),
            description=description or cls.__doc__ or cls.__name__,
            factory_kwargs_builder=factory_kwargs_builder,
        )
        logger.debug("已註冊相機插件: %s → %s", name, cls.__name__)
        return cls
    return decorator


def get_registered_cameras() -> dict[str, CameraInfo]:
    """取得所有已註冊相機的快照（唯讀）"""
    return dict(CAMERA_REGISTRY)


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
