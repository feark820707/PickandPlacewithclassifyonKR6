# =============================================================================
#  KR6 Pick & Place — 設定檔載入與驗證
#  分層載入：default.yaml → site_X.yaml → local.yaml（後者覆蓋前者）
# =============================================================================
from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
#  Custom Exceptions
# ---------------------------------------------------------------------------
class ConfigError(Exception):
    """設定檔驗證失敗時拋出"""
    pass


# ---------------------------------------------------------------------------
#  Deep Merge Utility
# ---------------------------------------------------------------------------
def deep_merge(base: dict, override: dict) -> dict:
    """
    遞迴合併兩個 dict，override 的值覆蓋 base。
    - 若兩者皆為 dict → 遞迴合併
    - 否則 → override 優先
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# ---------------------------------------------------------------------------
#  Config Loader
# ---------------------------------------------------------------------------
def load_config(site: str | None = None,
                config_dir: str | Path = "configs") -> dict:
    """
    分層載入 YAML 設定檔：
      1. default.yaml     — 所有預設值（必須存在）
      2. site_{site}.yaml — 現場覆蓋（可選）
      3. local.yaml       — 開發者本機覆蓋（可選，.gitignore）

    Args:
        site: 現場名稱，如 "A" → 載入 site_A.yaml
        config_dir: 設定檔目錄路徑

    Returns:
        合併後的設定 dict

    Raises:
        ConfigError: 設定檔驗證失敗
        FileNotFoundError: default.yaml 不存在
    """
    config_path = Path(config_dir)

    # 1. 載入 default（必須）
    default_file = config_path / "default.yaml"
    if not default_file.exists():
        raise FileNotFoundError(f"必要設定檔不存在: {default_file}")
    cfg = _load_yaml(default_file)

    # 2. 載入 site（可選）
    if site:
        site_file = config_path / f"site_{site}.yaml"
        if site_file.exists():
            site_cfg = _load_yaml(site_file)
            cfg = deep_merge(cfg, site_cfg)

    # 3. 載入 local（可選）
    local_file = config_path / "local.yaml"
    if local_file.exists():
        local_cfg = _load_yaml(local_file)
        cfg = deep_merge(cfg, local_cfg)

    # 4. 驗證
    validate_config(cfg)

    return cfg


def _load_yaml(path: Path) -> dict:
    """安全載入 YAML 檔案"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
#  Config Validation
# ---------------------------------------------------------------------------
VALID_DEPTH_MODES = {"3D", "2D"}


def _get_valid_camera_types() -> set[str]:
    """
    從 CAMERA_REGISTRY 動態取得合法 camera_type 集合。
    若 registry 尚未載入（例如單元測試直接呼叫 validate_config），
    則回退到內建預設值。
    """
    try:
        from camera.base import CAMERA_REGISTRY
        if CAMERA_REGISTRY:
            return set(CAMERA_REGISTRY.keys())
    except ImportError:
        pass
    # 回退預設值（保證核心測試可獨立運行）
    return {"d435", "cognex"}


def _get_camera_info(camera_type: str):
    """取得指定相機的 CameraInfo，若無法取得則回傳 None"""
    try:
        from camera.base import CAMERA_REGISTRY
        return CAMERA_REGISTRY.get(camera_type)
    except ImportError:
        return None


def validate_config(cfg: dict) -> None:
    """
    驗證設定檔合法性，不合法則拋出 ConfigError。

    檢查項目：
      1. camera_type 合法值（依 CAMERA_REGISTRY 動態判斷）
      2. depth_mode 合法值
      3. camera_type + depth_mode 相容性（依 registry metadata）
      4. 2D 模式下 thickness_map 必須涵蓋 place_map 所有類別
      5. 必要參數存在（通用 + 相機專屬，由 registry 提供）
      6. 偏移量零值警告
    """
    camera_type = cfg.get("camera_type", "")
    depth_mode = cfg.get("depth_mode", "")

    # 1. camera_type（動態查詢 registry）
    valid_camera_types = _get_valid_camera_types()
    if camera_type not in valid_camera_types:
        raise ConfigError(
            f"camera_type='{camera_type}' 不合法，"
            f"允許值: {valid_camera_types}"
        )

    # 2. depth_mode
    if depth_mode not in VALID_DEPTH_MODES:
        raise ConfigError(
            f"depth_mode='{depth_mode}' 不合法，"
            f"允許值: {VALID_DEPTH_MODES}"
        )

    # 3. camera_type + depth_mode 相容性（由 registry metadata 驅動）
    cam_info = _get_camera_info(camera_type)
    if cam_info is not None:
        if depth_mode not in cam_info.supported_depth_modes:
            raise ConfigError(
                f"{camera_type} 不支援 DEPTH_MODE='{depth_mode}'。"
                f"支援的模式: {set(cam_info.supported_depth_modes)}。"
            )
    else:
        # 回退硬編碼（registry 不可用時保持向下相容）
        if camera_type == "cognex" and depth_mode == "3D":
            raise ConfigError(
                "Cognex 無深度硬體，不支援 DEPTH_MODE='3D'。"
                "請改為 DEPTH_MODE='2D' 或切換 CAMERA_TYPE='d435'。"
            )

    # 4. 2D 模式 → thickness_map 必須涵蓋 place_map
    if depth_mode == "2D":
        thickness_map = cfg.get("depth_2d", {}).get("thickness_map", {})
        place_map = cfg.get("place_map", {})
        missing = [k for k in place_map if k not in thickness_map]
        if missing:
            raise ConfigError(
                f"DEPTH_MODE='2D' 但 thickness_map 缺少類別: {missing}"
            )

    # 5. 必要參數檢查
    _require_key(cfg, "plc.ip", "PLC IP 位址")
    _require_key(cfg, "mechanical.suction_length_mm", "吸盤長度")
    _require_key(cfg, "yolo.model", "YOLO 模型路徑")

    # 相機專屬必要參數（由 registry metadata 驅動）
    if cam_info is not None and cam_info.required_config_keys:
        for dotted_key in cam_info.required_config_keys:
            desc = f"{camera_type} 必要參數"
            _require_key(cfg, dotted_key, desc)
    else:
        # 回退硬編碼
        if camera_type == "d435":
            _require_key(cfg, "d435.width", "D435 影像寬度")
            _require_key(cfg, "d435.height", "D435 影像高度")
        elif camera_type == "cognex":
            _require_key(cfg, "cognex.ip", "Cognex 相機 IP")

    if depth_mode == "3D":
        _require_key(cfg, "depth_3d.camera_height_mm", "相機安裝高度")

    # 6. 偏移量零值警告
    mech = cfg.get("mechanical", {})
    if mech.get("offset_x", 0.0) == 0.0 and mech.get("offset_y", 0.0) == 0.0:
        warnings.warn(
            "OFFSET_X 與 OFFSET_Y 皆為 0，"
            "若鏡頭與吸盤不同軸請量測後填入 configs/。",
            UserWarning,
            stacklevel=2,
        )


def _require_key(cfg: dict, dotted_key: str, description: str) -> None:
    """檢查巢狀 key 是否存在且非 None"""
    keys = dotted_key.split(".")
    node: Any = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            raise ConfigError(f"必要參數缺失: {dotted_key} ({description})")
        node = node[k]
    if node is None:
        raise ConfigError(f"必要參數為 None: {dotted_key} ({description})")


# ---------------------------------------------------------------------------
#  Config Accessor — 方便取用的 helper
# ---------------------------------------------------------------------------
class AppConfig:
    """
    封裝已驗證的設定 dict，提供便捷屬性存取。
    用法:
        cfg = load_config(site="A")
        app = AppConfig(cfg)
        print(app.camera_type)   # "d435"
        print(app.plc_ip)        # "192.168.0.10"
    """

    def __init__(self, cfg: dict):
        self._cfg = cfg

    # ---- 核心雙開關 ----
    @property
    def camera_type(self) -> str:
        return self._cfg["camera_type"]

    @property
    def depth_mode(self) -> str:
        return self._cfg["depth_mode"]

    # ---- PLC ----
    @property
    def plc_ip(self) -> str:
        return self._cfg["plc"]["ip"]

    @property
    def opc_port(self) -> int:
        return self._cfg["plc"]["opc_port"]

    @property
    def opc_namespace(self) -> str:
        return self._cfg["plc"]["opc_namespace"]

    @property
    def plc_cmd_timeout(self) -> float:
        return self._cfg["plc"]["cmd_timeout_sec"]

    # ---- D435 ----
    @property
    def d435_width(self) -> int:
        return self._cfg.get("d435", {}).get("width", 1280)

    @property
    def d435_height(self) -> int:
        return self._cfg.get("d435", {}).get("height", 720)

    @property
    def d435_fps(self) -> int:
        return self._cfg.get("d435", {}).get("fps", 30)

    # ---- Cognex (In-Sight Native Mode) ----
    @property
    def cognex_ip(self) -> str:
        return self._cfg.get("cognex", {}).get("ip", "")

    @property
    def cognex_port(self) -> int:
        """Legacy 相容 — 回傳 telnet_port"""
        return self.cognex_telnet_port

    @property
    def cognex_telnet_port(self) -> int:
        return self._cfg.get("cognex", {}).get("telnet_port", 23)

    @property
    def cognex_ftp_port(self) -> int:
        return self._cfg.get("cognex", {}).get("ftp_port", 21)

    @property
    def cognex_ftp_user(self) -> str:
        return self._cfg.get("cognex", {}).get("ftp_user", "admin")

    @property
    def cognex_ftp_password(self) -> str:
        return self._cfg.get("cognex", {}).get("ftp_password", "")

    @property
    def cognex_cti(self) -> str:
        """Legacy 相容 — 已不再使用（IS8505P 用 Native Mode）"""
        return self._cfg.get("cognex", {}).get("cti_path", "")

    # ---- 通用相機參數存取 ----
    @property
    def camera_config(self) -> dict:
        """
        通用相機參數存取 — 回傳 YAML 中 camera_type 對應 section 的 dict。

        例如 camera_type="d435" → 回傳 cfg["d435"]
             camera_type="cognex" → 回傳 cfg["cognex"]
             camera_type="my_cam" → 回傳 cfg["my_cam"]

        若該 section 不存在則回傳空 dict。
        """
        return dict(self._cfg.get(self.camera_type, {}))

    # ---- 機械參數 ----
    @property
    def suction_length_mm(self) -> float:
        return self._cfg["mechanical"]["suction_length_mm"]

    @property
    def safety_margin_mm(self) -> float:
        return self._cfg["mechanical"]["safety_margin_mm"]

    @property
    def offset_x(self) -> float:
        return self._cfg["mechanical"]["offset_x"]

    @property
    def offset_y(self) -> float:
        return self._cfg["mechanical"]["offset_y"]

    # ---- 3D 模式 ----
    @property
    def camera_height_mm(self) -> float:
        return self._cfg.get("depth_3d", {}).get("camera_height_mm", 800.0)

    # ---- 2D 模式 ----
    @property
    def worktable_z_mm(self) -> float:
        return self._cfg.get("depth_2d", {}).get("worktable_z_mm", 0.0)

    @property
    def thickness_map(self) -> dict[str, float]:
        return self._cfg.get("depth_2d", {}).get("thickness_map", {})

    # ---- YOLO ----
    @property
    def yolo_model(self) -> str:
        return self._cfg["yolo"]["model"]

    @property
    def yolo_confidence(self) -> float:
        return self._cfg["yolo"]["confidence"]

    @property
    def yolo_device(self) -> str:
        return self._cfg.get("yolo", {}).get("device", "cuda:0")

    # ---- 放置位置 ----
    @property
    def place_map(self) -> dict:
        return self._cfg.get("place_map", {})

    # ---- CPU 核心分配 ----
    @property
    def core_geometry(self) -> list[int]:
        return self._cfg.get("cpu_affinity", {}).get("geometry_pool", [6, 7, 8, 9])

    @property
    def core_coordinator(self) -> list[int]:
        return self._cfg.get("cpu_affinity", {}).get("coordinator", [12])

    @property
    def core_opcua(self) -> list[int]:
        return self._cfg.get("cpu_affinity", {}).get("opcua", [13])

    @property
    def core_ui(self) -> list[int]:
        return self._cfg.get("cpu_affinity", {}).get("ui", [14])

    # ---- 日誌 ----
    @property
    def log_level(self) -> str:
        return self._cfg.get("logging", {}).get("level", "INFO")

    @property
    def log_dir(self) -> str:
        return self._cfg.get("logging", {}).get("dir", "logs")

    @property
    def log_retention_days(self) -> int:
        return self._cfg.get("logging", {}).get("retention_days", 30)

    # ---- 健康監控 ----
    @property
    def heartbeat_timeout(self) -> float:
        return self._cfg.get("health", {}).get("heartbeat_timeout_sec", 5.0)

    @property
    def queue_max_size(self) -> int:
        return self._cfg.get("health", {}).get("queue_max_size", 10)

    # ---- 重試 ----
    @property
    def retry_max(self) -> int:
        return self._cfg.get("retry", {}).get("max_retries", 3)

    @property
    def retry_delay(self) -> float:
        return self._cfg.get("retry", {}).get("delay_sec", 2.0)

    @property
    def retry_backoff(self) -> float:
        return self._cfg.get("retry", {}).get("backoff_factor", 1.5)

    # ---- 校準 ----
    @property
    def homography_path(self) -> str:
        return self._cfg.get("calibration", {}).get("homography_path", "assets/H.npy")

    # ---- Raw dict access ----
    @property
    def raw(self) -> dict:
        """取得原始 dict（唯讀複本）"""
        return copy.deepcopy(self._cfg)
