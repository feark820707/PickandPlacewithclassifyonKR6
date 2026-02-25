# =============================================================================
#  Unit Tests — config.py 設定檔載入與驗證
# =============================================================================
import os
import sys
import tempfile
import warnings
from pathlib import Path

import pytest
import yaml

# 確保能 import 專案模組
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    AppConfig,
    ConfigError,
    deep_merge,
    load_config,
    validate_config,
)


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_config_dir(tmp_path):
    """建立臨時設定檔目錄，含完整 default.yaml"""
    default_cfg = {
        "camera_type": "d435",
        "depth_mode": "3D",
        "plc": {
            "ip": "192.168.0.10",
            "opc_port": 4840,
            "opc_namespace": "urn:siemens:s71500",
            "cmd_timeout_sec": 10.0,
        },
        "d435": {"width": 1280, "height": 720, "fps": 30},
        "cognex": {
            "ip": "192.168.0.50",
            "port": 3000,
            "cti_path": "C:\\CognexGigEVision.cti",
        },
        "mechanical": {
            "suction_length_mm": 80.0,
            "safety_margin_mm": 5.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
        },
        "depth_3d": {"camera_height_mm": 800.0},
        "depth_2d": {
            "worktable_z_mm": 0.0,
            "thickness_map": {
                "classA": 5.0,
                "classB": 10.0,
                "classC": 20.0,
            },
        },
        "yolo": {
            "model": "assets/best_obb.pt",
            "confidence": 0.5,
            "device": "cuda:0",
        },
        "place_map": {
            "classA": {"x": 400.0, "y": 100.0, "z": -50.0},
            "classB": {"x": 400.0, "y": 200.0, "z": -50.0},
            "classC": {"x": 400.0, "y": 300.0, "z": -50.0},
        },
        "cpu_affinity": {
            "geometry_pool": [6, 7, 8, 9],
            "coordinator": [12],
            "opcua": [13],
            "ui": [14],
        },
        "logging": {"level": "INFO", "dir": "logs", "retention_days": 30},
        "health": {"heartbeat_timeout_sec": 5.0, "queue_max_size": 10},
        "retry": {"max_retries": 3, "delay_sec": 2.0, "backoff_factor": 1.5},
        "calibration": {"homography_path": "assets/H.npy"},
    }
    default_file = tmp_path / "default.yaml"
    with open(default_file, "w", encoding="utf-8") as f:
        yaml.dump(default_cfg, f, allow_unicode=True)
    return tmp_path


@pytest.fixture
def valid_d435_3d_cfg():
    """合法的 D435 + 3D 設定"""
    return {
        "camera_type": "d435",
        "depth_mode": "3D",
        "plc": {"ip": "192.168.0.10", "opc_port": 4840,
                "opc_namespace": "ns", "cmd_timeout_sec": 10},
        "d435": {"width": 1280, "height": 720, "fps": 30},
        "mechanical": {"suction_length_mm": 80, "safety_margin_mm": 5,
                       "offset_x": 5.0, "offset_y": 3.0},
        "depth_3d": {"camera_height_mm": 800},
        "yolo": {"model": "assets/best_obb.pt", "confidence": 0.5},
        "place_map": {"classA": {"x": 400, "y": 100, "z": -50}},
        "depth_2d": {"thickness_map": {"classA": 5.0}},
    }


@pytest.fixture
def valid_cognex_2d_cfg():
    """合法的 Cognex + 2D 設定"""
    return {
        "camera_type": "cognex",
        "depth_mode": "2D",
        "plc": {"ip": "192.168.0.10", "opc_port": 4840,
                "opc_namespace": "ns", "cmd_timeout_sec": 10},
        "cognex": {"ip": "192.168.0.50", "port": 3000,
                   "cti_path": "C:\\test.cti"},
        "mechanical": {"suction_length_mm": 80, "safety_margin_mm": 5,
                       "offset_x": 0.0, "offset_y": 0.0},
        "depth_2d": {
            "worktable_z_mm": 0.0,
            "thickness_map": {"classA": 5.0, "classB": 10.0},
        },
        "yolo": {"model": "assets/best_obb.pt", "confidence": 0.5},
        "place_map": {
            "classA": {"x": 400, "y": 100, "z": -50},
            "classB": {"x": 400, "y": 200, "z": -50},
        },
    }


# ---------------------------------------------------------------------------
#  Tests: deep_merge
# ---------------------------------------------------------------------------
class TestDeepMerge:
    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 99}

    def test_nested_merge(self):
        base = {"plc": {"ip": "1.1.1.1", "port": 4840}}
        override = {"plc": {"ip": "2.2.2.2"}}
        result = deep_merge(base, override)
        assert result["plc"]["ip"] == "2.2.2.2"
        assert result["plc"]["port"] == 4840  # 未被覆蓋

    def test_add_new_key(self):
        base = {"a": 1}
        override = {"b": 2}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 2}

    def test_does_not_mutate_base(self):
        base = {"plc": {"ip": "1.1.1.1"}}
        override = {"plc": {"ip": "2.2.2.2"}}
        deep_merge(base, override)
        assert base["plc"]["ip"] == "1.1.1.1"  # 原始不被修改

    def test_deep_nested(self):
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"c": 99}}}
        result = deep_merge(base, override)
        assert result["a"]["b"]["c"] == 99
        assert result["a"]["b"]["d"] == 2


# ---------------------------------------------------------------------------
#  Tests: validate_config
# ---------------------------------------------------------------------------
class TestValidateConfig:
    def test_valid_d435_3d(self, valid_d435_3d_cfg):
        """D435 + 3D → 合法"""
        validate_config(valid_d435_3d_cfg)  # 不應拋出例外

    def test_valid_cognex_2d(self, valid_cognex_2d_cfg):
        """Cognex + 2D → 合法"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            validate_config(valid_cognex_2d_cfg)

    def test_cognex_3d_raises(self, valid_cognex_2d_cfg):
        """Cognex + 3D → 不允許"""
        valid_cognex_2d_cfg["depth_mode"] = "3D"
        with pytest.raises(ConfigError, match="Cognex"):
            validate_config(valid_cognex_2d_cfg)

    def test_invalid_camera_type(self, valid_d435_3d_cfg):
        """不合法的 camera_type"""
        valid_d435_3d_cfg["camera_type"] = "kinect"
        with pytest.raises(ConfigError, match="camera_type"):
            validate_config(valid_d435_3d_cfg)

    def test_invalid_depth_mode(self, valid_d435_3d_cfg):
        """不合法的 depth_mode"""
        valid_d435_3d_cfg["depth_mode"] = "4D"
        with pytest.raises(ConfigError, match="depth_mode"):
            validate_config(valid_d435_3d_cfg)

    def test_2d_missing_thickness(self, valid_d435_3d_cfg):
        """2D 模式 thickness_map 缺少 place_map 中的類別"""
        valid_d435_3d_cfg["depth_mode"] = "2D"
        valid_d435_3d_cfg["depth_2d"] = {"thickness_map": {}}  # 空的
        with pytest.raises(ConfigError, match="thickness_map"):
            validate_config(valid_d435_3d_cfg)

    def test_missing_plc_ip(self, valid_d435_3d_cfg):
        """缺少 PLC IP"""
        del valid_d435_3d_cfg["plc"]["ip"]
        with pytest.raises(ConfigError, match="plc.ip"):
            validate_config(valid_d435_3d_cfg)

    def test_missing_yolo_model(self, valid_d435_3d_cfg):
        """缺少 YOLO 模型路徑"""
        del valid_d435_3d_cfg["yolo"]["model"]
        with pytest.raises(ConfigError, match="yolo.model"):
            validate_config(valid_d435_3d_cfg)

    def test_missing_d435_params(self, valid_d435_3d_cfg):
        """D435 模式缺少 D435 參數"""
        del valid_d435_3d_cfg["d435"]
        with pytest.raises(ConfigError, match="d435.width"):
            validate_config(valid_d435_3d_cfg)

    def test_missing_cognex_ip(self, valid_cognex_2d_cfg):
        """Cognex 模式缺少 Cognex IP"""
        del valid_cognex_2d_cfg["cognex"]["ip"]
        with pytest.raises(ConfigError, match="cognex.ip"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                validate_config(valid_cognex_2d_cfg)

    def test_missing_camera_height_3d(self, valid_d435_3d_cfg):
        """3D 模式缺少相機高度"""
        del valid_d435_3d_cfg["depth_3d"]
        with pytest.raises(ConfigError, match="camera_height_mm"):
            validate_config(valid_d435_3d_cfg)

    def test_offset_zero_warning(self, valid_cognex_2d_cfg):
        """偏移量為零時應發出警告"""
        valid_cognex_2d_cfg["mechanical"]["offset_x"] = 0.0
        valid_cognex_2d_cfg["mechanical"]["offset_y"] = 0.0
        with pytest.warns(UserWarning, match="OFFSET"):
            validate_config(valid_cognex_2d_cfg)

    def test_no_warning_with_offset(self, valid_d435_3d_cfg):
        """偏移量非零時不應發出警告"""
        valid_d435_3d_cfg["mechanical"]["offset_x"] = 5.0
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # 有警告就 raise
            validate_config(valid_d435_3d_cfg)


# ---------------------------------------------------------------------------
#  Tests: load_config
# ---------------------------------------------------------------------------
class TestLoadConfig:
    def test_load_default(self, tmp_config_dir):
        """載入 default.yaml"""
        cfg = load_config(config_dir=str(tmp_config_dir))
        assert cfg["camera_type"] == "d435"
        assert cfg["depth_mode"] == "3D"

    def test_load_with_site(self, tmp_config_dir):
        """載入 site 覆蓋"""
        site_cfg = {"camera_type": "cognex", "depth_mode": "2D"}
        site_file = tmp_config_dir / "site_B.yaml"
        with open(site_file, "w") as f:
            yaml.dump(site_cfg, f)

        cfg = load_config(site="B", config_dir=str(tmp_config_dir))
        assert cfg["camera_type"] == "cognex"
        assert cfg["depth_mode"] == "2D"

    def test_load_with_local(self, tmp_config_dir):
        """載入 local.yaml 覆蓋"""
        local_cfg = {"plc": {"ip": "10.0.0.1"}}
        local_file = tmp_config_dir / "local.yaml"
        with open(local_file, "w") as f:
            yaml.dump(local_cfg, f)

        cfg = load_config(config_dir=str(tmp_config_dir))
        assert cfg["plc"]["ip"] == "10.0.0.1"  # local 覆蓋
        assert cfg["plc"]["opc_port"] == 4840   # default 保留

    def test_site_not_exist_no_error(self, tmp_config_dir):
        """site 檔案不存在不應報錯"""
        cfg = load_config(site="nonexistent", config_dir=str(tmp_config_dir))
        assert cfg["camera_type"] == "d435"

    def test_missing_default_raises(self, tmp_path):
        """缺少 default.yaml 應報錯"""
        with pytest.raises(FileNotFoundError):
            load_config(config_dir=str(tmp_path))

    def test_layering_order(self, tmp_config_dir):
        """確認覆蓋順序：default < site < local"""
        # site 設定
        site_cfg = {"plc": {"ip": "site_ip"}}
        with open(tmp_config_dir / "site_A.yaml", "w") as f:
            yaml.dump(site_cfg, f)

        # local 設定
        local_cfg = {"plc": {"ip": "local_ip"}}
        with open(tmp_config_dir / "local.yaml", "w") as f:
            yaml.dump(local_cfg, f)

        cfg = load_config(site="A", config_dir=str(tmp_config_dir))
        assert cfg["plc"]["ip"] == "local_ip"  # local 優先


# ---------------------------------------------------------------------------
#  Tests: AppConfig
# ---------------------------------------------------------------------------
class TestAppConfig:
    def test_properties(self, valid_d435_3d_cfg):
        app = AppConfig(valid_d435_3d_cfg)
        assert app.camera_type == "d435"
        assert app.depth_mode == "3D"
        assert app.plc_ip == "192.168.0.10"
        assert app.suction_length_mm == 80.0
        assert app.offset_x == 5.0
        assert app.offset_y == 3.0
        assert app.yolo_model == "assets/best_obb.pt"

    def test_raw_returns_copy(self, valid_d435_3d_cfg):
        app = AppConfig(valid_d435_3d_cfg)
        raw = app.raw
        raw["camera_type"] = "modified"
        assert app.camera_type == "d435"  # 原始不被修改
