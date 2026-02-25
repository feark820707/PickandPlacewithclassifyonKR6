# =============================================================================
#  KR6 Pick & Place — 系統入口
#  載入設定 → 驗證 → 初始化 → 啟動主迴圈
#
#  Usage:
#      python main.py                        # 預設（default.yaml only）
#      python main.py --site A               # 現場 A（D435 + 3D）
#      python main.py --site B               # 現場 B（Cognex + 2D）
#      python main.py --site A --dry-run     # Dry-run（不連 PLC/Camera）
#      python main.py --calibrate hand-eye   # 進入手眼標定模式
#      python main.py --calibrate height     # 進入相機高度量測模式
# =============================================================================
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
#  Early path setup (ensure project root on sys.path)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import AppConfig, ConfigError, load_config
from coordinator.coordinator import Coordinator
from coordinator.state_machine import SystemState
from utils.logger import setup_logger

logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
#  Banner
# ---------------------------------------------------------------------------
BANNER = r"""
╔═══════════════════════════════════════════════════════════════╗
║     KR6 Pick & Place  —  Vision-Guided Automation System     ║
║     KUKA KR6 + S7-1515 PLC + YOLO-OBB + OPC-UA              ║
╚═══════════════════════════════════════════════════════════════╝
"""


# ---------------------------------------------------------------------------
#  Signal Handler
# ---------------------------------------------------------------------------
class GracefulShutdown:
    """攔截 SIGINT / SIGTERM，設定 shutdown 旗標"""

    def __init__(self):
        self._shutdown_requested = False
        self._coordinator: Coordinator | None = None

        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def bind(self, coordinator: Coordinator) -> None:
        self._coordinator = coordinator

    def _handler(self, signum, frame):
        sig_name = signal.Signals(signum).name
        logger.warning("收到信號 %s — 執行安全關機...", sig_name)
        self._shutdown_requested = True
        if self._coordinator:
            self._coordinator.shutdown()

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_requested


# ---------------------------------------------------------------------------
#  CLI Argument Parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kr6_pick_place",
        description="KR6 Pick & Place Vision-Guided Automation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python main.py                        預設設定啟動
  python main.py --site A               使用 site_A.yaml
  python main.py --site B --dry-run     Cognex+2D dry-run
  python main.py --calibrate hand-eye   手眼標定模式
  python main.py --calibrate height     相機高度量測
  python main.py --status               顯示系統狀態後退出
        """,
    )

    parser.add_argument(
        "--site",
        type=str,
        default=None,
        help="現場名稱（如 A, B）→ 載入 site_{name}.yaml",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="設定檔目錄路徑（預設: configs/）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run 模式：不連線 PLC / Camera，僅驗證設定",
    )
    parser.add_argument(
        "--calibrate",
        choices=["hand-eye", "height"],
        default=None,
        help="進入標定模式（hand-eye: 手眼標定, height: 相機高度量測）",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="停用 UI 視窗（純後台模式）",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="覆蓋設定檔的 log level",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="顯示系統設定摘要後退出",
    )

    return parser


# ---------------------------------------------------------------------------
#  Startup Helpers
# ---------------------------------------------------------------------------
def show_config_summary(app_cfg: AppConfig) -> None:
    """顯示設定摘要"""
    print(BANNER)
    print("  System Configuration")
    print("  " + "─" * 45)
    print(f"  Camera Type:    {app_cfg.camera_type.upper()}")
    print(f"  Depth Mode:     {app_cfg.depth_mode}")
    print(f"  PLC IP:         {app_cfg.plc_ip}:{app_cfg.opc_port}")
    print(f"  YOLO Model:     {app_cfg.yolo_model}")
    print(f"  YOLO Device:    {app_cfg.yolo_device}")
    print(f"  YOLO Confidence:{app_cfg.yolo_confidence}")
    print(f"  Offset X/Y:     ({app_cfg.offset_x}, {app_cfg.offset_y}) mm")
    print(f"  Suction Length:  {app_cfg.suction_length_mm} mm")
    print(f"  Safety Margin:   {app_cfg.safety_margin_mm} mm")

    if app_cfg.depth_mode == "3D":
        print(f"  Camera Height:   {app_cfg.camera_height_mm} mm")
    else:
        print(f"  Worktable Z:     {app_cfg.worktable_z_mm} mm")
        print(f"  Thickness Map:   {app_cfg.thickness_map}")

    print(f"  Log Level:       {app_cfg.log_level}")
    print(f"  Log Dir:         {app_cfg.log_dir}")
    print(f"  Place Map:       {list(app_cfg.place_map.keys())}")
    print("  " + "─" * 45)


def run_calibration(mode: str, app_cfg: AppConfig) -> int:
    """執行標定模式"""
    if mode == "hand-eye":
        from calibration.hand_eye_calib import HandEyeCalibrator

        print("\n  進入手眼標定模式...")
        print("  確保機器人已就位，相機已連線。\n")

        calibrator = HandEyeCalibrator(
            camera_type=app_cfg.camera_type,
            output_path=app_cfg.homography_path,
        )
        try:
            calibrator.run_interactive()
            print("\n  ✅ 手眼標定完成！")
            return 0
        except Exception as e:
            logger.error("手眼標定失敗: %s", e, exc_info=True)
            print(f"\n  ❌ 手眼標定失敗: {e}")
            return 1

    elif mode == "height":
        from calibration.camera_height import CameraHeightMeasurer

        print("\n  進入相機高度量測模式...")
        measurer = CameraHeightMeasurer()
        try:
            height = measurer.measure()
            print(f"\n  ✅ 相機高度: {height:.1f} mm")
            print(f"  已儲存至設定。請將值填入 configs/ 中的 depth_3d.camera_height_mm")
            return 0
        except Exception as e:
            logger.error("高度量測失敗: %s", e, exc_info=True)
            print(f"\n  ❌ 高度量測失敗: {e}")
            return 1

    return 1


def dry_run_check(app_cfg: AppConfig) -> int:
    """Dry-run 模式：僅驗證設定與相依套件"""
    print(BANNER)
    print("  🔍 Dry-Run Mode — 僅驗證設定，不啟動系統")
    print("  " + "─" * 45)

    # 1. 設定驗證（已在 load_config 中通過）
    print("  ✅ Config validation passed")

    # 2. 相依套件檢查
    deps_ok = True
    required_imports = {
        "numpy": "numpy",
        "cv2": "opencv-python",
        "yaml": "pyyaml",
    }

    # 根據配置加入選用相依
    if app_cfg.camera_type == "d435":
        required_imports["pyrealsense2"] = "pyrealsense2"
    elif app_cfg.camera_type == "cognex":
        required_imports["harvesters"] = "harvesters"

    for module, package in required_imports.items():
        try:
            __import__(module)
            print(f"  ✅ {package} ({module})")
        except ImportError:
            print(f"  ❌ {package} ({module}) — pip install {package}")
            deps_ok = False

    # 選用套件（不影響 dry-run 通過）
    optional = {
        "ultralytics": "ultralytics (YOLO)",
        "opcua": "opcua (OPC-UA)",
        "psutil": "psutil",
    }
    for module, desc in optional.items():
        try:
            __import__(module)
            print(f"  ✅ {desc}")
        except ImportError:
            print(f"  ⚠️  {desc} — 未安裝（正式運行時需要）")

    # 3. 檢查 H.npy
    h_path = Path(app_cfg.homography_path)
    if h_path.exists():
        print(f"  ✅ Homography matrix: {h_path}")
    else:
        print(f"  ⚠️  Homography matrix 不存在: {h_path}")
        print(f"     請先執行: python main.py --calibrate hand-eye")

    # 4. 檢查 YOLO model
    model_path = Path(app_cfg.yolo_model)
    if model_path.exists():
        print(f"  ✅ YOLO model: {model_path}")
    else:
        print(f"  ⚠️  YOLO model 不存在: {model_path}")
        print(f"     請將 best_obb.pt 放至: {model_path}")

    print("  " + "─" * 45)

    if deps_ok:
        print("  ✅ Dry-run 通過！可以使用 --site 參數正式啟動。")
        return 0
    else:
        print("  ❌ 有必要相依套件未安裝，請先安裝後再啟動。")
        return 1


# ---------------------------------------------------------------------------
#  Main Entry Point
# ---------------------------------------------------------------------------
def main() -> int:
    """主入口"""
    parser = build_parser()
    args = parser.parse_args()

    # --- 1. 載入設定 ---
    try:
        cfg_dict = load_config(site=args.site, config_dir=args.config_dir)
    except FileNotFoundError as e:
        print(f"❌ 設定檔錯誤: {e}", file=sys.stderr)
        return 1
    except ConfigError as e:
        print(f"❌ 設定驗證失敗: {e}", file=sys.stderr)
        return 1

    app_cfg = AppConfig(cfg_dict)

    # 覆蓋 log level
    if args.log_level:
        cfg_dict.setdefault("logging", {})["level"] = args.log_level

    # --- 2. 初始化日誌 ---
    setup_logger(
        "main",
        log_dir=app_cfg.log_dir,
        level=app_cfg.log_level,
        retention_days=app_cfg.log_retention_days,
        use_json=True,
    )

    # --- 3. 特殊模式 ---
    if args.status:
        show_config_summary(app_cfg)
        return 0

    if args.dry_run:
        return dry_run_check(app_cfg)

    if args.calibrate:
        return run_calibration(args.calibrate, app_cfg)

    # --- 4. 正式啟動 ---
    show_config_summary(app_cfg)

    # 設定 graceful shutdown
    shutdown_handler = GracefulShutdown()

    logger.info(
        "系統啟動: camera=%s, depth=%s, site=%s",
        app_cfg.camera_type, app_cfg.depth_mode, args.site or "default",
    )

    # 建立 Coordinator
    coordinator = Coordinator(app_cfg)
    shutdown_handler.bind(coordinator)

    # 啟動 UI（若未停用）
    ui_thread = None
    if not args.no_ui:
        try:
            from ui.ui_monitor import UIMonitor
            ui_monitor = UIMonitor(coordinator)
            ui_thread = ui_monitor.start_thread()
            logger.info("UI Monitor 已啟動")
        except ImportError:
            logger.warning("UI Monitor 無法載入（缺少 opencv-python），以純後台模式運行")
        except Exception as e:
            logger.warning("UI Monitor 啟動失敗: %s，以純後台模式運行", e)

    # --- 5. 初始化 ---
    print("\n  🔧 系統初始化中...")
    if not coordinator.initialize():
        logger.error("系統初始化失敗")
        print("  ❌ 系統初始化失敗！請檢查日誌。")
        return 1

    print("  ✅ 系統初始化完成")
    print("  🚀 啟動主迴圈... (Ctrl+C 安全關機)\n")

    # --- 6. 主迴圈 ---
    try:
        coordinator.run()
    except Exception as e:
        logger.error("系統異常退出: %s", e, exc_info=True)
        print(f"\n  ❌ 系統異常: {e}")
        return 1
    finally:
        # 確保 UI Thread 結束
        if ui_thread and ui_thread.is_alive():
            ui_thread.join(timeout=3.0)

    # --- 7. 正常退出 ---
    if shutdown_handler.shutdown_requested:
        print("\n  ✅ 系統已安全關機。")
    else:
        print("\n  ℹ️  系統已停止。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
