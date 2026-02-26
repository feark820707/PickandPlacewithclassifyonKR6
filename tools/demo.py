# =============================================================================
#  KR6 Pick & Place — YOLO-OBB 即時 DEMO
#
#  使用預訓練 yolo11m-obb.pt（DOTA 類別）或自訂模型
#  用途：驗證推論管線 + OBB 框顯示，為後續 fine-tune 做基準確認
#
#  Usage:
#      python tools/demo.py --source cognex           # Cognex 即時串流
#      python tools/demo.py --source d435             # D435 即時串流
#      python tools/demo.py --source img.jpg          # 單張圖檔
#      python tools/demo.py --source datasets/raw/    # 目錄輪播
#      python tools/demo.py --source cognex --model assets/best_obb.pt
#      python tools/demo.py --source img.jpg --conf 0.3 --save
#
#  操作：
#      r / Space → 取下一幀（相機：重新觸發；目錄：下一張）
#      p         → 上一張（目錄模式）
#      s         → 儲存當前幀到 runs/demo/
#      q / Esc   → 離開
# =============================================================================
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

logger = logging.getLogger("demo")

# DOTA 預訓練類別（yolo11m-obb.pt 預設）
DOTA_CLASSES = [
    "plane", "ship", "storage-tank", "baseball-diamond",
    "tennis-court", "basketball-court", "ground-track-field",
    "harbor", "bridge", "large-vehicle", "small-vehicle",
    "helicopter", "roundabout", "soccer-ball-field", "swimming-pool",
]

OUTPUT_DIR = PROJECT_ROOT / "runs" / "demo"

# ---------------------------------------------------------------------------
#  已知相機類型（registry name 前綴）
# ---------------------------------------------------------------------------
_CAMERA_NAMES = {"cognex", "d435", "usb"}

_PALETTE = [
    (0, 220, 60),   (0, 140, 255),  (80, 80, 255),  (220, 40, 220),
    (0, 200, 200),  (255, 180, 0),  (120, 255, 0),  (255, 60, 120),
]

def _color(cls_id: int):
    return _PALETTE[cls_id % len(_PALETTE)]


# ---------------------------------------------------------------------------
#  OBB 繪製
# ---------------------------------------------------------------------------
def _draw_results(img: np.ndarray, results, class_names: list[str],
                  conf_thresh: float) -> tuple[np.ndarray, int]:
    out   = img.copy()
    count = 0

    for result in results:
        if result.obb is None:
            continue
        for obb in result.obb:
            conf = float(obb.conf[0])
            if conf < conf_thresh:
                continue
            cls_id = int(obb.cls[0])
            name   = class_names[cls_id] if cls_id < len(class_names) else f"cls{cls_id}"
            color  = _color(cls_id)

            pts = obb.xyxyxyxy.cpu().numpy().reshape(4, 2).astype(np.int32)
            cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)

            lx = int(pts[:, 0].min())
            ly = int(pts[:, 1].min()) - 6
            label = f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (lx - 1, ly - th - 2), (lx + tw + 1, ly + 2), color, -1)
            cv2.putText(out, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
            count += 1

    return out, count


# ---------------------------------------------------------------------------
#  OSD
# ---------------------------------------------------------------------------
def _draw_osd(img: np.ndarray, fps: float, n_det: int,
              model_name: str, conf: float, extra: str = "") -> None:
    h = img.shape[0]
    cv2.putText(img,
                f"FPS:{fps:.1f}  Det:{n_det}  {model_name}  conf>={conf:.2f}",
                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)
    hint = "r/Space:next  p:prev  s:save  q:quit"
    if extra:
        hint = extra + "  |  " + hint
    cv2.putText(img, hint, (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 50), 1)


# ---------------------------------------------------------------------------
#  載入模型
# ---------------------------------------------------------------------------
def _load_model(model_path: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("請安裝 ultralytics：pip install ultralytics")
        sys.exit(1)
    logger.info("載入模型: %s（首次使用會自動下載）", model_path)
    model = YOLO(model_path)
    logger.info("模型載入完成  classes=%d", len(model.names))
    return model


# ---------------------------------------------------------------------------
#  建立影像來源
# ---------------------------------------------------------------------------
def _build_source(source: str):
    """
    依 source 字串建立影像來源（CameraBase 子類別）。

      "cognex" / "d435"  → 相機驅動
      其他字串            → 視為路徑，使用 FileCamera
    """
    src_lower = source.lower()

    if src_lower == "cognex":
        import yaml
        sec: dict = {}
        try:
            with open(PROJECT_ROOT / "configs" / "site_B.yaml", encoding="utf-8") as f:
                sec = (yaml.safe_load(f) or {}).get("cognex", {})
        except Exception:
            pass
        from camera.cognex_stream import CognexCamera
        return CognexCamera(
            ip=sec.get("ip", "192.168.0.10"),
            telnet_port=sec.get("telnet_port", 23),
            ftp_port=sec.get("ftp_port", 21),
            ftp_user=sec.get("ftp_user", "admin"),
            ftp_password=sec.get("ftp_password", ""),
            depth_mode="2D",
        )

    if src_lower == "d435":
        from camera.d435_stream import D435Camera
        return D435Camera(depth_mode="2D")

    if src_lower.startswith("usb"):
        # "usb" → index=0；"usb:1" → index=1
        idx = 0
        if ":" in source:
            idx = int(source.split(":", 1)[1])
        from camera.usb_cam import UsbCamera
        return UsbCamera(index=idx)

    # 視為路徑 → FileCamera
    from camera.file_source import FileCamera
    return FileCamera(path=source, loop=True)


# ---------------------------------------------------------------------------
#  主 DEMO 迴圈
# ---------------------------------------------------------------------------
def run_demo(args):
    cam         = _build_source(args.source)
    model       = _load_model(args.model)
    class_names = list(model.names.values())
    logger.info("類別列表: %s", class_names)

    is_file_src = not args.source.lower().split(":")[0] in _CAMERA_NAMES
    is_dir_mode = False

    WIN = f"YOLO-OBB Demo — {args.source}"

    save_count  = 0
    frame_count = 0
    fps         = 0.0

    try:
        cam.connect()

        from camera.file_source import FileCamera
        if isinstance(cam, FileCamera):
            is_dir_mode = cam.is_dir_mode()
            if is_dir_mode:
                logger.info("目錄模式: %d 張影像", cam.file_count)

        raw, _ = cam.get_frame()

        # 連線並取得第一幀後才建立視窗，避免先出現空窗
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

        while True:
            t0 = time.perf_counter()

            results = model(raw, conf=args.conf, verbose=False)
            out, n  = _draw_results(raw, results, class_names, args.conf)

            elapsed = time.perf_counter() - t0
            fps = 0.9 * fps + 0.1 * (1.0 / max(elapsed, 1e-6))

            extra = ""
            if is_dir_mode and isinstance(cam, FileCamera):
                extra = f"[{cam.current_index}/{cam.file_count}] {cam.current_filename}"
            _draw_osd(out, fps, n, Path(args.model).stem, args.conf, extra)

            cv2.imshow(WIN, out)
            frame_count += 1

            # 單圖（非目錄）：等待按鍵；其他：1ms polling
            wait_ms = 0 if (is_file_src and not is_dir_mode) else 1
            key = cv2.waitKey(wait_ms) & 0xFF

            # ── 離開 ────────────────────────────────────────────────
            if key in (ord('q'), 27):
                break

            # ── 下一幀 ──────────────────────────────────────────────
            elif key in (ord('r'), ord(' ')):
                try:
                    raw, _ = cam.get_frame()
                    logger.debug("[%d] 取得新幀", frame_count)
                except StopIteration:
                    logger.info("目錄已播放完畢")
                    break
                except Exception as e:
                    logger.error("取幀失敗: %s", e)

            # ── 上一張（目錄模式）───────────────────────────────────
            elif key == ord('p'):
                if is_dir_mode and isinstance(cam, FileCamera):
                    raw = cam.go_prev()

            # ── 儲存 ────────────────────────────────────────────────
            elif key == ord('s') or args.save:
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                path = OUTPUT_DIR / f"demo_{save_count:04d}.jpg"
                cv2.imwrite(str(path), out)
                save_count += 1
                logger.info("已儲存: %s", path)
                if args.save and not is_file_src:
                    try:
                        raw, _ = cam.get_frame()
                    except Exception:
                        pass

    finally:
        cam.disconnect()
        cv2.destroyAllWindows()
        logger.info("DEMO 結束  幀數:%d  儲存:%d 張", frame_count, save_count)




# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="YOLO-OBB 即時 DEMO",
    )
    parser.add_argument(
        "--source", default=None,
        help="輸入來源：cognex / d435 / usb / usb:N / 圖檔路徑 / 目錄路徑\n"
             "省略時顯示互動式選單",
    )
    parser.add_argument("--model", default="yolo11m-obb.pt",
                        help="模型路徑（預設 yolo11m-obb.pt）")
    parser.add_argument("--conf",  default=0.25, type=float,
                        help="信心閾值（預設 0.25）")
    parser.add_argument("--save",  action="store_true",
                        help="自動儲存每幀推論結果到 runs/demo/")
    args = parser.parse_args()

    if args.source is None:
        from camera.picker import pick_source
        args.source = pick_source(title="YOLO-OBB Demo — 選擇影像來源")
        if args.source is None:
            sys.exit(0)

    run_demo(args)


if __name__ == "__main__":
    main()
