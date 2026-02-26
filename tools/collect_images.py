# =============================================================================
#  KR6 Pick & Place — Cognex 影像採集工具（YOLO 訓練資料）
#
#  Usage:
#      cd pick_AIDC
#      python tools/collect_images.py               # 互動式，數字鍵切換品類
#      python tools/collect_images.py -c classA     # 直接指定品類
#      python tools/collect_images.py -o datasets/raw
#
#  操作：
#      1 / 2 / 3 / 4  → 切換目標品類（classA/B/C/D）
#      空白鍵 / s     → 觸發拍照並存圖到目前品類
#      r              → 刷新預覽（不存圖）
#      q / Esc        → 離開並顯示採集摘要
#
#  輸出結構：
#      datasets/raw/
#        classA/  classA_0001.jpg  classA_0002.jpg ...
#        classB/  classB_0001.jpg ...
#        classC/  ...
#        classD/  ...
# =============================================================================
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── 專案 root ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from camera.cognex_stream import CognexCamera

logger = logging.getLogger("collect_images")

# ---------------------------------------------------------------------------
#  品類設定（未來增加品類時在此擴充）
# ---------------------------------------------------------------------------
CLASSES = ["classA", "classB", "classC", "classD"]

# 數字鍵 1~4 對應品類索引
_KEY_TO_IDX = {ord(str(i + 1)): i for i in range(len(CLASSES))}


# ---------------------------------------------------------------------------
#  Cognex 連線參數
# ---------------------------------------------------------------------------
def _load_cognex_params(args) -> dict:
    try:
        import yaml
        cfg_path = PROJECT_ROOT / "configs" / "site_B.yaml"
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        sec = cfg.get("cognex", {})
    except Exception:
        sec = {}

    return {
        "ip":           args.ip          or sec.get("ip",           "192.168.0.10"),
        "telnet_port":  args.telnet_port or sec.get("telnet_port",  23),
        "ftp_port":     args.ftp_port    or sec.get("ftp_port",     21),
        "ftp_user":     sec.get("ftp_user",     "admin"),
        "ftp_password": sec.get("ftp_password", ""),
    }


# ---------------------------------------------------------------------------
#  各品類目前已存張數（讀既有檔案，避免覆蓋）
# ---------------------------------------------------------------------------
def _count_existing(out_root: Path, class_name: str) -> int:
    class_dir = out_root / class_name
    if not class_dir.exists():
        return 0
    return len(list(class_dir.glob(f"{class_name}_*.jpg")))


# ---------------------------------------------------------------------------
#  存圖
# ---------------------------------------------------------------------------
def _save_image(frame, out_root: Path, class_name: str, index: int) -> Path:
    class_dir = out_root / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    path = class_dir / f"{class_name}_{index:04d}.jpg"
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


# ---------------------------------------------------------------------------
#  OSD（On-Screen Display）
# ---------------------------------------------------------------------------
def _draw_osd(frame, current_class: str, counts: dict[str, int]) -> None:
    """在影像上疊加品類與計數資訊"""
    h, w = frame.shape[:2]

    # 上方：目前品類 + 操作提示
    cv2.putText(
        frame,
        f"Class: {current_class}  |  SPACE/s: save  r: refresh  1-4: switch  q: quit",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
    )

    # 右上角：各品類計數
    for i, cls in enumerate(CLASSES):
        marker = ">>>" if cls == current_class else "   "
        text = f"{marker} {cls}: {counts[cls]}"
        color = (0, 255, 255) if cls == current_class else (180, 180, 180)
        cv2.putText(
            frame, text,
            (w - 220, 30 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )


# ---------------------------------------------------------------------------
#  主迴圈
# ---------------------------------------------------------------------------
def run(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    out_root = Path(args.output)

    # 初始化各品類計數（讀既有檔案）
    counts: dict[str, int] = {cls: _count_existing(out_root, cls) for cls in CLASSES}
    logger.info("既有影像數: %s", counts)

    # 起始品類
    if args.cls and args.cls in CLASSES:
        current_idx = CLASSES.index(args.cls)
    else:
        current_idx = 0
    logger.info("起始品類: %s", CLASSES[current_idx])

    params = _load_cognex_params(args)
    logger.info("Cognex: IP=%s  Telnet=%d  FTP=%d",
                params["ip"], params["telnet_port"], params["ftp_port"])

    cam = CognexCamera(**params, depth_mode="2D")
    current_frame = None

    try:
        cam.connect()
        logger.info("連線成功，開始採集")

        current_frame, _ = cam.get_frame()

        while True:
            current_class = CLASSES[current_idx]

            display = current_frame.copy()
            _draw_osd(display, current_class, counts)

            win_title = f"Cognex Collect — {sum(counts.values())} total"
            cv2.imshow(win_title, display)

            key = cv2.waitKey(50) & 0xFF

            # 離開
            if key in (ord('q'), 27):
                break

            # 切換品類（數字鍵 1-4）
            elif key in _KEY_TO_IDX:
                current_idx = _KEY_TO_IDX[key]
                logger.info("切換品類 → %s", CLASSES[current_idx])

            # 存圖
            elif key in (ord(' '), ord('s')):
                try:
                    current_frame, _ = cam.get_frame()
                    idx = counts[current_class]
                    path = _save_image(current_frame, out_root, current_class, idx)
                    counts[current_class] += 1
                    logger.info("[%s #%04d] %s", current_class, counts[current_class], path.name)

                    # 閃爍提示
                    flash = current_frame.copy()
                    _draw_osd(flash, current_class, counts)
                    cv2.putText(flash, f"SAVED {path.name}",
                                (10, flash.shape[0] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow(win_title, flash)
                    cv2.waitKey(250)
                except Exception as e:
                    logger.error("存圖失敗: %s", e)

            # 刷新
            elif key == ord('r'):
                try:
                    current_frame, _ = cam.get_frame()
                    logger.info("畫面已刷新")
                except Exception as e:
                    logger.error("刷新失敗: %s", e)

    finally:
        cam.disconnect()
        cv2.destroyAllWindows()

        total = sum(counts.values())
        logger.info("═" * 40)
        logger.info("採集結束 — 合計 %d 張", total)
        for cls in CLASSES:
            logger.info("  %-8s %d 張", cls, counts[cls])
        logger.info("輸出目錄: %s", out_root.resolve())
        logger.info("═" * 40)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Cognex IS8505P 影像採集工具（YOLO 訓練資料，4 品類）",
    )
    parser.add_argument("-o", "--output",      default="datasets/raw",
                        help="輸出根目錄（預設 datasets/raw）")
    parser.add_argument("-c", "--cls",         default=None,
                        choices=CLASSES, metavar="CLASS",
                        help=f"起始品類（{'/'.join(CLASSES)}），預設 classA")
    parser.add_argument("--ip",                default=None,  help="覆蓋 Cognex IP")
    parser.add_argument("--telnet-port",       default=None, type=int,
                        dest="telnet_port",    help="覆蓋 Telnet port")
    parser.add_argument("--ftp-port",          default=None, type=int,
                        dest="ftp_port",       help="覆蓋 FTP port")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
