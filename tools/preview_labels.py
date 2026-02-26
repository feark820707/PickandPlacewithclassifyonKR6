# =============================================================================
#  KR6 Pick & Place — YOLO-OBB 標注視覺預覽工具
#
#  功能：
#    - 讀取 labeled/ 或 train/ 目錄的影像 + OBB label
#    - 在影像上繪製旋轉邊界框 + 品類名稱
#    - 鍵盤瀏覽，快速確認標注品質
#
#  Usage:
#      python tools/preview_labels.py                   # 預設 datasets/labeled/
#      python tools/preview_labels.py -d datasets/train/images
#      python tools/preview_labels.py -c classA         # 只看 classA
#
#  操作：
#      → / d / Space  下一張
#      ← / a          上一張
#      q / Esc        離開
# =============================================================================
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

logger = logging.getLogger("preview_labels")

CLASSES = ["classA", "classB", "classC", "classD"]

# 品類顏色（BGR）
_COLORS = [
    (0,   255,  80),   # classA — 綠
    (0,   140, 255),   # classB — 橙
    (255,  60,  60),   # classC — 藍
    (180,   0, 255),   # classD — 紫
]


# ---------------------------------------------------------------------------
#  OBB 解析與繪製
# ---------------------------------------------------------------------------
def _parse_obb_label(txt_path: Path, img_w: int, img_h: int) -> list[dict]:
    """
    解析 YOLO-OBB .txt，回傳 [{cls_id, cls_name, pts_px}] 列表。
    格式：class_id x1 y1 x2 y2 x3 y3 x4 y4（正規化）
    """
    objects = []
    if not txt_path.exists():
        return objects

    for line in txt_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 9:
            continue
        try:
            cls_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            continue

        # 反正規化到像素
        pts = []
        for i in range(0, 8, 2):
            px = int(coords[i]     * img_w)
            py = int(coords[i + 1] * img_h)
            pts.append((px, py))

        cls_name = CLASSES[cls_id] if 0 <= cls_id < len(CLASSES) else f"cls{cls_id}"
        objects.append({"cls_id": cls_id, "cls_name": cls_name, "pts": pts})

    return objects


def _draw_obb(img: np.ndarray, objects: list[dict]) -> np.ndarray:
    """在影像上繪製 OBB 框與品類標籤"""
    out = img.copy()
    for obj in objects:
        cls_id  = obj["cls_id"]
        color   = _COLORS[cls_id % len(_COLORS)]
        pts     = np.array(obj["pts"], dtype=np.int32)

        # 旋轉框
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)

        # 品類標籤（左上角點）
        lx = min(p[0] for p in obj["pts"])
        ly = min(p[1] for p in obj["pts"]) - 6
        cv2.putText(
            out, obj["cls_name"],
            (lx, max(ly, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )

    return out


# ---------------------------------------------------------------------------
#  影像列表建立
# ---------------------------------------------------------------------------
def _collect_pairs(root: Path, cls_filter: str | None) -> list[tuple[Path, Path]]:
    """
    收集 (image_path, label_path) 配對。
    label 與 image 同名但在 ../labels/ 或同目錄。
    """
    root = root.resolve()
    jpgs = sorted(root.rglob("*.jpg"))

    if cls_filter:
        jpgs = [p for p in jpgs if cls_filter in p.stem]

    pairs = []
    for jpg in jpgs:
        # 嘗試找對應 label（同目錄，或 ../labels/）
        candidates = [
            jpg.with_suffix(".txt"),
            jpg.parent.parent / "labels" / jpg.with_suffix(".txt").name,
        ]
        txt = next((c for c in candidates if c.exists()), None)
        pairs.append((jpg, txt))  # txt 可能為 None

    return pairs


# ---------------------------------------------------------------------------
#  主迴圈
# ---------------------------------------------------------------------------
def run(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    scan_dir = Path(args.directory)
    pairs = _collect_pairs(scan_dir, args.cls)

    if not pairs:
        logger.error("在 %s 找不到影像", scan_dir)
        sys.exit(1)

    logger.info("共 %d 張影像（含標注: %d）",
                len(pairs), sum(1 for _, t in pairs if t))

    idx = 0

    while True:
        img_path, txt_path = pairs[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("無法讀取: %s", img_path.name)
            idx = (idx + 1) % len(pairs)
            continue

        h, w = img.shape[:2]

        if txt_path:
            objects = _parse_obb_label(txt_path, w, h)
            display = _draw_obb(img, objects)
            n_obj = len(objects)
        else:
            display = img.copy()
            n_obj = 0
            cv2.putText(display, "NO LABEL", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # 底部資訊欄
        info = f"[{idx+1}/{len(pairs)}] {img_path.name}  objects={n_obj}  <>/d/Space: next  a: prev  q: quit"
        cv2.putText(display, info, (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 0), 1)

        cv2.imshow("OBB Preview", display)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key in (ord('d'), ord(' '), 83):   # d / Space / →
            idx = (idx + 1) % len(pairs)
        elif key in (ord('a'), 81):             # a / ←
            idx = (idx - 1) % len(pairs)

    cv2.destroyAllWindows()
    logger.info("預覽結束")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="YOLO-OBB 標注視覺預覽工具",
    )
    parser.add_argument(
        "-d", "--directory",
        default="datasets/labeled",
        help="影像目錄（預設 datasets/labeled）",
    )
    parser.add_argument(
        "-c", "--cls",
        default=None,
        choices=CLASSES, metavar="CLASS",
        help=f"只顯示特定品類（{'/'.join(CLASSES)}）",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
