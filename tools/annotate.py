# =============================================================================
#  KR6 Pick & Place — YOLO-OBB 標注工具
#
#  Usage:
#      python tools/annotate.py                    # 標注 datasets/raw/ 所有影像
#      python tools/annotate.py -c classA          # 只標注 classA
#      python tools/annotate.py -i datasets/raw    # 自訂輸入目錄
#
#  操作說明：
#    ── 繪製 OBB ──────────────────────────────────
#      左鍵點擊 ×3   第1點→第2點（定義第一條邊：方向+長度）
#                    →第3點（定義垂直距離，完成旋轉框）
#      右鍵 / Esc    取消目前繪製
#      z             撤銷最後一個標注框
#      c             清除目前影像所有標注
#    ── 品類切換 ──────────────────────────────────
#      1 / 2 / 3 / 4  切換品類 (classA/B/C/D)
#    ── 瀏覽 ──────────────────────────────────────
#      Space / n / →  存檔並下一張
#      p / ←          存檔並上一張
#      Enter           存檔（停留在目前影像）
#      q               存檔並離開
#
#  輸出格式（YOLO-OBB）：
#      datasets/labeled/{class}/{name}.txt
#      每行：class_id  x1 y1  x2 y2  x3 y3  x4 y4  （正規化 0~1）
# =============================================================================
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

logger = logging.getLogger("annotate")

CLASSES = ["classA", "classB", "classC", "classD"]

# 品類顏色 BGR
_COLORS = [
    (0,   220,  60),   # classA 綠
    (0,   140, 255),   # classB 橙
    (80,   80, 255),   # classC 紅
    (220,  40, 220),   # classD 紫
]

# ---------------------------------------------------------------------------
#  幾何計算
# ---------------------------------------------------------------------------

def _compute_obb(p1, p2, p3) -> list[tuple[int, int]] | None:
    """
    p1, p2 定義第一條邊；p3 定義垂直距離，計算 4 個角點。
    """
    ex = p2[0] - p1[0]
    ey = p2[1] - p1[1]
    edge_len = math.sqrt(ex * ex + ey * ey)
    if edge_len < 2:
        return None

    # 單位法向量（向右旋轉 90°）
    nx = -ey / edge_len
    ny =  ex / edge_len

    # p3 投影到法向量上的帶符號距離
    dx = p3[0] - p1[0]
    dy = p3[1] - p1[1]
    d  = dx * nx + dy * ny

    c1 = p1
    c2 = p2
    c3 = (int(round(p2[0] + d * nx)), int(round(p2[1] + d * ny)))
    c4 = (int(round(p1[0] + d * nx)), int(round(p1[1] + d * ny)))
    return [c1, c2, c3, c4]


def _corners_to_yolo(corners, w: int, h: int) -> list[float]:
    """
    4 角點 → YOLO-OBB 正規化座標（用 minAreaRect 標準化順序）。
    """
    pts = np.array(corners, dtype=np.float32)
    rect = cv2.minAreaRect(pts)
    box  = cv2.boxPoints(rect)          # 標準化順序的 4 點
    result = []
    for x, y in box:
        result.extend([max(0.0, min(1.0, x / w)),
                       max(0.0, min(1.0, y / h))])
    return result


def _yolo_to_corners(values: list[float], w: int, h: int) -> list[tuple[int, int]]:
    """YOLO 正規化座標 → 像素角點"""
    pts = []
    for i in range(0, 8, 2):
        pts.append((int(values[i] * w), int(values[i + 1] * h)))
    return pts


# ---------------------------------------------------------------------------
#  Label I/O
# ---------------------------------------------------------------------------

def _load_label(txt_path: Path, w: int, h: int) -> list[dict]:
    """載入已存在的 label 檔"""
    annotations = []
    if not txt_path.exists():
        return annotations
    for line in txt_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 9:
            continue
        try:
            cls_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]
            corners = _yolo_to_corners(coords, w, h)
            annotations.append({"cls_id": cls_id, "corners": corners})
        except ValueError:
            pass
    return annotations


def _save_label(txt_path: Path, annotations: list[dict], w: int, h: int) -> None:
    """儲存 label 檔（空標注 → 建立空檔）"""
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for ann in annotations:
        coords = _corners_to_yolo(ann["corners"], w, h)
        coord_str = " ".join(f"{v:.6f}" for v in coords)
        lines.append(f"{ann['cls_id']} {coord_str}")
    txt_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
#  繪製函式
# ---------------------------------------------------------------------------

def _draw_annotations(canvas, annotations: list[dict]) -> None:
    """繪製所有已完成的標注"""
    for ann in annotations:
        cls_id  = ann["cls_id"]
        color   = _COLORS[cls_id % len(_COLORS)]
        pts     = np.array(ann["corners"], dtype=np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
        lx = min(p[0] for p in ann["corners"])
        ly = min(p[1] for p in ann["corners"]) - 6
        cv2.putText(canvas, CLASSES[cls_id] if cls_id < len(CLASSES) else f"cls{cls_id}",
                    (lx, max(ly, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def _draw_crosshair(canvas, pos, color=(200, 200, 200)) -> None:
    h, w = canvas.shape[:2]
    x, y = pos
    cv2.line(canvas, (0, y), (w, y), color, 1)
    cv2.line(canvas, (x, 0), (x, h), color, 1)


def _draw_hud(canvas, idx: int, total: int, cls_id: int,
              annotations: list[dict], state: int) -> None:
    h, w = canvas.shape[:2]
    # 上方：品類 + 進度
    color = _COLORS[cls_id % len(_COLORS)]
    cv2.putText(canvas,
                f"[{idx+1}/{total}]  Class: {CLASSES[cls_id]}  Boxes: {len(annotations)}",
                (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    # 下方：操作說明
    hint = {
        0: "左鍵點擊: 設定第1點",
        1: "左鍵點擊: 設定第2點 (定義邊方向)",
        2: "左鍵點擊: 確定垂直距離 | 右鍵/Esc: 取消",
    }.get(state, "")
    cv2.putText(canvas, f"{hint}  |  1-4:切換品類  z:撤銷  c:清除  Space/n:下一張  q:離開",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 50), 1)


def _draw_in_progress(canvas, points: list, mouse_pos, state: int, cls_id: int) -> None:
    """繪製目前正在畫的框（預覽）"""
    color = _COLORS[cls_id % len(_COLORS)]
    if state >= 1 and points:
        cv2.circle(canvas, points[0], 5, color, -1)
        cv2.line(canvas, points[0], mouse_pos, color, 1)
    if state >= 2 and len(points) >= 2:
        cv2.line(canvas, points[0], points[1], color, 2)
        corners = _compute_obb(points[0], points[1], mouse_pos)
        if corners:
            pts = np.array(corners, dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)


# ---------------------------------------------------------------------------
#  影像列表
# ---------------------------------------------------------------------------

def _collect_images(in_root: Path, cls_filter: str | None) -> list[Path]:
    jpgs = sorted(in_root.rglob("*.jpg"))
    if cls_filter:
        jpgs = [p for p in jpgs if cls_filter in p.stem or p.parent.name == cls_filter]
    return jpgs


def _label_path(img_path: Path, in_root: Path, out_root: Path) -> Path:
    """計算輸出 label 路徑（維持同樣的子目錄結構）"""
    rel = img_path.relative_to(in_root)
    return (out_root / rel).with_suffix(".txt")


# ---------------------------------------------------------------------------
#  主迴圈
# ---------------------------------------------------------------------------

class _MouseState:
    def __init__(self):
        self.pos   = (0, 0)
        self.click = None   # 最新一次左鍵點擊座標


def _mouse_cb(event, x, y, flags, state: _MouseState):
    state.pos = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        state.click = (x, y)


def run(args):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    in_root  = Path(args.input).resolve()
    out_root = Path(args.output).resolve()

    images = _collect_images(in_root, args.cls)
    if not images:
        logger.error("找不到影像: %s", in_root)
        sys.exit(1)
    logger.info("共 %d 張影像，輸出: %s", len(images), out_root)

    idx      = 0
    ms       = _MouseState()
    WIN      = "OBB Annotator"

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, _mouse_cb, ms)

    # 每張影像的工作狀態
    img        = None
    img_h = img_w = 0
    annotations: list[dict] = []
    draw_pts:   list        = []  # 目前繪製中的點 (0~2 個)
    draw_state  = 0               # 0=idle 1=p1_set 2=edge_set
    current_cls = 0

    def load_image(i):
        nonlocal img, img_h, img_w, annotations, draw_pts, draw_state
        path = images[i]
        raw  = cv2.imread(str(path))
        if raw is None:
            logger.warning("無法讀取: %s", path.name)
            return False
        img   = raw
        img_h, img_w = img.shape[:2]
        txt   = _label_path(path, in_root, out_root)
        annotations = _load_label(txt, img_w, img_h)
        draw_pts  = []
        draw_state = 0
        ms.click   = None
        logger.info("[%d/%d] %s  既有標注: %d", i + 1, len(images),
                    path.name, len(annotations))
        return True

    def save_current():
        txt = _label_path(images[idx], in_root, out_root)
        _save_label(txt, annotations, img_w, img_h)
        # 同時把影像複製到 labeled/ (如果不在同目錄)
        dst_img = txt.with_suffix(".jpg")
        if not dst_img.exists():
            import shutil
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(images[idx], dst_img)
        logger.info("已存: %s (%d 個框)", txt.name, len(annotations))

    load_image(idx)

    while True:
        # ── 建立顯示畫面 ──────────────────────────────────────────────
        canvas = img.copy()
        _draw_annotations(canvas, annotations)
        _draw_crosshair(canvas, ms.pos)
        _draw_in_progress(canvas, draw_pts, ms.pos, draw_state, current_cls)
        _draw_hud(canvas, idx, len(images), current_cls, annotations, draw_state)
        cv2.imshow(WIN, canvas)

        # ── 處理滑鼠點擊 ──────────────────────────────────────────────
        if ms.click is not None:
            click = ms.click
            ms.click = None

            if draw_state == 0:
                draw_pts   = [click]
                draw_state = 1
            elif draw_state == 1:
                draw_pts.append(click)
                draw_state = 2
            elif draw_state == 2:
                corners = _compute_obb(draw_pts[0], draw_pts[1], click)
                if corners:
                    annotations.append({"cls_id": current_cls, "corners": corners})
                draw_pts   = []
                draw_state = 0

        # ── 鍵盤 ──────────────────────────────────────────────────────
        key = cv2.waitKey(30) & 0xFF

        if key == 255:
            continue

        # 取消目前繪製
        if key in (ord('q') & 0xFF if draw_state > 0 else 255,
                   cv2.EVENT_RBUTTONDOWN):
            pass   # 右鍵透過 mouse_cb 無法在這裡直接捕捉，用 Esc 代替
        if key == 27:                          # Esc
            if draw_state > 0:
                draw_pts   = []
                draw_state = 0
            else:
                save_current()
                break

        elif key == ord('q'):                  # q 離開
            save_current()
            break

        elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
            current_cls = key - ord('1')
            logger.info("品類: %s", CLASSES[current_cls])

        elif key == ord('z'):                  # 撤銷
            if draw_state > 0:
                draw_pts   = []
                draw_state = 0
            elif annotations:
                annotations.pop()
                logger.info("撤銷，剩 %d 框", len(annotations))

        elif key == ord('c'):                  # 清除全部
            annotations.clear()
            draw_pts   = []
            draw_state = 0
            logger.info("已清除所有標注")

        elif key in (ord(' '), ord('n'), 83):  # Space/n/→ 下一張
            save_current()
            idx = min(idx + 1, len(images) - 1)
            load_image(idx)

        elif key in (ord('p'), 81):            # p/← 上一張
            save_current()
            idx = max(idx - 1, 0)
            load_image(idx)

        elif key == 13:                        # Enter 存檔留原位
            save_current()

    cv2.destroyAllWindows()
    total_labeled = sum(
        1 for img_p in images
        if _label_path(img_p, in_root, out_root).exists()
    )
    logger.info("結束  已標注: %d / %d", total_labeled, len(images))


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="YOLO-OBB 標注工具（三點點擊繪製旋轉框）",
    )
    parser.add_argument(
        "-i", "--input",
        default="datasets/raw",
        help="影像來源目錄（預設 datasets/raw）",
    )
    parser.add_argument(
        "-o", "--output",
        default="datasets/labeled",
        help="標注輸出目錄（預設 datasets/labeled）",
    )
    parser.add_argument(
        "-c", "--cls",
        default=None, choices=CLASSES, metavar="CLASS",
        help=f"只標注指定品類（{'/'.join(CLASSES)}）",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
