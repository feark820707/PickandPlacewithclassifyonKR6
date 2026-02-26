# =============================================================================
#  KR6 Pick & Place — 曝光度 / 影像品質調整工具
#
#  功能：
#    - Cognex 即時畫面 + 直方圖同步顯示
#    - 軟體層調整：亮度、對比、CLAHE、Gamma
#    - 硬體層：Cognex In-Sight Native Mode 曝光設定
#      （需在 In-Sight 工作表中設定曝光控制格）
#    - 儲存最佳參數到 configs/preprocess_tuned.yaml
#
#  Usage:
#      python tools/adjust_exposure.py               # 軟體調整
#      python tools/adjust_exposure.py --hw-cell 1 2 # 寫入 In-Sight 格(col=1,row=2)
#      python tools/adjust_exposure.py --no-camera   # 離線調整（載入靜態圖）
#
#  操作：
#      r        → 刷新畫面（重新拍一張）
#      s        → 儲存目前軟體參數到 preprocess_tuned.yaml
#      q / Esc  → 離開
#
#  In-Sight 曝光設定說明：
#      1. 在 EasyBuilder 工作表中新增一個整數格（例如 col=1, row=2）
#      2. 將格的值連結到 AcquireImage 的 Exposure 參數
#      3. 執行本工具時指定 --hw-cell 1 2
#      4. 使用 +/- 鍵調整曝光值（單位：微秒）
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

logger = logging.getLogger("adjust_exposure")

ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUT_CFG = PROJECT_ROOT / "configs" / "preprocess_tuned.yaml"

# ---------------------------------------------------------------------------
#  直方圖面板
# ---------------------------------------------------------------------------
def _make_histogram(img: np.ndarray, panel_w: int = 256, panel_h: int = 120) -> np.ndarray:
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # B G R
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [panel_w], [0, 256])
        cv2.normalize(hist, hist, 0, panel_h, cv2.NORM_MINMAX)
        pts = [(x, panel_h - int(hist[x])) for x in range(panel_w)]
        for j in range(len(pts) - 1):
            cv2.line(panel, pts[j], pts[j + 1], color, 1)
    # 中線（128）
    cv2.line(panel, (128, 0), (128, panel_h), (80, 80, 80), 1)
    cv2.putText(panel, "Histogram", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return panel


# ---------------------------------------------------------------------------
#  軟體影像調整
# ---------------------------------------------------------------------------
def _apply_software(img: np.ndarray, brightness: int, contrast: int,
                    clahe_on: bool, gamma_val: float) -> np.ndarray:
    out = img.copy()

    # Brightness / Contrast: alpha=contrast*0.01, beta=brightness-128
    alpha = contrast / 100.0
    beta  = brightness - 128
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    # CLAHE
    if clahe_on:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Gamma
    if abs(gamma_val - 1.0) > 0.01:
        inv_g = 1.0 / gamma_val
        lut = np.array(
            [((i / 255.0) ** inv_g) * 255 for i in range(256)], dtype=np.uint8
        )
        out = cv2.LUT(out, lut)

    return out


# ---------------------------------------------------------------------------
#  Cognex 曝光控制（Native Mode）
# ---------------------------------------------------------------------------
def _set_hw_exposure(sock, col: int, row: int, value: int) -> bool:
    """
    寫入 In-Sight 工作表指定格的整數值。
    命令格式：SI col row value\r\n
    回覆：1 = 成功
    """
    try:
        cmd = f"SI {col} {row} {value}\r\n"
        sock.sendall(cmd.encode("ascii"))
        reply = sock.recv(256).decode("ascii", errors="replace").strip()
        return reply.startswith("1")
    except Exception as e:
        logger.error("HW 曝光設定失敗: %s", e)
        return False


# ---------------------------------------------------------------------------
#  儲存參數
# ---------------------------------------------------------------------------
def _save_config(brightness: int, contrast: int, clahe_on: bool,
                 gamma_val: float) -> None:
    import yaml
    cfg = {
        "preprocess": {
            "enabled":       True,
            "undistort":     False,
            "clahe":         clahe_on,
            "clahe_clip":    2.0,
            "clahe_tile":    8,
            "gamma":         round(gamma_val, 2),
            "denoise_ksize": 0,
            "_sw_brightness": brightness,  # 僅供參考，不被 preprocess.py 使用
            "_sw_contrast":   contrast,
        }
    }
    OUTPUT_CFG.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CFG, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
    logger.info("參數已儲存: %s", OUTPUT_CFG)
    print(f"\n請將以下內容合併到 configs/site_B.yaml 的 preprocess 區段：")
    print(f"  clahe: {clahe_on}")
    print(f"  gamma: {round(gamma_val, 2)}")


# ---------------------------------------------------------------------------
#  主迴圈（有相機）
# ---------------------------------------------------------------------------
def run_live(args):
    import socket

    sec: dict = {}
    try:
        import yaml
        with open(PROJECT_ROOT / "configs" / "site_B.yaml", encoding="utf-8") as f:
            sec = (yaml.safe_load(f) or {}).get("cognex", {})
    except Exception:
        pass

    ip  = sec.get("ip",           "192.168.0.10")
    tp  = sec.get("telnet_port",  23)
    fp  = sec.get("ftp_port",     21)
    fu  = sec.get("ftp_user",     "admin")
    fpw = sec.get("ftp_password", "")

    from camera.cognex_stream import CognexCamera
    cam = CognexCamera(ip=ip, telnet_port=tp, ftp_port=fp,
                       ftp_user=fu, ftp_password=fpw, depth_mode="2D")

    # 軟體參數初始值
    brightness = 128   # 0~255，128 = 不調整
    contrast   = 100   # 50~200，100 = 不調整
    clahe_on   = False
    gamma_val  = 1.0
    hw_exp     = 5000  # 微秒（Cognex 曝光初始值）

    WIN_MAIN = "Exposure Adjustment"
    WIN_HIST = "Histogram"

    def _on_brightness(v): nonlocal brightness; brightness = v
    def _on_contrast(v):   nonlocal contrast;   contrast   = max(1, v)
    def _on_clahe(v):      nonlocal clahe_on;   clahe_on   = bool(v)
    def _on_gamma(v):      nonlocal gamma_val;  gamma_val  = max(0.1, v / 10.0)

    raw_frame = None

    try:
        cam.connect()
        raw_frame, _ = cam.get_frame()

        # 取得第一幀後才建立視窗與 trackbar，避免先出現空窗
        cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_HIST, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_HIST, 300, 130)
        cv2.createTrackbar("Brightness 0-255",  WIN_MAIN, brightness, 255,  _on_brightness)
        cv2.createTrackbar("Contrast  50-200",  WIN_MAIN, contrast,   200,  _on_contrast)
        cv2.createTrackbar("CLAHE 0/1",         WIN_MAIN, 0,          1,    _on_clahe)
        cv2.createTrackbar("Gamma x0.1 (10=1)", WIN_MAIN, 10,         30,   _on_gamma)

        while True:
            processed = _apply_software(raw_frame, brightness, contrast, clahe_on, gamma_val)

            display = processed.copy()
            cv2.putText(display,
                        f"brightness={brightness} contrast={contrast} "
                        f"CLAHE={'ON' if clahe_on else 'OFF'} gamma={gamma_val:.1f}",
                        (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
            hint = "r:刷新  s:儲存  q:離開"
            if args.hw_cell:
                hint += f"  +/-:HW曝光({hw_exp}µs)"
            cv2.putText(display, hint,
                        (8, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 50), 1)

            cv2.imshow(WIN_MAIN, display)
            cv2.imshow(WIN_HIST, _make_histogram(processed, 300, 130))

            key = cv2.waitKey(50) & 0xFF

            if key in (ord('q'), 27):
                break
            elif key == ord('r'):
                try:
                    raw_frame, _ = cam.get_frame()
                    logger.info("刷新畫面")
                except Exception as e:
                    logger.error("刷新失敗: %s", e)
            elif key == ord('s'):
                _save_config(brightness, contrast, clahe_on, gamma_val)

            # HW 曝光（僅在有 --hw-cell 時有效）
            elif key == ord('+') or key == 43:
                if args.hw_cell:
                    hw_exp = min(hw_exp + 500, 100000)
                    col, row = args.hw_cell
                    ok = _set_hw_exposure(cam._sock, col, row, hw_exp)
                    logger.info("HW 曝光 → %d µs (%s)", hw_exp, "OK" if ok else "FAIL")
            elif key == ord('-') or key == 45:
                if args.hw_cell:
                    hw_exp = max(hw_exp - 500, 100)
                    col, row = args.hw_cell
                    ok = _set_hw_exposure(cam._sock, col, row, hw_exp)
                    logger.info("HW 曝光 → %d µs (%s)", hw_exp, "OK" if ok else "FAIL")

    finally:
        cam.disconnect()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
#  離線模式（載入靜態圖）
# ---------------------------------------------------------------------------
def run_offline(args):
    img_path = Path(args.no_camera)
    raw_frame = cv2.imread(str(img_path))
    if raw_frame is None:
        logger.error("無法讀取影像: %s", img_path)
        sys.exit(1)

    brightness = 128
    contrast   = 100
    clahe_on   = False
    gamma_val  = 1.0

    WIN = "Exposure Adjustment (Offline)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    def _on_brightness(v): nonlocal brightness; brightness = v
    def _on_contrast(v):   nonlocal contrast;   contrast   = max(1, v)
    def _on_clahe(v):      nonlocal clahe_on;   clahe_on   = bool(v)
    def _on_gamma(v):      nonlocal gamma_val;  gamma_val  = max(0.1, v / 10.0)

    cv2.createTrackbar("Brightness",         WIN, brightness, 255, _on_brightness)
    cv2.createTrackbar("Contrast 50-200",    WIN, contrast,   200, _on_contrast)
    cv2.createTrackbar("CLAHE",              WIN, 0,          1,   _on_clahe)
    cv2.createTrackbar("Gamma x0.1",         WIN, 10,         30,  _on_gamma)

    while True:
        processed = _apply_software(raw_frame, brightness, contrast, clahe_on, gamma_val)
        cv2.imshow(WIN, processed)
        cv2.imshow("Histogram", _make_histogram(processed, 300, 130))

        key = cv2.waitKey(50) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            _save_config(brightness, contrast, clahe_on, gamma_val)

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="曝光度 / 影像品質調整工具")
    parser.add_argument("--hw-cell", nargs=2, type=int, metavar=("COL", "ROW"),
                        default=None,
                        help="Cognex In-Sight 曝光格位置（col row），例如 --hw-cell 1 2",
                        dest="hw_cell")
    parser.add_argument("--no-camera", default=None, metavar="IMG_PATH",
                        help="離線模式：指定靜態影像路徑")
    args = parser.parse_args()

    if args.no_camera:
        run_offline(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
