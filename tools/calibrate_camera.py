# =============================================================================
#  KR6 Pick & Place — 棋盤格相機內參校正工具
#
#  功能：
#    - 從 Cognex 即時拍攝棋盤格影像
#    - 用 OpenCV findChessboardCorners 找角點
#    - calibrateCamera 計算內參矩陣 K 與畸變係數 D
#    - 儲存 assets/camera_K.npy + assets/camera_D.npy
#    - 報告重投影誤差（< 1.0 pixel 為佳）
#
#  Usage:
#      python tools/calibrate_camera.py               # 預設 9×6 棋盤格
#      python tools/calibrate_camera.py --cols 8 --rows 5
#      python tools/calibrate_camera.py --from-dir calibration_imgs/
#
#  操作（即時拍攝模式）：
#      Space / s  → 嘗試偵測棋盤格並收集（自動顯示角點）
#      d          → 刪除最後一張
#      c          → 開始計算（需 >= 10 張）
#      q / Esc    → 離開
#
#  棋盤格規格：
#      本工具使用「內角點數」（例如 10×7 的棋盤格 = 9×6 內角點）
#      建議：> 15 張、覆蓋視野四角與中心、各種傾斜角度
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

from camera.cognex_stream import CognexCamera

logger = logging.getLogger("calibrate_camera")

ASSETS_DIR = PROJECT_ROOT / "assets"
K_PATH     = ASSETS_DIR / "camera_K.npy"
D_PATH     = ASSETS_DIR / "camera_D.npy"

# 棋盤格每格實際尺寸（mm）— 影響重投影誤差單位，不影響校正準確性
_SQUARE_MM = 25.0


# ---------------------------------------------------------------------------
#  建立物件點（3D 棋盤格角點，Z=0 平面）
# ---------------------------------------------------------------------------
def _make_objp(cols: int, rows: int) -> np.ndarray:
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * _SQUARE_MM
    return objp


# ---------------------------------------------------------------------------
#  偵測棋盤格
# ---------------------------------------------------------------------------
def _detect(img: np.ndarray, cols: int, rows: int):
    """
    回傳 (corners, gray) 或 (None, gray)。
    corners: shape (N,1,2) float32
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
             cv2.CALIB_CB_NORMALIZE_IMAGE |
             cv2.CALIB_CB_FAST_CHECK)
    found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if not found:
        return None, gray

    # 亞像素精化
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners  = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners, gray


# ---------------------------------------------------------------------------
#  載入既有影像資料夾
# ---------------------------------------------------------------------------
def _load_from_dir(img_dir: Path, cols: int, rows: int):
    objp   = _make_objp(cols, rows)
    obj_pts, img_pts = [], []
    img_size = None
    valid = 0

    for jpg in sorted(img_dir.glob("*.jpg")):
        img  = cv2.imread(str(jpg))
        if img is None:
            continue
        corners, gray = _detect(img, cols, rows)
        if corners is None:
            logger.warning("找不到棋盤格: %s", jpg.name)
            continue
        obj_pts.append(objp)
        img_pts.append(corners)
        img_size = (gray.shape[1], gray.shape[0])
        valid += 1
        logger.info("  [OK] %s", jpg.name)

    return obj_pts, img_pts, img_size, valid


# ---------------------------------------------------------------------------
#  執行校正
# ---------------------------------------------------------------------------
def _calibrate(obj_pts, img_pts, img_size) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns: (K, D, mean_reprojection_error_px)
    """
    flags = 0
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, img_size, None, None, flags=flags,
    )

    # 計算平均重投影誤差
    total_err = 0.0
    for i in range(len(obj_pts)):
        proj, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], K, D)
        total_err += cv2.norm(img_pts[i], proj, cv2.NORM_L2) / len(proj)
    mean_err = total_err / len(obj_pts)

    return K, D.flatten(), mean_err


# ---------------------------------------------------------------------------
#  儲存結果
# ---------------------------------------------------------------------------
def _save(K: np.ndarray, D: np.ndarray, mean_err: float) -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(K_PATH), K)
    np.save(str(D_PATH), D)
    logger.info("K 矩陣已儲存: %s", K_PATH)
    logger.info("D 係數已儲存: %s", D_PATH)
    logger.info("平均重投影誤差: %.4f px %s",
                mean_err, "✓ (< 1.0)" if mean_err < 1.0 else "⚠ (> 1.0，建議重拍)")
    print("\n=== 校正結果 ===")
    print(f"K =\n{K}")
    print(f"D = {D}")
    print(f"重投影誤差: {mean_err:.4f} px")


# ---------------------------------------------------------------------------
#  Cognex 即時拍攝模式
# ---------------------------------------------------------------------------
def _load_cognex_params() -> dict:
    try:
        import yaml
        with open(PROJECT_ROOT / "configs" / "site_B.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("cognex", {})
    except Exception:
        return {}


def run_live(args):
    sec = _load_cognex_params()
    cam = CognexCamera(
        ip=sec.get("ip", "192.168.0.10"),
        telnet_port=sec.get("telnet_port", 23),
        ftp_port=sec.get("ftp_port", 21),
        ftp_user=sec.get("ftp_user", "admin"),
        ftp_password=sec.get("ftp_password", ""),
        depth_mode="2D",
    )

    cols, rows = args.cols, args.rows
    objp       = _make_objp(cols, rows)
    obj_pts, img_pts = [], []
    img_size   = None
    MIN_FRAMES = args.min_frames
    WIN        = "Camera Calibration"

    try:
        cam.connect()
        logger.info("連線成功，開始校正（棋盤格 %d×%d 內角點）", cols, rows)

        current_frame, _ = cam.get_frame()
        last_corners = None

        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

        while True:
            display = current_frame.copy()

            # 顯示上次偵測結果
            if last_corners is not None:
                cv2.drawChessboardCorners(display, (cols, rows), last_corners, True)

            n = len(obj_pts)
            color = (0, 200, 0) if n >= MIN_FRAMES else (0, 140, 255)
            cv2.putText(display,
                        f"收集: {n}/{MIN_FRAMES}  Space:拍攝  d:刪除  c:計算  q:離開",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.imshow(WIN, display)

            key = cv2.waitKey(30) & 0xFF

            if key in (ord('q'), 27):
                break

            elif key in (ord(' '), ord('s')):
                # 拍一張 + 偵測
                try:
                    frame, _ = cam.get_frame()
                    corners, gray = _detect(frame, cols, rows)
                    if corners is None:
                        logger.warning("未偵測到棋盤格，請調整角度後重試")
                        # 閃紅框
                        flash = frame.copy()
                        cv2.putText(flash, "NOT FOUND", (10, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        cv2.imshow(WIN, flash)
                        cv2.waitKey(600)
                    else:
                        obj_pts.append(objp)
                        img_pts.append(corners)
                        img_size = (gray.shape[1], gray.shape[0])
                        last_corners  = corners
                        current_frame = frame
                        logger.info("[%d] 棋盤格偵測成功", len(obj_pts))
                except Exception as e:
                    logger.error("拍攝失敗: %s", e)

            elif key == ord('d'):
                if obj_pts:
                    obj_pts.pop()
                    img_pts.pop()
                    last_corners = None
                    logger.info("已刪除最後一張，剩 %d 張", len(obj_pts))

            elif key == ord('c'):
                if len(obj_pts) < MIN_FRAMES:
                    logger.warning("需要至少 %d 張（目前 %d）", MIN_FRAMES, len(obj_pts))
                    continue
                logger.info("開始計算校正矩陣...")
                K, D, mean_err = _calibrate(obj_pts, img_pts, img_size)
                _save(K, D, mean_err)
                break

    finally:
        cam.disconnect()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
#  離線資料夾模式
# ---------------------------------------------------------------------------
def run_offline(args):
    img_dir = Path(args.from_dir)
    if not img_dir.exists():
        logger.error("目錄不存在: %s", img_dir)
        sys.exit(1)

    logger.info("離線模式: %s", img_dir)
    obj_pts, img_pts, img_size, valid = _load_from_dir(img_dir, args.cols, args.rows)

    if valid < args.min_frames:
        logger.error("有效影像不足（%d < %d）", valid, args.min_frames)
        sys.exit(1)

    logger.info("共 %d 張有效影像，開始計算...", valid)
    K, D, mean_err = _calibrate(obj_pts, img_pts, img_size)
    _save(K, D, mean_err)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="棋盤格相機內參校正工具")
    parser.add_argument("--cols",       default=9,   type=int, help="棋盤格內角點列數（預設 9）")
    parser.add_argument("--rows",       default=6,   type=int, help="棋盤格內角點行數（預設 6）")
    parser.add_argument("--min-frames", default=10,  type=int, help="最少需要幾張（預設 10）",
                        dest="min_frames")
    parser.add_argument("--from-dir",   default=None,          help="離線模式：從目錄載入影像",
                        dest="from_dir")
    args = parser.parse_args()

    if args.from_dir:
        run_offline(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
