# =============================================================================
#  KR6 Pick & Place — 手眼校正工具（cam center → tool center，6DOF）
#
#  計算相機光軸中心點 到 工具中心點（TCP）的 6DOF 剛體變換：
#    T_cam2tool = [tx, ty, tz, rx, ry, rz]  (mm, mm, mm, deg, deg, deg)
#
#  各分量意義：
#    tx, ty  — 相機光軸在機器人 XY 平面上與 TCP 的水平偏移 (mm)
#    tz      — 相機到 TCP 的垂直距離 (mm，通常 = 相機高度 - 工具長度)
#    rx, ry  — 相機傾斜角（鏡頭非完全朝下時使用，通常 ≈ 0）
#    rz      — 影像座標系相對機器人座標系的旋轉角 (deg)，從 H 矩陣自動計算
#
#  流程：
#    Step 1  自動從 H.npy 計算 rz（影像 X 軸 vs 機器人 X 軸夾角）
#    Step 2  互動量測 tx, ty（相機看到一點 P，工具移到 P，記差值）
#    Step 3  輸入 tz（相機安裝高度 - 工具長度，或直接量測）
#    Step 4  輸入 rx, ry（通常填 0，棋盤格量測後填入）
#    Step 5  儲存 assets/T_cam2tool.npy + 更新 site_B.yaml
#
#  Usage:
#      python tools/calibrate_hand_eye.py
#      python tools/calibrate_hand_eye.py --points 5   # 量測 5 組點取平均
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

logger = logging.getLogger("calibrate_hand_eye")

ASSETS_DIR    = PROJECT_ROOT / "assets"
H_PATH        = ASSETS_DIR   / "H.npy"
T_CAM2TOOL    = ASSETS_DIR   / "T_cam2tool.npy"
SITE_B_YAML   = PROJECT_ROOT / "configs" / "site_B.yaml"


# ---------------------------------------------------------------------------
#  Step 1：從 Homography 提取 rz
# ---------------------------------------------------------------------------
def _extract_rz_from_H(H: np.ndarray, img_w: int, img_h: int) -> float:
    """
    將影像中心與右側點投影到機器人座標，
    計算影像 X 軸方向相對機器人 X 軸的旋轉角 rz (度)。
    """
    cx, cy = img_w / 2.0, img_h / 2.0
    p1_px = np.array([cx,        cy, 1.0])
    p2_px = np.array([cx + 100,  cy, 1.0])

    def proj(p):
        w = H @ p
        return w[0] / w[2], w[1] / w[2]

    wx1, wy1 = proj(p1_px)
    wx2, wy2 = proj(p2_px)
    rz = math.degrees(math.atan2(wy2 - wy1, wx2 - wx1))
    return round(rz, 4)


# ---------------------------------------------------------------------------
#  Step 2：互動量測 tx, ty
# ---------------------------------------------------------------------------
def _measure_offset_interactive(
    cam,
    H: np.ndarray,
    n_points: int,
) -> tuple[float, float]:
    """
    互動式量測：拍照 → 點擊影像中心（光軸）→ 輸入工具到達同一點的機器人座標。
    重複 n_points 次後取平均。
    """
    def pixel_to_world(px, py):
        src = H @ np.array([px, py, 1.0])
        return src[0] / src[2], src[1] / src[2]

    offsets_x, offsets_y = [], []
    WIN = "Hand-Eye: Click image center of calibration point"

    print(f"\n{'─'*60}")
    print(f"  Step 2：量測 tx / ty  （共 {n_points} 組）")
    print("  流程：在工作台上放一個標記點 P")
    print("        1. 拍照後點擊 P 的影像位置")
    print("        2. 移動機器人讓 TCP 對準 P")
    print("        3. 輸入此時機器人 X Y 座標")
    print(f"{'─'*60}\n")

    clicked = []

    def _cb(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.clear()
            clicked.append((x, y))

    for i in range(n_points):
        print(f"\n  --- 第 {i+1}/{n_points} 組 ---")
        frame, _ = cam.get_frame()
        clicked.clear()

        # 第一幀取得後才建立視窗（僅第一次迴圈有效）
        if i == 0:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(WIN, _cb)

        while True:
            disp = frame.copy()
            h, w = disp.shape[:2]
            # 準星
            cv2.line(disp, (w//2 - 30, h//2), (w//2 + 30, h//2), (0, 255, 255), 1)
            cv2.line(disp, (w//2, h//2 - 30), (w//2, h//2 + 30), (0, 255, 255), 1)
            if clicked:
                cx, cy_ = clicked[0]
                cv2.circle(disp, (cx, cy_), 6, (0, 0, 255), -1)
                cv2.putText(disp, f"P=({cx},{cy_})", (cx + 8, cy_ - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(disp,
                        f"[{i+1}/{n_points}] 左鍵點擊標記點 P，Enter 確認，r 重新拍照",
                        (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
            cv2.imshow(WIN, disp)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('r'):
                frame, _ = cam.get_frame()
                clicked.clear()
            elif key == 13 and clicked:   # Enter 確認
                break
            elif key in (ord('q'), 27):
                cv2.destroyWindow(WIN)
                raise KeyboardInterrupt("使用者取消")

        px_x, px_y = clicked[0]
        cam_wx, cam_wy = pixel_to_world(px_x, px_y)
        print(f"  相機看到 P 的機器人座標: ({cam_wx:.2f}, {cam_wy:.2f}) mm")

        rob_x = float(input("  TCP 對準 P 時的機器人 X (mm): "))
        rob_y = float(input("  TCP 對準 P 時的機器人 Y (mm): "))

        dx = rob_x - cam_wx
        dy = rob_y - cam_wy
        offsets_x.append(dx)
        offsets_y.append(dy)
        print(f"  本次偏移: tx={dx:.3f} mm  ty={dy:.3f} mm")

    cv2.destroyWindow(WIN)
    tx = float(np.mean(offsets_x))
    ty = float(np.mean(offsets_y))
    std_x = float(np.std(offsets_x))
    std_y = float(np.std(offsets_y))
    print(f"\n  平均 tx={tx:.3f} mm (±{std_x:.3f})  ty={ty:.3f} mm (±{std_y:.3f})")
    return tx, ty


# ---------------------------------------------------------------------------
#  儲存結果
# ---------------------------------------------------------------------------
def _save_result(T: np.ndarray) -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(T_CAM2TOOL), T)
    logger.info("T_cam2tool 已儲存: %s", T_CAM2TOOL)

    tx, ty, tz, rx, ry, rz = T
    print(f"\n{'═'*55}")
    print(f"  T_cam2tool = [tx, ty, tz, rx, ry, rz]")
    print(f"  tx={tx:.3f} mm  ty={ty:.3f} mm  tz={tz:.3f} mm")
    print(f"  rx={rx:.3f}°    ry={ry:.3f}°    rz={rz:.3f}°")
    print(f"{'═'*55}")

    # 更新 site_B.yaml
    _patch_site_yaml(tx, ty, tz, rx, ry, rz)


def _patch_site_yaml(tx, ty, tz, rx, ry, rz) -> None:
    """將量測結果寫回 site_B.yaml 的 mechanical 區段"""
    try:
        import yaml
        with open(SITE_B_YAML, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        mech = cfg.setdefault("mechanical", {})
        mech["offset_x"] = round(tx, 4)
        mech["offset_y"] = round(ty, 4)
        # 額外儲存完整 6DOF（供參考）
        cam2tool = cfg.setdefault("cam2tool", {})
        cam2tool["tx"] = round(tx, 4)
        cam2tool["ty"] = round(ty, 4)
        cam2tool["tz"] = round(tz, 4)
        cam2tool["rx"] = round(rx, 4)
        cam2tool["ry"] = round(ry, 4)
        cam2tool["rz"] = round(rz, 4)

        with open(SITE_B_YAML, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
        logger.info("site_B.yaml 已更新: offset_x=%.3f  offset_y=%.3f", tx, ty)
    except Exception as e:
        logger.warning("無法更新 site_B.yaml: %s  （請手動填入）", e)
        print(f"\n  請手動在 site_B.yaml 填入：")
        print(f"    mechanical.offset_x: {tx:.4f}")
        print(f"    mechanical.offset_y: {ty:.4f}")


# ---------------------------------------------------------------------------
#  主流程
# ---------------------------------------------------------------------------
def run(args):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # 載入 H.npy
    if not H_PATH.exists():
        logger.error("H.npy 不存在 (%s)，請先執行 main.py --calibrate hand-eye", H_PATH)
        sys.exit(1)
    H = np.load(str(H_PATH))
    logger.info("H.npy 已載入")

    # 連線 Cognex（取得影像尺寸）
    sec: dict = {}
    try:
        import yaml
        with open(SITE_B_YAML, encoding="utf-8") as f:
            sec = (yaml.safe_load(f) or {}).get("cognex", {})
    except Exception:
        pass

    from camera.cognex_stream import CognexCamera
    cam = CognexCamera(
        ip=sec.get("ip", "192.168.0.10"),
        telnet_port=sec.get("telnet_port", 23),
        ftp_port=sec.get("ftp_port", 21),
        ftp_user=sec.get("ftp_user", "admin"),
        ftp_password=sec.get("ftp_password", ""),
        depth_mode="2D",
    )

    try:
        cam.connect()
        frame, _ = cam.get_frame()
        img_h, img_w = frame.shape[:2]

        # ── Step 1：從 H 計算 rz ───────────────────────────────────────
        rz = _extract_rz_from_H(H, img_w, img_h)
        print(f"\n  Step 1 完成：rz（影像→機器人旋轉角）= {rz:.4f}°")

        # ── Step 2：互動量測 tx, ty ────────────────────────────────────
        tx, ty = _measure_offset_interactive(cam, H, args.points)

        # ── Step 3：輸入 tz ────────────────────────────────────────────
        print(f"\n  Step 3：輸入 tz")
        print(f"    tz = 相機安裝高度 - 末端工具長度")
        print(f"    （可從 configs 讀取：camera_height_mm - suction_length_mm）")
        tz_default = 800.0 - 80.0  # 預設值
        tz_str = input(f"    請輸入 tz (mm) [預設 {tz_default}]: ").strip()
        tz = float(tz_str) if tz_str else tz_default

        # ── Step 4：輸入 rx, ry ────────────────────────────────────────
        print(f"\n  Step 4：輸入 rx, ry（相機傾斜角，通常為 0）")
        rx_str = input(f"    rx (deg) [預設 0.0]: ").strip()
        ry_str = input(f"    ry (deg) [預設 0.0]: ").strip()
        rx = float(rx_str) if rx_str else 0.0
        ry = float(ry_str) if ry_str else 0.0

        # ── Step 5：儲存 ───────────────────────────────────────────────
        T = np.array([tx, ty, tz, rx, ry, rz], dtype=np.float64)
        _save_result(T)

    except KeyboardInterrupt:
        logger.info("使用者取消")
    finally:
        cam.disconnect()


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="手眼校正工具：計算 cam center → TCP 的 6DOF 變換",
    )
    parser.add_argument(
        "--points", default=3, type=int,
        help="量測點組數（越多越準，預設 3）",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
