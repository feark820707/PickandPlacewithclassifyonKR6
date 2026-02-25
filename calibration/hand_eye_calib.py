# =============================================================================
#  KR6 Pick & Place — 手眼標定
#  產生 Homography 矩陣 H.npy + 量測鏡頭−吸盤偏移 OFFSET_X/Y
# =============================================================================
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def collect_calibration_points(
    image_points: list[tuple[float, float]],
    robot_points: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    整理標定點對。

    Args:
        image_points: 影像座標 [(px_x, px_y), ...]
        robot_points: 對應的機器人基座標 [(rob_x, rob_y), ...]

    Returns:
        (img_pts, rob_pts): 兩個 np.ndarray, shape (N, 2), dtype float64

    Raises:
        ValueError: 點數不足（至少需 4 組）或數量不匹配
    """
    if len(image_points) != len(robot_points):
        raise ValueError(
            f"影像點數 ({len(image_points)}) 與機器人點數 "
            f"({len(robot_points)}) 不匹配"
        )
    if len(image_points) < 4:
        raise ValueError(
            f"至少需要 4 組標定點，目前只有 {len(image_points)} 組"
        )

    img_pts = np.array(image_points, dtype=np.float64)
    rob_pts = np.array(robot_points, dtype=np.float64)
    return img_pts, rob_pts


def compute_homography(
    image_points: np.ndarray,
    robot_points: np.ndarray,
) -> np.ndarray:
    """
    計算 Homography 矩陣 H：像素座標 → 機器人基座標。

    H 矩陣使得：
        [rob_x, rob_y, 1]^T = H @ [px_x, px_y, 1]^T

    Args:
        image_points: shape (N, 2), 像素座標
        robot_points: shape (N, 2), 機器人座標

    Returns:
        H: 3×3 Homography 矩陣

    Raises:
        RuntimeError: findHomography 失敗
    """
    H, mask = cv2.findHomography(image_points, robot_points, cv2.RANSAC, 5.0)

    if H is None:
        raise RuntimeError(
            "Homography 計算失敗，請確認標定點是否正確"
        )

    # 計算重投影誤差
    inliers = int(mask.sum())
    total = len(mask)
    logger.info(
        "Homography 計算完成: inliers=%d/%d",
        inliers, total,
    )

    # 逐點重投影誤差
    errors = []
    for i in range(len(image_points)):
        px = np.array([image_points[i][0], image_points[i][1], 1.0])
        predicted = H @ px
        predicted = predicted[:2] / predicted[2]
        actual = robot_points[i]
        err = np.linalg.norm(predicted - actual)
        errors.append(err)

    mean_err = np.mean(errors)
    max_err = np.max(errors)
    logger.info(
        "重投影誤差: mean=%.3f mm, max=%.3f mm",
        mean_err, max_err,
    )

    if mean_err > 5.0:
        logger.warning(
            "重投影誤差偏高 (%.3f mm)，建議重新標定", mean_err,
        )

    return H


def save_homography(H: np.ndarray, path: str = "assets/H.npy") -> None:
    """儲存 Homography 矩陣"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, H)
    logger.info("Homography 矩陣已儲存至 %s", path)


def load_homography(path: str = "assets/H.npy") -> np.ndarray:
    """載入 Homography 矩陣"""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Homography 矩陣不存在: {path}。請先執行手眼標定。"
        )
    H = np.load(path)
    logger.info("Homography 矩陣已載入: %s", path)
    return H


def pixel_to_robot(
    px_x: float,
    px_y: float,
    H: np.ndarray,
) -> tuple[float, float]:
    """
    像素座標 → 機器人基座標（XY）。

    Args:
        px_x, px_y: 像素座標
        H: 3×3 Homography 矩陣

    Returns:
        (robot_x, robot_y)
    """
    px = np.array([px_x, px_y, 1.0])
    rob = H @ px
    rob_x = rob[0] / rob[2]
    rob_y = rob[1] / rob[2]
    return float(rob_x), float(rob_y)


def measure_offset(
    camera_point_robot: tuple[float, float],
    suction_point_robot: tuple[float, float],
) -> tuple[float, float]:
    """
    量測鏡頭−吸盤偏移量。

    步驟：
    1. 用相機拍照，辨識標定板上已知點 P → 透過 H 轉換得機器人座標
    2. 移動機器人讓吸盤中心對準 P → 記錄機器人座標
    3. 差值即為 (OFFSET_X, OFFSET_Y)

    Args:
        camera_point_robot: 相機辨識出的機器人座標 (x, y)
        suction_point_robot: 吸盤對準 P 時的機器人座標 (x, y)

    Returns:
        (offset_x, offset_y)
    """
    offset_x = suction_point_robot[0] - camera_point_robot[0]
    offset_y = suction_point_robot[1] - camera_point_robot[1]

    logger.info(
        "鏡頭−吸盤偏移量測: OFFSET_X=%.2f mm, OFFSET_Y=%.2f mm",
        offset_x, offset_y,
    )

    return offset_x, offset_y


def run_calibration_interactive(camera, num_points: int = 9):
    """
    互動式標定流程（命令列引導）。

    Args:
        camera: CameraBase 實例（已連線）
        num_points: 標定點數量（建議 ≥ 9）
    """
    print(f"\n{'='*60}")
    print(f"  手眼標定流程 — 共 {num_points} 個標定點")
    print(f"{'='*60}\n")

    image_points = []
    robot_points = []

    for i in range(num_points):
        print(f"\n--- 標定點 {i+1}/{num_points} ---")

        # 1. 擷取影像
        rgb, _ = camera.get_frame()

        # 2. 顯示影像，讓使用者點擊標定點
        print("請在影像視窗中點擊標定點（按 Enter 確認）...")

        clicked = []

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked.clear()
                clicked.append((x, y))
                img_copy = rgb.copy()
                cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow("Calibration", img_copy)

        cv2.imshow("Calibration", rgb)
        cv2.setMouseCallback("Calibration", on_click)
        cv2.waitKey(0)

        if not clicked:
            print("未選擇點，跳過")
            continue

        px_x, px_y = clicked[0]
        image_points.append((px_x, px_y))
        print(f"  影像座標: ({px_x}, {px_y})")

        # 3. 輸入機器人座標
        rob_x = float(input("  請輸入機器人 X (mm): "))
        rob_y = float(input("  請輸入機器人 Y (mm): "))
        robot_points.append((rob_x, rob_y))

    cv2.destroyAllWindows()

    if len(image_points) < 4:
        print(f"\n標定點不足（{len(image_points)}），需至少 4 組。")
        return None

    # 4. 計算 Homography
    img_pts, rob_pts = collect_calibration_points(image_points, robot_points)
    H = compute_homography(img_pts, rob_pts)

    # 5. 儲存
    save_homography(H)
    print(f"\n✅ 標定完成！Homography 矩陣已儲存至 assets/H.npy")

    return H
