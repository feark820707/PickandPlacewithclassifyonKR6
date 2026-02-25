# =============================================================================
#  KR6 Pick & Place — 相機安裝高度量測
#  用於 DEPTH_MODE="3D" 時的 Z 軸計算基準
# =============================================================================
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def measure_camera_height(
    depth_map: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
) -> float:
    """
    從深度圖量測相機到工作台面的距離。

    方法：在空曠工作台面上擷取深度圖，取 ROI 區域的中位數深度值。

    Args:
        depth_map: 深度圖, shape (H, W), dtype float32, 單位 mm
        roi: (x, y, w, h) 感興趣區域。None=使用中心 1/4 區域

    Returns:
        相機高度 (mm)

    Raises:
        ValueError: 深度值異常（NaN 或 <= 0）
    """
    h, w = depth_map.shape[:2]

    if roi is None:
        # 使用中心 1/4 區域
        cx, cy = w // 2, h // 2
        qw, qh = w // 4, h // 4
        roi = (cx - qw // 2, cy - qh // 2, qw, qh)

    x, y, rw, rh = roi
    region = depth_map[y:y+rh, x:x+rw]

    # 過濾無效值
    valid = region[(region > 0) & (~np.isnan(region)) & (region < 5000)]

    if len(valid) < 100:
        raise ValueError(
            f"有效深度點不足 ({len(valid)})，請確認工作台面在視野內"
        )

    height_mm = float(np.median(valid))

    # 統計資訊
    std_mm = float(np.std(valid))
    logger.info(
        "相機高度量測: median=%.1f mm, std=%.2f mm, valid_points=%d",
        height_mm, std_mm, len(valid),
    )

    if std_mm > 10.0:
        logger.warning(
            "深度標準差偏高 (%.2f mm)，台面可能不平或有物件", std_mm,
        )

    return height_mm


def validate_camera_height(
    measured: float,
    configured: float,
    tolerance: float = 20.0,
) -> bool:
    """
    驗證量測的相機高度與設定值是否一致。

    Args:
        measured: 量測值 (mm)
        configured: 設定檔中的值 (mm)
        tolerance: 允許誤差 (mm)

    Returns:
        True=一致, False=差距過大
    """
    diff = abs(measured - configured)
    if diff > tolerance:
        logger.error(
            "相機高度不一致: 量測=%.1f mm, 設定=%.1f mm, 差距=%.1f mm (閾值=%.1f)",
            measured, configured, diff, tolerance,
        )
        return False

    logger.info(
        "相機高度驗證通過: 量測=%.1f mm, 設定=%.1f mm, 差距=%.1f mm",
        measured, configured, diff,
    )
    return True
