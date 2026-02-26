# =============================================================================
#  KR6 Pick & Place — 影像前處理管線
#
#  SKILL 用法：
#    from vision.preprocess import ImagePreprocessor
#
#    # 從 config 建立（訓練與推論用同一個實例）
#    pp = ImagePreprocessor.from_config(cfg.raw.get("preprocess", {}))
#    processed = pp.process(raw_img)
#
#  管線順序：
#    1. undistort   — 鏡頭畸變校正（需要 K.npy + D.npy）
#    2. crop        — ROI 裁切（不縮放，保持像素比例）
#    3. clahe       — 局部自適應對比增強（建議日光/LED 不均勻場景）
#    4. gamma       — 全域亮度校正（< 1.0 提亮暗部，> 1.0 壓暗高光）
#    5. denoise     — 高斯去噪（低噪感測器通常不需要）
#
#  重要原則：
#    訓練資料收集（collect_images.py）與推論（yolo_worker.py）
#    **必須使用相同的 ImagePreprocessor 配置**，否則模型精度下降。
# =============================================================================
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    可配置的影像前處理管線。

    Attributes:
        enabled (bool): False 時 process() 直接回傳原圖（快速關閉所有處理）
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        # 1. 鏡頭畸變校正
        undistort: bool = False,
        camera_K: Optional[np.ndarray] = None,
        camera_D: Optional[np.ndarray] = None,
        # 2. ROI 裁切 [x1, y1, x2, y2]（像素，未裁切前的座標）
        crop: Optional[list[int]] = None,
        # 3. CLAHE（局部自適應對比增強）
        clahe: bool = False,
        clahe_clip: float = 2.0,
        clahe_tile: int = 8,
        # 4. Gamma 校正（1.0 = 不處理）
        gamma: float = 1.0,
        # 5. 高斯去噪（0 = 不處理，>0 = kernel size，需奇數）
        denoise_ksize: int = 0,
    ):
        self.enabled = enabled

        # undistort
        self._undistort  = undistort and camera_K is not None and camera_D is not None
        self._K  = camera_K
        self._D  = camera_D
        self._map1: Optional[np.ndarray] = None
        self._map2: Optional[np.ndarray] = None  # 預計算 remap 表

        # crop
        self._crop = crop  # [x1, y1, x2, y2] or None

        # CLAHE
        self._clahe_obj = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=(clahe_tile, clahe_tile),
        ) if clahe else None

        # gamma
        self._gamma_lut: Optional[np.ndarray] = None
        if abs(gamma - 1.0) > 1e-4:
            inv_gamma = 1.0 / gamma
            self._gamma_lut = np.array(
                [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
                dtype=np.uint8,
            )

        # denoise
        self._denoise_ksize = denoise_ksize if denoise_ksize > 0 and denoise_ksize % 2 == 1 else 0

        logger.debug(
            "ImagePreprocessor: undistort=%s crop=%s clahe=%s gamma=%.2f denoise=%d",
            self._undistort, crop, clahe, gamma, self._denoise_ksize,
        )

    # ------------------------------------------------------------------
    #  主要 API
    # ------------------------------------------------------------------
    def process(self, img: np.ndarray) -> np.ndarray:
        """
        依序套用所有啟用的前處理步驟。

        Args:
            img: BGR uint8 影像

        Returns:
            處理後的 BGR uint8 影像（若 enabled=False 直接回傳原圖）
        """
        if not self.enabled:
            return img

        # 1. 鏡頭畸變校正
        if self._undistort:
            img = self._apply_undistort(img)

        # 2. ROI 裁切
        if self._crop is not None:
            x1, y1, x2, y2 = self._crop
            h, w = img.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 > x1 and y2 > y1:
                img = img[y1:y2, x1:x2]

        # 3. CLAHE（在 L 通道做，保留色彩）
        if self._clahe_obj is not None:
            lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self._clahe_obj.apply(l)
            img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        # 4. Gamma 校正
        if self._gamma_lut is not None:
            img = cv2.LUT(img, self._gamma_lut)

        # 5. 高斯去噪
        if self._denoise_ksize > 0:
            img = cv2.GaussianBlur(img, (self._denoise_ksize, self._denoise_ksize), 0)

        return img

    # ------------------------------------------------------------------
    #  Undistort 預計算
    # ------------------------------------------------------------------
    def prepare_undistort(self, img_w: int, img_h: int) -> None:
        """
        預計算 remap 映射表（提升推論速度）。
        在首次 get_frame() 後呼叫。
        """
        if not self._undistort or self._K is None:
            return
        new_K, _ = cv2.getOptimalNewCameraMatrix(
            self._K, self._D, (img_w, img_h), alpha=0,
        )
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self._K, self._D, None, new_K, (img_w, img_h), cv2.CV_32FC1,
        )
        logger.info("undistort remap 預計算完成: %dx%d", img_w, img_h)

    def _apply_undistort(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if self._map1 is None:
            self.prepare_undistort(w, h)
        if self._map1 is not None:
            return cv2.remap(img, self._map1, self._map2, cv2.INTER_LINEAR)
        return cv2.undistort(img, self._K, self._D)

    # ------------------------------------------------------------------
    #  工廠方法：從 YAML config dict 建立
    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        cfg: dict,
        assets_dir: str | Path = "assets",
    ) -> "ImagePreprocessor":
        """
        從 YAML preprocess 區段建立 ImagePreprocessor。

        YAML 範例::

            preprocess:
              enabled: true
              undistort: true          # 需要 assets/camera_K.npy + camera_D.npy
              crop: [0, 0, 1600, 1200] # ROI [x1,y1,x2,y2]，null 表示不裁切
              clahe: true
              clahe_clip: 2.0
              clahe_tile: 8
              gamma: 1.0              # 1.0 = 不調整
              denoise_ksize: 0        # 0 = 不去噪，需奇數如 3/5/7
        """
        assets = Path(assets_dir)
        K = D = None
        if cfg.get("undistort", False):
            k_path = assets / "camera_K.npy"
            d_path = assets / "camera_D.npy"
            if k_path.exists() and d_path.exists():
                K = np.load(str(k_path))
                D = np.load(str(d_path))
                logger.info("載入相機內參: K=%s", k_path)
            else:
                logger.warning("undistort=true 但找不到 K.npy / D.npy，跳過畸變校正")

        return cls(
            enabled       = cfg.get("enabled",       True),
            undistort     = cfg.get("undistort",     False),
            camera_K      = K,
            camera_D      = D,
            crop          = cfg.get("crop",          None),
            clahe         = cfg.get("clahe",         False),
            clahe_clip    = cfg.get("clahe_clip",    2.0),
            clahe_tile    = cfg.get("clahe_tile",    8),
            gamma         = cfg.get("gamma",         1.0),
            denoise_ksize = cfg.get("denoise_ksize", 0),
        )

    # ------------------------------------------------------------------
    #  描述
    # ------------------------------------------------------------------
    def describe(self) -> dict:
        """回傳目前管線設定摘要"""
        return {
            "enabled":   self.enabled,
            "undistort": self._undistort,
            "crop":      self._crop,
            "clahe":     self._clahe_obj is not None,
            "gamma_lut": self._gamma_lut is not None,
            "denoise":   self._denoise_ksize,
        }
