# =============================================================================
#  KR6 Pick & Place — 本地檔案影像來源（FileCamera 插件）
#
#  實作 CameraBase 介面，以本地圖檔或目錄模擬相機，
#  讓所有工具可在無實體相機時仍正常執行。
#
#  支援：
#    - 單一圖檔：每次 get_frame() 回傳同一張（適合靜態測試）
#    - 目錄：依字母順序逐張讀取，可循環或單次播放
#
#  使用範例（YAML configs）：
#    camera_type: file
#    file:
#      path: datasets/raw/classA/
#      loop: true
#
#  或直接以 factory 建立：
#    from camera.file_source import FileCamera
#    cam = FileCamera(path="img.jpg")
#
#  Register name: "file"
# =============================================================================
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from camera.base import CameraBase, register_camera

logger = logging.getLogger(__name__)

_IMG_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"})


@register_camera(
    "file",
    supported_depth_modes={"2D"},
    required_config_keys=("file.path",),
    description="本地影像檔案來源（單圖 / 目錄輪播）",
)
class FileCamera(CameraBase):
    """
    本地影像來源插件。

    Args:
        path:       圖檔路徑（str 或 Path），或影像目錄路徑。
        loop:       目錄模式下，讀完後是否循環回第一張（預設 True）。
        depth_mode: 固定為 "2D"（無深度資訊）。
    """

    def __init__(
        self,
        path: str | Path,
        loop: bool = True,
        depth_mode: str = "2D",
    ):
        super().__init__(depth_mode="2D")   # FileCamera 永遠 2D
        self._root     = Path(path)
        self._loop     = loop
        self._files:  list[Path] = []
        self._index:  int        = 0
        self._single: Optional[np.ndarray] = None   # 單圖模式快取

    # ------------------------------------------------------------------
    #  CameraBase 介面實作
    # ------------------------------------------------------------------
    def connect(self) -> None:
        if self._root.is_file():
            img = cv2.imdecode(np.fromfile(str(self._root), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ConnectionError(f"無法讀取影像: {self._root}")
            self._single = img
            self._files  = [self._root]
            logger.info("FileCamera 單圖模式: %s", self._root.name)

        elif self._root.is_dir():
            self._files = sorted(
                p for p in self._root.iterdir()
                if p.suffix.lower() in _IMG_EXTS
            )
            if not self._files:
                raise ConnectionError(f"目錄中無支援的影像檔: {self._root}")
            logger.info(
                "FileCamera 目錄模式: %d 張影像  loop=%s  (%s)",
                len(self._files), self._loop, self._root,
            )
        else:
            raise ConnectionError(f"路徑不存在: {self._root}")

        self._index     = 0
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False
        self._single    = None
        logger.info("FileCamera 已關閉")

    def _capture(self) -> tuple[np.ndarray, None]:
        # 單圖模式：永遠回傳同一張
        if self._single is not None:
            return self._single.copy(), None

        # 目錄模式
        if self._index >= len(self._files):
            if self._loop:
                self._index = 0
            else:
                raise StopIteration("FileCamera: 影像序列已播放完畢")

        path = self._files[self._index]
        img  = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"FileCamera: 無法讀取 {path}")

        logger.debug(
            "FileCamera: [%d/%d] %s",
            self._index + 1, len(self._files), path.name,
        )
        self._index += 1
        return img, None

    # ------------------------------------------------------------------
    #  額外控制方法（目錄模式輔助）
    # ------------------------------------------------------------------
    def advance(self) -> np.ndarray:
        """手動取下一張（目錄模式）。"""
        rgb, _ = self.get_frame()
        return rgb

    def go_prev(self) -> np.ndarray:
        """回到上一張（目錄模式）。"""
        if self._single is not None:
            return self._single.copy()
        self._index = max(0, self._index - 2)   # _capture 會 +1，所以退 2
        rgb, _ = self.get_frame()
        return rgb

    def reset(self) -> None:
        """重置到第一張。"""
        self._index = 0
        logger.debug("FileCamera: 重置到第一張")

    # ------------------------------------------------------------------
    #  屬性
    # ------------------------------------------------------------------
    @property
    def file_count(self) -> int:
        return len(self._files)

    @property
    def current_index(self) -> int:
        """下次 get_frame() 將讀取的索引（0-based）。"""
        return self._index

    @property
    def current_filename(self) -> str:
        if not self._files:
            return ""
        idx = min(self._index, len(self._files) - 1)
        return self._files[idx].name

    def is_dir_mode(self) -> bool:
        return self._single is None
