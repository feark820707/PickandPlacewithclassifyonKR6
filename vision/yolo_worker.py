# =============================================================================
#  KR6 Pick & Place — YOLO-OBB 推論 Worker
#  GPU Thread：接收 RGB → 輸出 label, OBB bbox, θ, confidence
# =============================================================================
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from queue import Queue, Full
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Data Classes
# ---------------------------------------------------------------------------
@dataclass
class OBBDetection:
    """單一 OBB 偵測結果"""
    label: str                          # 物件類別名稱
    cx: float                           # OBB 中心 X (像素)
    cy: float                           # OBB 中心 Y (像素)
    width: float                        # OBB 寬 (像素)
    height: float                       # OBB 高 (像素)
    theta_deg: float                    # 旋轉角（度）→ Rz
    confidence: float                   # 信心度
    place_pos: dict = field(default_factory=dict)  # 放置位置 {x, y, z}


@dataclass
class YOLOResult:
    """一幀的完整偵測結果"""
    cycle_id: int
    timestamp: float
    detections: list[OBBDetection]
    inference_ms: float
    frame_rgb: Optional[np.ndarray] = None   # 原始 RGB 幀（可選保留）
    frame_depth: Optional[np.ndarray] = None  # 深度圖（透傳給 Geometry）


# ---------------------------------------------------------------------------
#  YOLO Worker
# ---------------------------------------------------------------------------
class YOLOWorker:
    """
    YOLO-OBB 推論 Worker。

    運行在獨立 Thread 中，持續從輸入 Queue 取得 (rgb, depth) 幀，
    透過 YOLO-OBB 模型推論，將結果放入輸出 Queue。

    GPU 資源獨佔，CPU 開銷極低。

    用法:
        worker = YOLOWorker(
            model_path="assets/best_obb.pt",
            confidence=0.5,
            place_map={"classA": {"x": 400, "y": 100, "z": -50}},
        )
        worker.start(input_queue, output_queue)
        ...
        worker.stop()
    """

    def __init__(
        self,
        model_path: str = "assets/best_obb.pt",
        confidence: float = 0.5,
        device: str = "cuda:0",
        place_map: dict | None = None,
        queue_max_size: int = 10,
    ):
        self._model_path = model_path
        self._confidence = confidence
        self._device = device
        self._place_map = place_map or {}
        self._queue_max_size = queue_max_size

        self._model = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._cycle_counter = 0

    def load_model(self) -> None:
        """載入 YOLO-OBB 模型到 GPU"""
        from ultralytics import YOLO

        logger.info(
            "載入 YOLO-OBB 模型: %s → %s",
            self._model_path, self._device,
        )
        self._model = YOLO(self._model_path)
        # Warm-up: 跑一次空推論
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model(dummy, device=self._device, verbose=False)
        logger.info("YOLO-OBB 模型載入完成")

    def start(
        self,
        input_queue: Queue,
        output_queue: Queue,
    ) -> None:
        """
        啟動推論 Thread。

        Args:
            input_queue:  輸入 Queue，元素為 (cycle_id, rgb, depth_or_none)
            output_queue: 輸出 Queue，元素為 YOLOResult
        """
        if self._model is None:
            self.load_model()

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(input_queue, output_queue),
            name="YOLOWorker",
            daemon=True,
        )
        self._thread.start()
        logger.info("YOLOWorker Thread 已啟動")

    def stop(self) -> None:
        """停止推論 Thread"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("YOLOWorker Thread 已停止")

    def _run_loop(
        self,
        input_queue: Queue,
        output_queue: Queue,
    ) -> None:
        """推論主迴圈"""
        while self._running:
            try:
                # 從輸入 Queue 取得幀（timeout 避免阻塞）
                item = input_queue.get(timeout=1.0)
                if item is None:  # poison pill
                    break

                cycle_id, rgb, depth_or_none = item
                result = self.infer(cycle_id, rgb, depth_or_none)

                # 放入輸出 Queue（背壓控制）
                try:
                    output_queue.put(result, timeout=0.5)
                except Full:
                    logger.warning(
                        "輸出 Queue 已滿，丟棄 cycle_id=%d",
                        cycle_id,
                    )
            except Exception as e:
                if self._running:  # 非正常停止時才記錄
                    if "Empty" not in str(type(e)):
                        logger.error("YOLOWorker 推論錯誤: %s", e)

    def infer(
        self,
        cycle_id: int,
        rgb: np.ndarray,
        depth_or_none: Optional[np.ndarray] = None,
    ) -> YOLOResult:
        """
        單幀推論。

        Args:
            cycle_id: 循環 ID
            rgb: BGR 影像, shape (H, W, 3)
            depth_or_none: 深度圖（透傳，不在此處使用）

        Returns:
            YOLOResult
        """
        t0 = time.perf_counter()

        results = self._model(
            rgb,
            device=self._device,
            conf=self._confidence,
            verbose=False,
        )

        inference_ms = (time.perf_counter() - t0) * 1000

        detections = []
        if results and results[0].obb is not None:
            obb_data = results[0].obb

            for i in range(len(obb_data)):
                # OBB 輸出：(cx, cy, w, h, theta_rad)
                xywhr = obb_data.xywhr[i]
                cx = float(xywhr[0])
                cy = float(xywhr[1])
                w = float(xywhr[2])
                h = float(xywhr[3])
                theta_rad = float(xywhr[4])
                theta_deg = math.degrees(theta_rad)

                cls_id = int(obb_data.cls[i])
                label = self._model.names[cls_id]
                conf = float(obb_data.conf[i])

                # 查找放置位置
                place_pos = self._place_map.get(label, {})

                detections.append(OBBDetection(
                    label=label,
                    cx=cx, cy=cy,
                    width=w, height=h,
                    theta_deg=theta_deg,
                    confidence=conf,
                    place_pos=place_pos,
                ))

        logger.info(
            "YOLO-OBB 推論完成: cycle=%d, 偵測=%d, %.1fms",
            cycle_id, len(detections), inference_ms,
            extra={
                "cycle_id": cycle_id,
                "data": {
                    "num_detections": len(detections),
                    "inference_ms": round(inference_ms, 1),
                },
            },
        )

        return YOLOResult(
            cycle_id=cycle_id,
            timestamp=time.time(),
            detections=detections,
            inference_ms=inference_ms,
            frame_rgb=rgb,
            frame_depth=depth_or_none,
        )

    @property
    def is_running(self) -> bool:
        return self._running
