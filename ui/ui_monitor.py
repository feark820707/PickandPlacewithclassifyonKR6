# =============================================================================
#  KR6 Pick & Place — UI 即時監控
#  OpenCV-based 即時影像疊加 + 健康儀表板 + 效能指標
#
#  Thread 運行，綁定 Core 14（E-core 限速顯示）
# =============================================================================
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from coordinator.coordinator import Coordinator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Color Palette (BGR)
# ---------------------------------------------------------------------------
class Colors:
    GREEN = (0, 200, 0)
    RED = (0, 0, 220)
    YELLOW = (0, 220, 220)
    WHITE = (255, 255, 255)
    GRAY = (128, 128, 128)
    CYAN = (220, 200, 0)
    ORANGE = (0, 140, 255)
    BG_DARK = (30, 30, 30)
    BG_PANEL = (45, 45, 45)


# ---------------------------------------------------------------------------
#  UI Monitor
# ---------------------------------------------------------------------------
class UIMonitor:
    """
    即時監控 UI：OpenCV 視窗顯示。

    功能：
      1. 即時相機影像（RGB）
      2. OBB 偵測框疊加
      3. 系統狀態指示燈
      4. 效能指標（FPS, cycle time, yolo latency）
      5. 健康狀態儀表板
      6. 最近 Pick Pose 資訊

    鍵盤控制：
      - 'q' / ESC: 退出
      - 's': 截圖
      - 'p': 暫停/恢復顯示
      - 'd': 顯示/隱藏偵測框

    用法:
        ui = UIMonitor(coordinator)
        ui.start_thread()   # 啟動獨立 Thread
        # ... 主系統運行 ...
        ui.stop()
    """

    WINDOW_NAME = "KR6 Pick & Place Monitor"
    PANEL_WIDTH = 320       # 右側資訊面板寬度
    TARGET_FPS = 15         # UI 刷新率（限速）
    SCREENSHOT_DIR = "screenshots"

    def __init__(
        self,
        coordinator: Coordinator,
        window_width: int = 1280,
        window_height: int = 720,
    ):
        self._coordinator = coordinator
        self._window_w = window_width
        self._window_h = window_height

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._paused = False
        self._show_detections = True
        self._screenshot_count = 0

        # 最近的幀與偵測結果（由 coordinator 更新）
        self._last_frame: Optional[np.ndarray] = None
        self._last_detections: list = []
        self._last_pose: Optional[dict] = None
        self._frame_lock = threading.Lock()

    # ------------------------------------------------------------------
    #  Thread Management
    # ------------------------------------------------------------------
    def start_thread(self) -> threading.Thread:
        """啟動 UI Thread（Daemon）"""
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name="UIMonitor",
            daemon=True,
        )
        self._thread.start()
        logger.info("UIMonitor Thread 已啟動")
        return self._thread

    def stop(self) -> None:
        """停止 UI"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        logger.info("UIMonitor 已停止")

    # ------------------------------------------------------------------
    #  外部更新介面
    # ------------------------------------------------------------------
    def update_frame(self, rgb: np.ndarray) -> None:
        """更新顯示幀（由 coordinator 呼叫）"""
        with self._frame_lock:
            self._last_frame = rgb.copy()

    def update_detections(self, detections: list) -> None:
        """更新偵測結果"""
        with self._frame_lock:
            self._last_detections = list(detections)

    def update_pose(self, pose_dict: dict) -> None:
        """更新最近的 PickPose"""
        with self._frame_lock:
            self._last_pose = dict(pose_dict)

    # ------------------------------------------------------------------
    #  Main Loop
    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        """UI 主迴圈"""
        try:
            import cv2
        except ImportError:
            logger.error("UI Monitor 需要 opencv-python，請安裝: pip install opencv-python")
            return

        frame_interval = 1.0 / self.TARGET_FPS

        try:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                self.WINDOW_NAME,
                self._window_w + self.PANEL_WIDTH,
                self._window_h,
            )
        except Exception as e:
            logger.error("無法建立 UI 視窗: %s", e)
            return

        logger.info("UI 視窗已開啟: %dx%d", self._window_w + self.PANEL_WIDTH, self._window_h)

        while self._running:
            loop_start = time.time()

            # 建立畫布
            canvas = self._build_canvas(cv2)

            # 顯示
            try:
                cv2.imshow(self.WINDOW_NAME, canvas)
            except Exception:
                break

            # 鍵盤處理
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC
                logger.info("UI: 使用者按下退出鍵")
                self._running = False
                break
            elif key == ord("s"):
                self._take_screenshot(cv2, canvas)
            elif key == ord("p"):
                self._paused = not self._paused
            elif key == ord("d"):
                self._show_detections = not self._show_detections

            # 限速
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    # ------------------------------------------------------------------
    #  Canvas Builder
    # ------------------------------------------------------------------
    def _build_canvas(self, cv2) -> np.ndarray:
        """組合影像 + 資訊面板"""
        total_w = self._window_w + self.PANEL_WIDTH
        canvas = np.full((self._window_h, total_w, 3), 30, dtype=np.uint8)

        # 左側：相機影像
        with self._frame_lock:
            frame = self._last_frame
            detections = list(self._last_detections)

        if frame is not None and not self._paused:
            # 縮放到視窗大小
            display = cv2.resize(frame, (self._window_w, self._window_h))

            # 繪製偵測框
            if self._show_detections and detections:
                display = self._draw_detections(cv2, display, detections)

            canvas[:, :self._window_w] = display
        elif self._paused:
            self._draw_text(
                cv2, canvas, "PAUSED", (self._window_w // 2 - 80, self._window_h // 2),
                Colors.YELLOW, scale=2.0, thickness=3,
            )
        else:
            self._draw_text(
                cv2, canvas, "No camera feed", (self._window_w // 2 - 120, self._window_h // 2),
                Colors.GRAY, scale=1.0,
            )

        # 右側：資訊面板
        panel_x = self._window_w
        self._draw_panel(cv2, canvas, panel_x)

        return canvas

    # ------------------------------------------------------------------
    #  Detection Overlay
    # ------------------------------------------------------------------
    def _draw_detections(self, cv2, frame: np.ndarray, detections: list) -> np.ndarray:
        """繪製 OBB 偵測框"""
        for det in detections:
            try:
                # 取得 OBB 屬性
                cx = getattr(det, "cx", 0)
                cy = getattr(det, "cy", 0)
                w = getattr(det, "width", 50)
                h = getattr(det, "height", 30)
                theta = getattr(det, "theta_deg", 0)
                label = getattr(det, "label", "?")
                conf = getattr(det, "confidence", 0)

                # 計算 OBB 四個頂點
                import math
                rad = math.radians(theta)
                cos_t, sin_t = math.cos(rad), math.sin(rad)
                half_w, half_h = w / 2, h / 2

                corners = []
                for dx, dy in [(-half_w, -half_h), (half_w, -half_h),
                               (half_w, half_h), (-half_w, half_h)]:
                    rx = dx * cos_t - dy * sin_t + cx
                    ry = dx * sin_t + dy * cos_t + cy
                    corners.append([int(rx), int(ry)])

                pts = np.array(corners, dtype=np.int32)

                # 繪製 OBB
                cv2.polylines(frame, [pts], True, Colors.GREEN, 2)

                # 標籤
                text = f"{label} {conf:.2f} | Rz={theta:.1f}"
                text_x, text_y = int(cx - 40), int(cy - h / 2 - 10)
                self._draw_text(cv2, frame, text, (text_x, text_y), Colors.GREEN, scale=0.5)

                # 中心點
                cv2.circle(frame, (int(cx), int(cy)), 4, Colors.CYAN, -1)

            except Exception:
                continue

        return frame

    # ------------------------------------------------------------------
    #  Information Panel
    # ------------------------------------------------------------------
    def _draw_panel(self, cv2, canvas: np.ndarray, x0: int) -> None:
        """繪製右側資訊面板"""
        y = 20

        # 面板背景
        cv2.rectangle(
            canvas,
            (x0, 0),
            (x0 + self.PANEL_WIDTH, self._window_h),
            Colors.BG_PANEL,
            -1,
        )

        # --- Title ---
        self._draw_text(cv2, canvas, "KR6 SYSTEM STATUS", (x0 + 10, y), Colors.CYAN, scale=0.7, thickness=2)
        y += 35

        # --- State Machine ---
        y = self._draw_state_section(cv2, canvas, x0 + 10, y)

        # --- Metrics ---
        y = self._draw_metrics_section(cv2, canvas, x0 + 10, y)

        # --- Health ---
        y = self._draw_health_section(cv2, canvas, x0 + 10, y)

        # --- Last Pose ---
        y = self._draw_pose_section(cv2, canvas, x0 + 10, y)

        # --- Controls ---
        y = self._window_h - 80
        self._draw_text(cv2, canvas, "Controls:", (x0 + 10, y), Colors.GRAY, scale=0.45)
        y += 18
        self._draw_text(cv2, canvas, "Q/ESC: Quit  S: Screenshot", (x0 + 10, y), Colors.GRAY, scale=0.4)
        y += 16
        self._draw_text(cv2, canvas, "P: Pause  D: Toggle detections", (x0 + 10, y), Colors.GRAY, scale=0.4)

    def _draw_state_section(self, cv2, canvas, x, y) -> int:
        """繪製狀態機區塊"""
        self._draw_text(cv2, canvas, "State Machine", (x, y), Colors.WHITE, scale=0.55, thickness=1)
        y += 5
        cv2.line(canvas, (x, y), (x + self.PANEL_WIDTH - 20, y), Colors.GRAY, 1)
        y += 20

        try:
            sm = self._coordinator.state_machine
            state = sm.state
            state_name = state.name

            # 狀態顏色
            state_colors = {
                "INITIALIZING": Colors.YELLOW,
                "READY": Colors.CYAN,
                "RUNNING": Colors.GREEN,
                "ERROR": Colors.RED,
                "SAFE_STOP": Colors.ORANGE,
            }
            color = state_colors.get(state_name, Colors.WHITE)

            # 狀態指示燈
            cv2.circle(canvas, (x + 8, y - 5), 6, color, -1)
            self._draw_text(cv2, canvas, f" {state_name}", (x + 20, y), color, scale=0.55)
            y += 22

            # 持續時間
            duration = sm.state_duration
            self._draw_text(
                cv2, canvas, f"Duration: {duration:.1f}s",
                (x, y), Colors.GRAY, scale=0.4,
            )
            y += 18

            # 錯誤資訊
            if sm.error_level:
                self._draw_text(
                    cv2, canvas, f"Error: {sm.error_level.name}",
                    (x, y), Colors.RED, scale=0.4,
                )
                y += 16

        except Exception:
            self._draw_text(cv2, canvas, "State: N/A", (x, y), Colors.GRAY, scale=0.5)
            y += 20

        y += 10
        return y

    def _draw_metrics_section(self, cv2, canvas, x, y) -> int:
        """繪製效能指標區塊"""
        self._draw_text(cv2, canvas, "Performance", (x, y), Colors.WHITE, scale=0.55, thickness=1)
        y += 5
        cv2.line(canvas, (x, y), (x + self.PANEL_WIDTH - 20, y), Colors.GRAY, 1)
        y += 20

        try:
            metrics = self._coordinator.metrics
            summary = metrics.summary()

            items = [
                ("Cycles", f"{summary.get('cycle_count', 0)}"),
                ("Errors", f"{summary.get('error_count', 0)}"),
                ("Skips", f"{summary.get('skip_count', 0)}"),
            ]

            for label, value in items:
                self._draw_text(cv2, canvas, f"{label}:", (x, y), Colors.GRAY, scale=0.4)
                self._draw_text(cv2, canvas, value, (x + 80, y), Colors.WHITE, scale=0.4)
                y += 16

            # 各階段平均耗時
            avgs = summary.get("averages", {})
            if avgs:
                y += 5
                self._draw_text(cv2, canvas, "Avg Latency (ms):", (x, y), Colors.GRAY, scale=0.4)
                y += 16
                for stage, ms in avgs.items():
                    color = Colors.GREEN if ms < 50 else Colors.YELLOW if ms < 100 else Colors.RED
                    self._draw_text(
                        cv2, canvas, f"  {stage}: {ms:.1f}",
                        (x, y), color, scale=0.4,
                    )
                    y += 15

        except Exception:
            self._draw_text(cv2, canvas, "Metrics: N/A", (x, y), Colors.GRAY, scale=0.4)
            y += 16

        y += 10
        return y

    def _draw_health_section(self, cv2, canvas, x, y) -> int:
        """繪製健康監控區塊"""
        self._draw_text(cv2, canvas, "Health", (x, y), Colors.WHITE, scale=0.55, thickness=1)
        y += 5
        cv2.line(canvas, (x, y), (x + self.PANEL_WIDTH - 20, y), Colors.GRAY, 1)
        y += 20

        try:
            health = self._coordinator.health
            statuses = health.check_all()

            for module, status in statuses.items():
                status_name = status.name if hasattr(status, "name") else str(status)

                color_map = {
                    "OK": Colors.GREEN,
                    "WARN": Colors.YELLOW,
                    "TIMEOUT": Colors.RED,
                    "UNKNOWN": Colors.GRAY,
                }
                color = color_map.get(status_name, Colors.GRAY)

                # 狀態圖示
                cv2.circle(canvas, (x + 6, y - 4), 4, color, -1)
                self._draw_text(
                    cv2, canvas, f" {module}: {status_name}",
                    (x + 16, y), color, scale=0.4,
                )
                y += 16

        except Exception:
            self._draw_text(cv2, canvas, "Health: N/A", (x, y), Colors.GRAY, scale=0.4)
            y += 16

        y += 10
        return y

    def _draw_pose_section(self, cv2, canvas, x, y) -> int:
        """繪製最近 PickPose 區塊"""
        self._draw_text(cv2, canvas, "Last Pick Pose", (x, y), Colors.WHITE, scale=0.55, thickness=1)
        y += 5
        cv2.line(canvas, (x, y), (x + self.PANEL_WIDTH - 20, y), Colors.GRAY, 1)
        y += 20

        with self._frame_lock:
            pose = self._last_pose

        if pose:
            items = [
                ("X", f"{pose.get('pick_x', 0):.1f} mm"),
                ("Y", f"{pose.get('pick_y', 0):.1f} mm"),
                ("Z", f"{pose.get('pick_z', 0):.1f} mm"),
                ("Rz", f"{pose.get('rz', 0):.1f} deg"),
                ("Label", f"{pose.get('label', 'N/A')}"),
            ]
            for label, value in items:
                self._draw_text(cv2, canvas, f"{label}:", (x, y), Colors.GRAY, scale=0.4)
                self._draw_text(cv2, canvas, value, (x + 40, y), Colors.CYAN, scale=0.4)
                y += 16
        else:
            self._draw_text(cv2, canvas, "No pose data", (x, y), Colors.GRAY, scale=0.4)
            y += 16

        y += 10
        return y

    # ------------------------------------------------------------------
    #  Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _draw_text(
        cv2,
        img: np.ndarray,
        text: str,
        pos: tuple[int, int],
        color: tuple[int, int, int],
        scale: float = 0.5,
        thickness: int = 1,
    ) -> None:
        """Helper: 在影像上繪製文字"""
        cv2.putText(
            img, text, pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale, color, thickness,
            cv2.LINE_AA,
        )

    def _take_screenshot(self, cv2, canvas: np.ndarray) -> None:
        """截圖儲存"""
        from pathlib import Path
        Path(self.SCREENSHOT_DIR).mkdir(parents=True, exist_ok=True)

        self._screenshot_count += 1
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.SCREENSHOT_DIR}/screenshot_{ts}_{self._screenshot_count:03d}.png"

        cv2.imwrite(filename, canvas)
        logger.info("截圖已儲存: %s", filename)
