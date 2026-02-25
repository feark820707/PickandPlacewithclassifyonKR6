# =============================================================================
#  KR6 Pick & Place — 系統協調器
#  整合狀態機 + Queue 管理 + 結果合併 + 任務指派
# =============================================================================
from __future__ import annotations

import logging
import threading
import time
from queue import Queue, Empty, Full
from typing import Optional

import numpy as np

from communication.opcua_bridge import DB1Data, OPCUABridge
from config import AppConfig
from coordinator.state_machine import (
    ErrorLevel,
    StateMachine,
    SystemState,
)
from utils.health_monitor import HealthMonitor
from utils.metrics import CycleMetrics
from vision.geometry_worker import GeometryPool, compute_pick_pose, PickPose
from vision.yolo_worker import YOLOResult, YOLOWorker

logger = logging.getLogger(__name__)


class Coordinator:
    """
    系統協調器 — 核心調度中心。

    職責：
      1. 管理系統狀態機（INIT → READY → RUNNING → ERROR → SAFE_STOP）
      2. 協調 Camera → YOLO → Geometry → OPC-UA 的資料流
      3. 合併推論結果與幾何計算，組裝完整 Pick & Place 指令
      4. 監控各模組健康狀態
      5. 收集效能指標

    架構：
      - CameraThread  → Queue_frame  → YOLOWorker
      - YOLOWorker    → Queue_yolo   → Coordinator._process_loop
      - Coordinator   → OPC-UA       → PLC → KR6

    用法:
        coordinator = Coordinator(cfg)
        coordinator.initialize()
        coordinator.run()    # 主迴圈（阻塞）
        coordinator.shutdown()
    """

    def __init__(self, cfg: AppConfig):
        self._cfg = cfg

        # 狀態機
        self.state_machine = StateMachine()

        # 健康監控
        self.health = HealthMonitor(
            timeout_sec=cfg.heartbeat_timeout,
        )

        # 效能指標
        self.metrics = CycleMetrics()

        # Queues
        self._frame_queue: Queue = Queue(maxsize=cfg.queue_max_size)
        self._yolo_queue: Queue = Queue(maxsize=cfg.queue_max_size)

        # Workers
        self._yolo_worker: Optional[YOLOWorker] = None
        self._geometry_pool: Optional[GeometryPool] = None
        self._opcua_bridge: Optional[OPCUABridge] = None

        # Camera
        self._camera = None

        # Homography 矩陣
        self._H: Optional[np.ndarray] = None

        # 控制
        self._running = False
        self._cycle_id = 0
        self._threads: list[threading.Thread] = []

    # ------------------------------------------------------------------
    #  Initialize
    # ------------------------------------------------------------------
    def initialize(self) -> bool:
        """
        系統初始化：載入模型、連線相機/PLC、載入標定資料。

        Returns:
            True=初始化成功, False=失敗
        """
        logger.info("系統初始化中...")
        self.state_machine.transition_to(SystemState.INITIALIZING)

        try:
            # 1. 載入 Homography 矩陣
            self._load_homography()

            # 2. 初始化相機
            self._init_camera()

            # 3. 初始化 YOLO Worker
            self._init_yolo()

            # 4. 初始化 Geometry Pool
            self._init_geometry_pool()

            # 5. 初始化 OPC-UA
            self._init_opcua()

            # 6. 轉換到 READY
            self.state_machine.transition_to(SystemState.READY)
            logger.info("系統初始化完成 → READY")
            return True

        except Exception as e:
            logger.error("系統初始化失敗: %s", e, exc_info=True)
            self.state_machine.enter_error(
                ErrorLevel.CRITICAL,
                f"初始化失敗: {e}",
            )
            return False

    def _load_homography(self) -> None:
        """載入 Homography 矩陣"""
        from calibration.hand_eye_calib import load_homography
        self._H = load_homography(self._cfg.homography_path)

    def _init_camera(self) -> None:
        """初始化相機"""
        from camera.factory import create_camera
        self._camera = create_camera(self._cfg)
        self._camera.connect()
        self.health.register("camera")

    def _init_yolo(self) -> None:
        """初始化 YOLO Worker"""
        self._yolo_worker = YOLOWorker(
            model_path=self._cfg.yolo_model,
            confidence=self._cfg.yolo_confidence,
            device=self._cfg.yolo_device,
            place_map=self._cfg.place_map,
            queue_max_size=self._cfg.queue_max_size,
        )
        self._yolo_worker.load_model()
        self.health.register("yolo_worker")

    def _init_geometry_pool(self) -> None:
        """初始化 Geometry Pool"""
        num_workers = len(self._cfg.core_geometry)
        self._geometry_pool = GeometryPool(max_workers=num_workers)
        self._geometry_pool.start()
        self.health.register("geometry_pool")

    def _init_opcua(self) -> None:
        """初始化 OPC-UA 連線"""
        self._opcua_bridge = OPCUABridge(
            plc_ip=self._cfg.plc_ip,
            opc_port=self._cfg.opc_port,
            namespace=self._cfg.opc_namespace,
            cmd_timeout=self._cfg.plc_cmd_timeout,
        )
        self._opcua_bridge.connect()
        self.health.register("opcua")

    # ------------------------------------------------------------------
    #  Main Run Loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        主運行迴圈（阻塞式）。

        流程：
        1. 啟動 YOLO Worker Thread
        2. 啟動 Camera Capture Thread
        3. 進入主處理迴圈：從 YOLO Queue 取結果 → 幾何計算 → OPC-UA 寫入
        """
        if self.state_machine.state != SystemState.READY:
            logger.error("系統未就緒，無法啟動 (state=%s)", self.state_machine.state.name)
            return

        self._running = True
        self.state_machine.transition_to(SystemState.RUNNING)

        # 啟動 YOLO Worker
        self._yolo_worker.start(self._frame_queue, self._yolo_queue)

        # 啟動 Camera Capture Thread
        cam_thread = threading.Thread(
            target=self._camera_loop,
            name="CameraCapture",
            daemon=True,
        )
        cam_thread.start()
        self._threads.append(cam_thread)

        logger.info("系統已啟動，進入主迴圈")

        # 主處理迴圈
        try:
            self._process_loop()
        except KeyboardInterrupt:
            logger.info("收到中斷信號")
        except Exception as e:
            logger.error("主迴圈異常: %s", e, exc_info=True)
            self.state_machine.enter_error(ErrorLevel.CRITICAL, str(e))
        finally:
            self.shutdown()

    def _camera_loop(self) -> None:
        """相機擷取迴圈（獨立 Thread）"""
        while self._running:
            try:
                rgb, depth = self._camera.get_frame()
                self._cycle_id += 1

                self.health.beat("camera", fps=self._cfg.d435_fps)

                try:
                    self._frame_queue.put(
                        (self._cycle_id, rgb, depth),
                        timeout=0.5,
                    )
                except Full:
                    # 背壓控制：Queue 滿時丟棄
                    try:
                        self._frame_queue.get_nowait()  # 丟棄最舊
                    except Empty:
                        pass
                    self._frame_queue.put((self._cycle_id, rgb, depth))
                    logger.warning("Frame Queue 背壓，丟棄最舊幀")

            except Exception as e:
                logger.error("相機擷取錯誤: %s", e)
                time.sleep(0.5)

    def _process_loop(self) -> None:
        """主處理迴圈：接收 YOLO 結果 → 幾何 → OPC-UA"""
        while self._running:
            try:
                # 從 YOLO Queue 取結果
                yolo_result: YOLOResult = self._yolo_queue.get(timeout=2.0)

                self.health.beat("yolo_worker")
                self.health.beat("coordinator")

                cycle_id = yolo_result.cycle_id
                self.metrics.start_cycle()
                self.metrics.record("yolo", yolo_result.inference_ms)

                if not yolo_result.detections:
                    logger.debug(
                        "cycle=%d 無偵測", cycle_id,
                        extra={"cycle_id": cycle_id},
                    )
                    self.metrics.record_skip()
                    continue

                # 取第一個偵測結果（優先級最高 or 最大信心度）
                det = max(yolo_result.detections, key=lambda d: d.confidence)

                # 幾何計算
                with self.metrics.measure("geometry"):
                    pose = self._compute_geometry(
                        det, yolo_result.frame_depth, cycle_id,
                    )

                if pose is None:
                    continue

                # 寫入 PLC
                with self.metrics.measure("opcua"):
                    success = self._send_to_plc(pose, cycle_id)

                if success:
                    self.metrics.complete_cycle()
                    self.state_machine.transition_to(SystemState.READY)
                    self.state_machine.transition_to(SystemState.RUNNING)

                    logger.info(
                        "cycle=%d 完成: %s → pick=(%.1f, %.1f, %.1f) rz=%.1f",
                        cycle_id, pose.label,
                        pose.pick_x, pose.pick_y, pose.pick_z, pose.rz,
                        extra={
                            "cycle_id": cycle_id,
                            "data": self.metrics.summary(),
                        },
                    )

            except Empty:
                continue
            except Exception as e:
                logger.error(
                    "處理迴圈錯誤: %s", e,
                    exc_info=True,
                )
                self.metrics.record_error()
                self.state_machine.enter_error(
                    ErrorLevel.RETRY,
                    str(e),
                )

    def _compute_geometry(
        self,
        det,
        depth_or_none: Optional[np.ndarray],
        cycle_id: int,
    ) -> Optional[PickPose]:
        """計算抓取姿態"""
        try:
            pose = compute_pick_pose(
                cx=det.cx,
                cy=det.cy,
                theta_deg=det.theta_deg,
                label=det.label,
                place_pos=det.place_pos,
                H=self._H,
                depth_or_none=depth_or_none,
                depth_mode=self._cfg.depth_mode,
                offset_x=self._cfg.offset_x,
                offset_y=self._cfg.offset_y,
                camera_height_mm=self._cfg.camera_height_mm,
                suction_length_mm=self._cfg.suction_length_mm,
                safety_margin_mm=self._cfg.safety_margin_mm,
                worktable_z_mm=self._cfg.worktable_z_mm,
                thickness_map=self._cfg.thickness_map,
            )
            self.health.beat("geometry_pool")
            return pose

        except Exception as e:
            logger.error(
                "幾何計算失敗 (cycle=%d): %s",
                cycle_id, e,
                extra={"cycle_id": cycle_id},
            )
            self.metrics.record_error()
            return None

    def _send_to_plc(self, pose: PickPose, cycle_id: int) -> bool:
        """寫入 PLC 並等待完成"""
        place = pose.place_pos or {}
        db1 = DB1Data(
            pick_x=pose.pick_x,
            pick_y=pose.pick_y,
            pick_z=pose.pick_z,
            rx=pose.rx,
            ry=pose.ry,
            rz=pose.rz,
            place_x=place.get("x", 0.0),
            place_y=place.get("y", 0.0),
            place_z=place.get("z", 0.0),
            cmd=1,
        )

        self._opcua_bridge.write_pick_command(db1, cycle_id=cycle_id)
        self.health.beat("opcua")

        done = self._opcua_bridge.wait_for_done(cycle_id=cycle_id)
        if not done:
            logger.warning(
                "PLC 逾時 (cycle=%d)", cycle_id,
                extra={"cycle_id": cycle_id},
            )
            self.state_machine.enter_error(
                ErrorLevel.RETRY,
                "PLC cmd=2 逾時",
                cycle_id=cycle_id,
            )
        return done

    # ------------------------------------------------------------------
    #  Shutdown
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """安全關閉系統"""
        logger.info("系統關閉中...")
        self._running = False

        # 停止 YOLO Worker
        if self._yolo_worker:
            self._yolo_worker.stop()

        # 關閉 Geometry Pool
        if self._geometry_pool:
            self._geometry_pool.shutdown()

        # 斷開 OPC-UA
        if self._opcua_bridge:
            try:
                self._opcua_bridge.reset_cmd()
            except Exception:
                pass
            self._opcua_bridge.disconnect()

        # 斷開相機
        if self._camera:
            self._camera.disconnect()

        # 等待所有 Thread
        for t in self._threads:
            t.join(timeout=3.0)

        self.state_machine.enter_safe_stop("正常關閉")
        logger.info("系統已安全關閉")

    # ------------------------------------------------------------------
    #  Status
    # ------------------------------------------------------------------
    def status(self) -> dict:
        """取得完整系統狀態"""
        return {
            "state_machine": self.state_machine.summary(),
            "health": self.health.summary(),
            "metrics": self.metrics.summary(),
            "cycle_id": self._cycle_id,
            "config": {
                "camera_type": self._cfg.camera_type,
                "depth_mode": self._cfg.depth_mode,
            },
        }
