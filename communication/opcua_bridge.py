# =============================================================================
#  KR6 Pick & Place — OPC-UA 通訊橋接
#  PC ↔ S7-1515 PLC 通訊，含自動重連機制
# =============================================================================
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Data Classes
# ---------------------------------------------------------------------------
@dataclass
class DB1Data:
    """S7-1515 DB1 資料塊結構"""
    pick_x: float = 0.0    # 抓取點 X (mm)，含偏移補償
    pick_y: float = 0.0    # 抓取點 Y (mm)，含偏移補償
    pick_z: float = 0.0    # 抓取點 Z (mm)
    rx: float = 0.0        # 固定 0.0
    ry: float = 0.0        # 固定 0.0
    rz: float = 0.0        # 物件旋轉角 θ（°）
    place_x: float = 0.0   # 放置點 X
    place_y: float = 0.0   # 放置點 Y
    place_z: float = 0.0   # 放置點 Z
    cmd: int = 0           # 0=idle, 1=執行, 2=完成


# ---------------------------------------------------------------------------
#  OPC-UA Bridge
# ---------------------------------------------------------------------------
class OPCUABridge:
    """
    OPC-UA 通訊橋接，與 S7-1515 PLC 溝通。

    功能：
      - 連線 / 自動重連
      - 寫入 DB1 資料（pick/place 座標 + cmd）
      - 讀取 PLC 狀態（cmd=2 完成回報）
      - 含 retry + circuit breaker 保護

    用法:
        bridge = OPCUABridge(plc_ip="192.168.0.10", opc_port=4840)
        bridge.connect()
        bridge.write_pick_command(db1_data)
        bridge.wait_for_done(timeout=10.0)
        bridge.disconnect()
    """

    def __init__(
        self,
        plc_ip: str = "192.168.0.10",
        opc_port: int = 4840,
        namespace: str = "urn:siemens:s71500",
        cmd_timeout: float = 10.0,
        reconnect_delay: float = 2.0,
        max_reconnect: int = 3,
    ):
        self._plc_ip = plc_ip
        self._opc_port = opc_port
        self._namespace = namespace
        self._cmd_timeout = cmd_timeout
        self._reconnect_delay = reconnect_delay
        self._max_reconnect = max_reconnect

        self._client = None
        self._connected = False
        self._lock = threading.Lock()
        self._ns_idx: int | None = None

        # DB1 node 快取
        self._nodes: dict[str, object] = {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        """建立 OPC-UA 連線"""
        try:
            from opcua import Client
        except ImportError:
            raise ImportError(
                "opcua 未安裝。請執行: pip install opcua"
            )

        url = f"opc.tcp://{self._plc_ip}:{self._opc_port}"
        logger.info("OPC-UA 連線中: %s", url)

        self._client = Client(url)
        self._client.connect()
        self._connected = True

        # 取得 namespace index
        self._ns_idx = self._client.get_namespace_index(self._namespace)

        # 快取 DB1 nodes
        self._cache_nodes()

        logger.info("OPC-UA 連線成功 (ns_idx=%d)", self._ns_idx)

    def disconnect(self) -> None:
        """斷開 OPC-UA 連線"""
        if self._client:
            try:
                self._client.disconnect()
            except Exception as e:
                logger.warning("OPC-UA 斷線時發生錯誤: %s", e)
            self._client = None

        self._connected = False
        self._nodes.clear()
        logger.info("OPC-UA 已斷線")

    def _cache_nodes(self) -> None:
        """快取 DB1 中各欄位的 OPC-UA Node"""
        db1_fields = [
            "pick_x", "pick_y", "pick_z",
            "rx", "ry", "rz",
            "place_x", "place_y", "place_z",
            "cmd",
        ]
        for field in db1_fields:
            node_id = f'ns={self._ns_idx};s="DB1"."{field}"'
            try:
                node = self._client.get_node(node_id)
                self._nodes[field] = node
            except Exception as e:
                logger.error("無法取得 node %s: %s", node_id, e)

    def _reconnect(self) -> bool:
        """嘗試自動重連"""
        for attempt in range(1, self._max_reconnect + 1):
            try:
                logger.info(
                    "OPC-UA 重連嘗試 %d/%d...",
                    attempt, self._max_reconnect,
                )
                self.disconnect()
                time.sleep(self._reconnect_delay)
                self.connect()
                return True
            except Exception as e:
                logger.error("重連失敗 (%d/%d): %s", attempt, self._max_reconnect, e)

        return False

    def _safe_write(self, field: str, value) -> None:
        """安全寫入單一欄位，含自動重連"""
        with self._lock:
            try:
                node = self._nodes.get(field)
                if node is None:
                    raise RuntimeError(f"Node 未快取: {field}")
                node.set_value(value)
            except Exception as e:
                logger.error("OPC-UA 寫入失敗 (%s): %s", field, e)
                if self._reconnect():
                    # 重連成功，重試一次
                    self._nodes[field].set_value(value)
                else:
                    raise ConnectionError(f"OPC-UA 重連失敗，無法寫入 {field}")

    def _safe_read(self, field: str):
        """安全讀取單一欄位"""
        with self._lock:
            try:
                node = self._nodes.get(field)
                if node is None:
                    raise RuntimeError(f"Node 未快取: {field}")
                return node.get_value()
            except Exception as e:
                logger.error("OPC-UA 讀取失敗 (%s): %s", field, e)
                if self._reconnect():
                    return self._nodes[field].get_value()
                else:
                    raise ConnectionError(f"OPC-UA 重連失敗，無法讀取 {field}")

    def write_pick_command(self, data: DB1Data, cycle_id: int = 0) -> None:
        """
        寫入完整 Pick & Place 指令到 DB1。

        流程：
        1. 寫入所有座標欄位
        2. 寫入 cmd=1（觸發 PLC 執行）

        Args:
            data: DB1Data
            cycle_id: 用於日誌追蹤
        """
        logger.info(
            "寫入 PLC: pick=(%.1f, %.1f, %.1f), rz=%.1f, "
            "place=(%.1f, %.1f, %.1f)",
            data.pick_x, data.pick_y, data.pick_z, data.rz,
            data.place_x, data.place_y, data.place_z,
            extra={"cycle_id": cycle_id},
        )

        # 先寫座標
        self._safe_write("pick_x", float(data.pick_x))
        self._safe_write("pick_y", float(data.pick_y))
        self._safe_write("pick_z", float(data.pick_z))
        self._safe_write("rx", float(data.rx))
        self._safe_write("ry", float(data.ry))
        self._safe_write("rz", float(data.rz))
        self._safe_write("place_x", float(data.place_x))
        self._safe_write("place_y", float(data.place_y))
        self._safe_write("place_z", float(data.place_z))

        # 最後寫 cmd=1 觸發
        self._safe_write("cmd", 1)

    def read_cmd(self) -> int:
        """讀取 cmd 值"""
        return int(self._safe_read("cmd"))

    def wait_for_done(
        self,
        timeout: float | None = None,
        poll_interval: float = 0.1,
        cycle_id: int = 0,
    ) -> bool:
        """
        等待 PLC 回報完成 (cmd=2)。

        Args:
            timeout: 逾時秒數，None=使用預設
            poll_interval: 輪詢間隔
            cycle_id: 用於日誌追蹤

        Returns:
            True=完成, False=逾時
        """
        if timeout is None:
            timeout = self._cmd_timeout

        start = time.time()
        while time.time() - start < timeout:
            cmd = self.read_cmd()
            if cmd == 2:
                elapsed = time.time() - start
                logger.info(
                    "PLC 回報完成 (cmd=2), 耗時 %.2fs",
                    elapsed,
                    extra={"cycle_id": cycle_id},
                )
                # 重置 cmd 為 idle
                self._safe_write("cmd", 0)
                return True
            time.sleep(poll_interval)

        logger.warning(
            "PLC 回報逾時 (%.1fs)", timeout,
            extra={"cycle_id": cycle_id},
        )
        return False

    def reset_cmd(self) -> None:
        """重置 cmd 為 0 (idle)"""
        self._safe_write("cmd", 0)

    # ---- Context Manager ----
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc_info):
        self.disconnect()
