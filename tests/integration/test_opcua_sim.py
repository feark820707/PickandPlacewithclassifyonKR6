# =============================================================================
#  Integration Tests — OPC-UA Simulation
#  模擬 OPC-UA + PLC 通訊邏輯（無需實體 PLC）
# =============================================================================
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from communication.opcua_bridge import DB1Data, OPCUABridge


# ---------------------------------------------------------------------------
#  Mock OPC-UA Bridge（模擬 PLC 通訊）
# ---------------------------------------------------------------------------
class MockOPCUABridge:
    """
    模擬 OPC-UA Bridge，不需要實體 PLC。
    用於驗證通訊邏輯、資料打包、cmd 狀態機。
    """

    def __init__(self, auto_done: bool = True, done_delay: float = 0.0):
        """
        Args:
            auto_done: True=寫入 cmd=1 後自動回報 cmd=2
            done_delay: 自動完成延遲（秒）
        """
        self._auto_done = auto_done
        self._done_delay = done_delay
        self._db1: dict[str, float | int] = {
            "pick_x": 0.0, "pick_y": 0.0, "pick_z": 0.0,
            "rx": 0.0, "ry": 0.0, "rz": 0.0,
            "place_x": 0.0, "place_y": 0.0, "place_z": 0.0,
            "cmd": 0,
        }
        self._connected = False
        self._write_log: list[dict] = []

    def connect(self):
        self._connected = True

    def disconnect(self):
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def write_pick_command(self, data: DB1Data, cycle_id: int = 0):
        """模擬寫入 PLC"""
        self._db1["pick_x"] = data.pick_x
        self._db1["pick_y"] = data.pick_y
        self._db1["pick_z"] = data.pick_z
        self._db1["rx"] = data.rx
        self._db1["ry"] = data.ry
        self._db1["rz"] = data.rz
        self._db1["place_x"] = data.place_x
        self._db1["place_y"] = data.place_y
        self._db1["place_z"] = data.place_z
        self._db1["cmd"] = 1

        self._write_log.append({
            "cycle_id": cycle_id,
            "data": dict(self._db1),
        })

        if self._auto_done:
            import time
            if self._done_delay > 0:
                time.sleep(self._done_delay)
            self._db1["cmd"] = 2

    def read_cmd(self) -> int:
        return int(self._db1["cmd"])

    def wait_for_done(
        self,
        timeout: float = 10.0,
        poll_interval: float = 0.01,
        cycle_id: int = 0,
    ) -> bool:
        import time
        start = time.time()
        while time.time() - start < timeout:
            if self._db1["cmd"] == 2:
                self._db1["cmd"] = 0
                return True
            time.sleep(poll_interval)
        return False

    def reset_cmd(self):
        self._db1["cmd"] = 0

    @property
    def write_log(self) -> list[dict]:
        return self._write_log

    @property
    def db1(self) -> dict:
        return dict(self._db1)


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------
class TestDB1Data:
    def test_default_values(self):
        """DB1Data 預設值"""
        d = DB1Data()
        assert d.cmd == 0
        assert d.rx == 0.0
        assert d.ry == 0.0

    def test_custom_values(self):
        """自訂值"""
        d = DB1Data(
            pick_x=312.5, pick_y=108.2, pick_z=-72.0,
            rz=15.3,
            place_x=400.0, place_y=100.0, place_z=-50.0,
            cmd=1,
        )
        assert d.pick_x == 312.5
        assert d.rz == 15.3
        assert d.cmd == 1


class TestMockOPCUA:
    @pytest.fixture
    def bridge(self):
        b = MockOPCUABridge(auto_done=True)
        b.connect()
        yield b
        b.disconnect()

    def test_connect_disconnect(self, bridge):
        assert bridge.is_connected
        bridge.disconnect()
        assert not bridge.is_connected

    def test_write_pick_command(self, bridge):
        """寫入 pick command"""
        data = DB1Data(
            pick_x=312.5, pick_y=108.2, pick_z=-80.0,
            rz=15.3,
            place_x=400.0, place_y=100.0, place_z=-50.0,
        )
        bridge.write_pick_command(data, cycle_id=42)

        # auto_done → cmd=2
        assert bridge.read_cmd() == 2

        # 驗證寫入紀錄
        assert len(bridge.write_log) == 1
        log = bridge.write_log[0]
        assert log["cycle_id"] == 42
        assert log["data"]["pick_x"] == 312.5
        assert log["data"]["rz"] == 15.3

    def test_wait_for_done(self, bridge):
        """等待完成"""
        data = DB1Data(pick_x=100, pick_y=200, pick_z=-80)
        bridge.write_pick_command(data)
        done = bridge.wait_for_done(timeout=1.0)
        assert done is True
        assert bridge.read_cmd() == 0  # reset 後

    def test_wait_for_done_timeout(self):
        """逾時情境"""
        bridge = MockOPCUABridge(auto_done=False)
        bridge.connect()
        bridge._db1["cmd"] = 1  # 模擬 cmd=1 但不會變 2
        done = bridge.wait_for_done(timeout=0.1)
        assert done is False

    def test_reset_cmd(self, bridge):
        bridge._db1["cmd"] = 2
        bridge.reset_cmd()
        assert bridge.read_cmd() == 0


class TestMultipleCycles:
    def test_sequential_cycles(self):
        """連續多個 cycle"""
        bridge = MockOPCUABridge(auto_done=True)
        bridge.connect()

        for cycle_id in range(1, 6):
            data = DB1Data(
                pick_x=100.0 * cycle_id,
                pick_y=200.0,
                pick_z=-80.0,
                rz=cycle_id * 5.0,
                place_x=400.0,
                place_y=100.0 * cycle_id,
                place_z=-50.0,
            )
            bridge.write_pick_command(data, cycle_id=cycle_id)
            done = bridge.wait_for_done(timeout=1.0, cycle_id=cycle_id)
            assert done

        assert len(bridge.write_log) == 5
        assert bridge.write_log[0]["data"]["pick_x"] == 100.0
        assert bridge.write_log[4]["data"]["pick_x"] == 500.0

    def test_cycle_with_rz_angles(self):
        """驗證 Rz 角度正確傳遞"""
        bridge = MockOPCUABridge(auto_done=True)
        bridge.connect()

        angles = [0.0, 15.0, 45.0, 90.0, -30.0, 180.0]
        for i, angle in enumerate(angles):
            data = DB1Data(
                pick_x=300.0, pick_y=200.0, pick_z=-80.0,
                rx=0.0, ry=0.0, rz=angle,
            )
            bridge.write_pick_command(data, cycle_id=i)

        for i, angle in enumerate(angles):
            assert bridge.write_log[i]["data"]["rz"] == angle
            assert bridge.write_log[i]["data"]["rx"] == 0.0
            assert bridge.write_log[i]["data"]["ry"] == 0.0
