# =============================================================================
#  Unit Tests — 例外陷阱（exception trap coverage）
#
#  驗證各相機元件在邊界條件與錯誤路徑下的正確行為：
#    - FileCamera  : 目錄損壞幀、雙重連線、index 下溢、無限loop後StopIteration
#    - UsbCamera   : 連線失敗後狀態、雙重 disconnect、context manager + 例外
#    - CameraBase  : __exit__ 在 body 拋出例外時仍呼叫 disconnect
#    - picker      : _pick_gui 拋出任意例外時降級 terminal、os.dup 失敗時掃描仍繼續
# =============================================================================
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---------------------------------------------------------------------------
#  共用 helper
# ---------------------------------------------------------------------------
def _write_img(path: Path, arr: np.ndarray) -> None:
    """用 imencode + write_bytes 繞過 Windows 非 ASCII 路徑限制。"""
    suffix = path.suffix.lower() or ".jpg"
    _, buf = cv2.imencode(suffix, arr)
    path.write_bytes(buf.tobytes())


def _make_cap_mock(read_ok: bool = True):
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.read.return_value = (read_ok, np.zeros((480, 640, 3), dtype=np.uint8))
    cap.get.return_value = 0.0
    return cap


# ===========================================================================
#  FileCamera — 例外陷阱
# ===========================================================================
class TestFileCameraExceptionTraps:

    # ------------------------------------------------------------------
    #  目錄中有損壞檔案 → RuntimeError
    # ------------------------------------------------------------------
    def test_corrupt_frame_in_dir_raises_runtime_error(self, tmp_path):
        """目錄模式下讀到損壞幀應拋出 RuntimeError，不是靜默返回。"""
        d = tmp_path / "mixed"
        d.mkdir()
        # 一張正常圖
        _write_img(d / "img_00.jpg", np.zeros((60, 80, 3), dtype=np.uint8))
        # 一張假圖（cv2.imdecode 會返回 None）
        (d / "img_01.jpg").write_bytes(b"not-an-image")

        from camera.file_source import FileCamera
        cam = FileCamera(path=d, loop=False)
        cam.connect()
        cam.get_frame()          # 讀 img_00 → OK
        with pytest.raises(RuntimeError):
            cam.get_frame()      # 讀 img_01 → imdecode 返回 None
        cam.disconnect()

    # ------------------------------------------------------------------
    #  get_frame() 在 disconnect 之後應拋出 RuntimeError
    # ------------------------------------------------------------------
    def test_get_frame_after_disconnect_raises(self, tmp_path):
        p = tmp_path / "x.jpg"
        _write_img(p, np.zeros((10, 10, 3), dtype=np.uint8))

        from camera.file_source import FileCamera
        cam = FileCamera(path=p)
        cam.connect()
        cam.disconnect()
        with pytest.raises(RuntimeError, match="未連線"):
            cam.get_frame()

    # ------------------------------------------------------------------
    #  no-loop 耗盡後 advance() 仍拋出 StopIteration
    # ------------------------------------------------------------------
    def test_advance_exhausted_no_loop_raises_stop_iteration(self, tmp_path):
        d = tmp_path / "imgs"
        d.mkdir()
        for i in range(2):
            _write_img(d / f"img_{i:02d}.jpg",
                       np.full((10, 10, 3), i * 100, dtype=np.uint8))

        from camera.file_source import FileCamera
        cam = FileCamera(path=d, loop=False)
        cam.connect()
        cam.advance()   # img_00
        cam.advance()   # img_01
        with pytest.raises(StopIteration):
            cam.advance()   # 超出邊界
        cam.disconnect()

    # ------------------------------------------------------------------
    #  go_prev() 在 index=0 時不應下溢（clamp to 0）
    # ------------------------------------------------------------------
    def test_go_prev_at_start_does_not_underflow(self, tmp_path):
        d = tmp_path / "imgs"
        d.mkdir()
        for i in range(3):
            _write_img(d / f"img_{i:02d}.jpg",
                       np.full((10, 10, 3), i * 80, dtype=np.uint8))

        from camera.file_source import FileCamera
        cam = FileCamera(path=d, loop=False)
        cam.connect()
        # 尚未讀任何幀，index=0；go_prev → max(0, 0-2)=0 → 回傳第一張
        frame = cam.go_prev()
        assert isinstance(frame, np.ndarray)
        assert cam.current_index == 1   # _capture 執行後 index 已 +1
        cam.disconnect()

    # ------------------------------------------------------------------
    #  雙重 connect() 不應拋出（重新連線語意）
    # ------------------------------------------------------------------
    def test_double_connect_does_not_raise(self, tmp_path):
        p = tmp_path / "x.jpg"
        _write_img(p, np.zeros((10, 10, 3), dtype=np.uint8))

        from camera.file_source import FileCamera
        cam = FileCamera(path=p)
        cam.connect()
        cam.connect()   # second call — should be safe
        assert cam.is_connected
        cam.disconnect()

    # ------------------------------------------------------------------
    #  context manager：body 拋出例外仍呼叫 disconnect
    # ------------------------------------------------------------------
    def test_context_manager_disconnects_on_exception(self, tmp_path):
        p = tmp_path / "x.jpg"
        _write_img(p, np.zeros((10, 10, 3), dtype=np.uint8))

        from camera.file_source import FileCamera
        cam = None
        with pytest.raises(ValueError):
            with FileCamera(path=p) as c:
                cam = c
                assert cam.is_connected
                raise ValueError("intentional body error")
        assert cam is not None
        assert not cam.is_connected

    # ------------------------------------------------------------------
    #  connect 路徑：is_file() True 但 is_dir() 分支不重複
    # ------------------------------------------------------------------
    def test_single_image_is_not_dir_mode(self, tmp_path):
        p = tmp_path / "x.jpg"
        _write_img(p, np.zeros((10, 10, 3), dtype=np.uint8))

        from camera.file_source import FileCamera
        cam = FileCamera(path=p)
        cam.connect()
        assert not cam.is_dir_mode()
        assert cam.file_count == 1
        cam.disconnect()


# ===========================================================================
#  UsbCamera — 例外陷阱
# ===========================================================================
class TestUsbCameraExceptionTraps:

    # ------------------------------------------------------------------
    #  連線失敗後 is_connected 應保持 False
    # ------------------------------------------------------------------
    @patch("camera.usb_cam.cv2.VideoCapture")
    def test_connect_failure_leaves_disconnected(self, MockVC):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        MockVC.return_value = mock_cap

        from camera.usb_cam import UsbCamera
        cam = UsbCamera()
        with pytest.raises(ConnectionError):
            cam.connect()
        assert not cam.is_connected

    # ------------------------------------------------------------------
    #  雙重 disconnect() 不應拋出
    # ------------------------------------------------------------------
    @patch("camera.usb_cam.cv2.VideoCapture")
    def test_double_disconnect_does_not_raise(self, MockVC):
        MockVC.return_value = _make_cap_mock()
        from camera.usb_cam import UsbCamera
        cam = UsbCamera()
        cam.connect()
        cam.disconnect()
        cam.disconnect()    # second call — _cap is None → should be no-op
        assert not cam.is_connected

    # ------------------------------------------------------------------
    #  disconnect() 在從未 connect 時不應拋出
    # ------------------------------------------------------------------
    def test_disconnect_before_connect_does_not_raise(self):
        from camera.usb_cam import UsbCamera
        cam = UsbCamera()
        cam.disconnect()    # never connected
        assert not cam.is_connected

    # ------------------------------------------------------------------
    #  context manager：body 拋出例外仍呼叫 disconnect
    # ------------------------------------------------------------------
    @patch("camera.usb_cam.cv2.VideoCapture")
    def test_context_manager_disconnects_on_exception(self, MockVC):
        MockVC.return_value = _make_cap_mock()
        from camera.usb_cam import UsbCamera
        cam = None
        with pytest.raises(RuntimeError):
            with UsbCamera() as c:
                cam = c
                assert cam.is_connected
                raise RuntimeError("intentional")
        assert cam is not None
        assert not cam.is_connected

    # ------------------------------------------------------------------
    #  _capture 中 ok=True 但 frame 為 None → RuntimeError
    # ------------------------------------------------------------------
    @patch("camera.usb_cam.cv2.VideoCapture")
    def test_capture_none_frame_raises(self, MockVC):
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, None)   # ok=True, frame=None
        cap.get.return_value = 0.0
        MockVC.return_value = cap

        from camera.usb_cam import UsbCamera
        cam = UsbCamera()
        cam.connect()
        with pytest.raises(RuntimeError):
            cam.get_frame()
        cam.disconnect()


# ===========================================================================
#  camera.picker — 例外陷阱
# ===========================================================================
class TestPickerExceptionTraps:

    # ------------------------------------------------------------------
    #  pick_source：_pick_gui 拋出非 ImportError 例外時仍降級 terminal
    # ------------------------------------------------------------------
    @patch("camera.picker._pick_gui", side_effect=RuntimeError("TclError"))
    @patch("camera.picker._pick_terminal")
    @patch("camera.picker.scan_usb")
    def test_falls_back_on_any_gui_exception(
        self, mock_scan, mock_terminal, mock_gui
    ):
        mock_scan.return_value = []
        mock_terminal.return_value = "d435"

        from camera.picker import pick_source
        result = pick_source()
        assert mock_terminal.called
        assert result == "d435"

    # ------------------------------------------------------------------
    #  scan_usb：os.dup() 失敗時仍完成掃描（不因 stderr 重導向失敗中斷）
    # ------------------------------------------------------------------
    @patch("camera.picker.os.dup", side_effect=OSError("dup not available"))
    @patch("camera.picker.cv2.VideoCapture")
    def test_scan_usb_continues_when_stderr_redirect_fails(
        self, MockVC, mock_dup
    ):
        def _side(idx, backend):
            cap = MagicMock()
            cap.isOpened.return_value = (idx == 0)
            cap.get.return_value = 0.0
            return cap
        MockVC.side_effect = _side

        from camera.picker import scan_usb
        result = scan_usb(max_index=3)
        assert len(result) == 1
        assert result[0][0] == "usb"

    # ------------------------------------------------------------------
    #  _pick_terminal：連續無效輸入最終正確選擇
    # ------------------------------------------------------------------
    def test_terminal_multiple_invalid_before_valid(self, monkeypatch):
        """連續無效輸入（超出範圍數字、空白）後正確選擇。"""
        inputs = iter(["0", "", "99", "3"])   # 0 invalid, empty, 99 invalid, 3 valid
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        from camera.picker import _pick_terminal
        options = [
            ("usb",    "USB cam-0"),
            ("cognex", "Cognex"),
            ("d435",   "D435"),
        ]
        result = _pick_terminal(options, "Test", allow_file=False)
        assert result == "d435"
