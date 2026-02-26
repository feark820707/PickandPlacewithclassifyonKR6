# =============================================================================
#  Unit Tests — camera/usb_cam.py (UsbCamera)
#  使用 unittest.mock 模擬 cv2.VideoCapture，不需要實體攝影機
# =============================================================================
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from camera.usb_cam import UsbCamera


# ---------------------------------------------------------------------------
#  Helper：建立成功的 VideoCapture mock
# ---------------------------------------------------------------------------
def _make_cap_mock(w=640, h=480, fps=30.0, read_ok=True):
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.read.return_value = (read_ok, np.zeros((h, w, 3), dtype=np.uint8))

    def _get(prop):
        return {
            3: float(w),   # CAP_PROP_FRAME_WIDTH
            4: float(h),   # CAP_PROP_FRAME_HEIGHT
            5: fps,        # CAP_PROP_FPS
        }.get(prop, 0.0)

    cap.get.side_effect = _get
    return cap


# ---------------------------------------------------------------------------
#  建構子
# ---------------------------------------------------------------------------
class TestUsbCameraConstructor:

    def test_default_index_zero(self):
        cam = UsbCamera()
        assert cam._index == 0

    def test_integer_index(self):
        cam = UsbCamera(index=2)
        assert cam._index == 2

    def test_string_digit_index(self):
        """字串 "1" 應轉為整數 1"""
        cam = UsbCamera(index="1")
        assert cam._index == 1

    def test_string_url(self):
        """非數字字串保持字串（RTSP URL）"""
        url = "rtsp://192.168.1.1/stream"
        cam = UsbCamera(index=url)
        assert cam._index == url

    def test_depth_mode_forced_2d(self):
        """無論傳什麼 depth_mode，UsbCamera 強制 2D"""
        cam = UsbCamera(depth_mode="3D")
        assert cam.depth_mode == "2D"

    def test_initial_state_not_connected(self):
        cam = UsbCamera()
        assert not cam.is_connected


# ---------------------------------------------------------------------------
#  connect / disconnect
# ---------------------------------------------------------------------------
class TestUsbCameraConnect:

    @patch("camera.usb_cam.cv2.VideoCapture")
    def test_connect_success(self, MockVC):
        MockVC.return_value = _make_cap_mock(640, 480, 30.0)
        cam = UsbCamera(index=0)
        cam.connect()
        assert cam.is_connected

    @patch("camera.usb_cam.cv2.VideoCapture")
    def test_connect_failure_raises(self, MockVC):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        MockVC.return_value = mock_cap
        cam = UsbCamera(index=5)
        with pytest.raises(ConnectionError):
            cam.connect()

    @patch("camera.usb_cam.cv2.VideoCapture")
    def test_disconnect_releases_cap(self, MockVC):
        MockVC.return_value = _make_cap_mock()
        cam = UsbCamera()
        cam.connect()
        cam.disconnect()
        assert not cam.is_connected
        assert cam._cap is None

    @patch("camera.usb_cam.cv2.VideoCapture")
    def test_resolution_applied(self, MockVC):
        """connect 時套用 width/height 設定"""
        import cv2 as _cv2
        mock_cap = _make_cap_mock(1280, 720)
        MockVC.return_value = mock_cap
        cam = UsbCamera(index=0, width=1280, height=720)
        cam.connect()
        # 應呼叫 set(CAP_PROP_FRAME_WIDTH, 1280) 和 set(CAP_PROP_FRAME_HEIGHT, 720)
        calls = [str(c) for c in mock_cap.set.call_args_list]
        assert any("1280" in c for c in calls)
        assert any("720"  in c for c in calls)
        cam.disconnect()

    @patch("camera.usb_cam.cv2.VideoCapture")
    def test_fps_applied(self, MockVC):
        mock_cap = _make_cap_mock(fps=60.0)
        MockVC.return_value = mock_cap
        cam = UsbCamera(index=0, fps=60)
        cam.connect()
        import cv2 as _cv2
        calls = [str(c) for c in mock_cap.set.call_args_list]
        assert any("60" in c for c in calls)
        cam.disconnect()

    @patch("camera.usb_cam.cv2.VideoCapture")
    def test_no_set_called_when_zero(self, MockVC):
        """width/height/fps = 0 時不應呼叫 set"""
        mock_cap = _make_cap_mock()
        MockVC.return_value = mock_cap
        cam = UsbCamera(index=0, width=0, height=0, fps=0)
        cam.connect()
        mock_cap.set.assert_not_called()
        cam.disconnect()


# ---------------------------------------------------------------------------
#  get_frame / _capture
# ---------------------------------------------------------------------------
class TestUsbCameraCapture:

    @patch("camera.usb_cam.cv2.VideoCapture")
    def test_get_frame_returns_image(self, MockVC):
        MockVC.return_value = _make_cap_mock(640, 480)
        cam = UsbCamera()
        cam.connect()
        rgb, depth = cam.get_frame()
        assert isinstance(rgb, np.ndarray)
        assert rgb.shape == (480, 640, 3)
        assert depth is None    # 2D 模式
        cam.disconnect()

    @patch("camera.usb_cam.cv2.VideoCapture")
    def test_capture_failure_raises(self, MockVC):
        mock_cap = _make_cap_mock(read_ok=False)
        MockVC.return_value = mock_cap
        cam = UsbCamera()
        cam.connect()
        with pytest.raises(RuntimeError, match="讀取失敗"):
            cam.get_frame()
        cam.disconnect()

    def test_get_frame_without_connect_raises(self):
        cam = UsbCamera()
        with pytest.raises(RuntimeError, match="未連線"):
            cam.get_frame()

    @patch("camera.usb_cam.cv2.VideoCapture")
    def test_capture_after_disconnect_raises(self, MockVC):
        MockVC.return_value = _make_cap_mock()
        cam = UsbCamera()
        cam.connect()
        cam.disconnect()
        with pytest.raises(RuntimeError):
            cam.get_frame()


# ---------------------------------------------------------------------------
#  resolution 屬性
# ---------------------------------------------------------------------------
class TestUsbCameraProperties:

    @patch("camera.usb_cam.cv2.VideoCapture")
    def test_resolution_before_connect(self, MockVC):
        cam = UsbCamera()
        assert cam.resolution == (0, 0)

    @patch("camera.usb_cam.cv2.VideoCapture")
    def test_resolution_after_connect(self, MockVC):
        MockVC.return_value = _make_cap_mock(1280, 720)
        cam = UsbCamera()
        cam.connect()
        assert cam.resolution == (1280, 720)
        cam.disconnect()

    @patch("camera.usb_cam.cv2.VideoCapture")
    def test_context_manager(self, MockVC):
        MockVC.return_value = _make_cap_mock()
        with UsbCamera() as cam:
            assert cam.is_connected
            rgb, _ = cam.get_frame()
            assert rgb is not None
        assert not cam.is_connected


# ---------------------------------------------------------------------------
#  Windows DSHOW 後端選擇
# ---------------------------------------------------------------------------
class TestUsbCameraBackend:

    @patch("camera.usb_cam.cv2.VideoCapture")
    @patch("camera.usb_cam.sys.platform", "win32")
    def test_windows_uses_dshow(self, MockVC):
        import cv2 as _cv2
        MockVC.return_value = _make_cap_mock()
        cam = UsbCamera(index=0)
        cam.connect()
        _, backend_used = MockVC.call_args[0]
        assert backend_used == _cv2.CAP_DSHOW
        cam.disconnect()

    @patch("camera.usb_cam.cv2.VideoCapture")
    @patch("camera.usb_cam.sys.platform", "linux")
    def test_linux_uses_cap_any(self, MockVC):
        import cv2 as _cv2
        MockVC.return_value = _make_cap_mock()
        cam = UsbCamera(index=0)
        cam.connect()
        _, backend_used = MockVC.call_args[0]
        assert backend_used == _cv2.CAP_ANY
        cam.disconnect()
