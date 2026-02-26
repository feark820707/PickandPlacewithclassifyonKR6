# =============================================================================
#  Unit Tests — camera/picker.py
#  測試 scan_usb、_pick_terminal、pick_source（mock tkinter/VideoCapture）
# =============================================================================
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from io import StringIO

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---------------------------------------------------------------------------
#  scan_usb
# ---------------------------------------------------------------------------
class TestScanUsb:

    @patch("camera.picker.cv2.VideoCapture")
    def test_returns_empty_when_no_cameras(self, MockVC):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        MockVC.return_value = mock_cap

        from camera.picker import scan_usb
        result = scan_usb(max_index=4)
        assert result == []

    @patch("camera.picker.cv2.VideoCapture")
    def test_detects_single_camera(self, MockVC):
        def _side(idx, backend):
            cap = MagicMock()
            cap.isOpened.return_value = (idx == 0)
            cap.get.side_effect = lambda p: {3: 640.0, 4: 480.0, 5: 30.0}.get(p, 0.0)
            return cap
        MockVC.side_effect = _side

        from camera.picker import scan_usb
        result = scan_usb(max_index=4)
        assert len(result) == 1
        src, label = result[0]
        assert src == "usb"
        assert "640" in label
        assert "480" in label

    @patch("camera.picker.cv2.VideoCapture")
    def test_detects_two_cameras(self, MockVC):
        def _side(idx, backend):
            cap = MagicMock()
            cap.isOpened.return_value = (idx in (0, 1))
            cap.get.side_effect = lambda p: {3: 640.0, 4: 480.0, 5: 0.0}.get(p, 0.0)
            return cap
        MockVC.side_effect = _side

        from camera.picker import scan_usb
        result = scan_usb(max_index=4)
        assert len(result) == 2
        assert result[0][0] == "usb"
        assert result[1][0] == "usb:1"

    @patch("camera.picker.cv2.VideoCapture")
    def test_source_names_correct(self, MockVC):
        """index 0 → "usb"，index N → "usb:N" """
        def _side(idx, backend):
            cap = MagicMock()
            cap.isOpened.return_value = (idx < 3)
            cap.get.return_value = 0.0
            return cap
        MockVC.side_effect = _side

        from camera.picker import scan_usb
        result = scan_usb(max_index=5)
        sources = [r[0] for r in result]
        assert sources == ["usb", "usb:1", "usb:2"]


# ---------------------------------------------------------------------------
#  _pick_terminal（文字降級選單）
# ---------------------------------------------------------------------------
class TestPickTerminal:

    def _options(self):
        return [
            ("usb",    "USB cam-0  640×480"),
            ("usb:1",  "USB cam-1  640×480"),
            ("cognex", "Cognex IS8505P"),
        ]

    def test_select_by_number(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "1")
        from camera.picker import _pick_terminal
        result = _pick_terminal(self._options(), "Test", allow_file=False)
        assert result == "usb"

    def test_select_second_option(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "2")
        from camera.picker import _pick_terminal
        result = _pick_terminal(self._options(), "Test", allow_file=False)
        assert result == "usb:1"

    def test_cancel_with_q(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "q")
        from camera.picker import _pick_terminal
        result = _pick_terminal(self._options(), "Test", allow_file=False)
        assert result is None

    def test_direct_path_input(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "/some/path/img.jpg")
        from camera.picker import _pick_terminal
        result = _pick_terminal(self._options(), "Test", allow_file=False)
        assert result == "/some/path/img.jpg"

    def test_invalid_number_retries(self, monkeypatch):
        """輸入超出範圍的數字 → 重試，第二次正確"""
        inputs = iter(["99", "2"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        from camera.picker import _pick_terminal
        result = _pick_terminal(self._options(), "Test", allow_file=False)
        assert result == "usb:1"

    def test_file_option_prompts_path(self, monkeypatch):
        """選 file 選項時，追問路徑"""
        calls = iter(["4", "/data/images"])   # 4 = file 選項
        monkeypatch.setattr("builtins.input", lambda _: next(calls))
        from camera.picker import _pick_terminal
        result = _pick_terminal(self._options(), "Test", allow_file=True)
        assert result == "/data/images"

    def test_empty_input_is_skipped(self, monkeypatch):
        """空輸入不視為有效選擇"""
        inputs = iter(["", "1"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        from camera.picker import _pick_terminal
        result = _pick_terminal(self._options(), "Test", allow_file=False)
        assert result == "usb"


# ---------------------------------------------------------------------------
#  pick_source（整合 GUI/terminal 路由）
# ---------------------------------------------------------------------------
class TestPickSource:

    @patch("camera.picker._pick_gui")
    @patch("camera.picker.scan_usb")
    def test_calls_gui_when_tkinter_available(self, mock_scan, mock_gui):
        mock_scan.return_value = [("usb", "USB cam-0")]
        mock_gui.return_value  = "usb"

        from camera.picker import pick_source
        result = pick_source(title="Test")
        assert mock_gui.called
        assert result == "usb"

    @patch("camera.picker._pick_gui", side_effect=ImportError("no tkinter"))
    @patch("camera.picker._pick_terminal")
    @patch("camera.picker.scan_usb")
    def test_falls_back_to_terminal_on_import_error(
        self, mock_scan, mock_terminal, mock_gui
    ):
        mock_scan.return_value    = []
        mock_terminal.return_value = "cognex"

        from camera.picker import pick_source
        result = pick_source(title="Test")
        assert mock_terminal.called
        assert result == "cognex"

    @patch("camera.picker._pick_gui")
    @patch("camera.picker.scan_usb")
    def test_cancel_returns_none(self, mock_scan, mock_gui):
        mock_scan.return_value = []
        mock_gui.return_value  = None

        from camera.picker import pick_source
        result = pick_source()
        assert result is None

    @patch("camera.picker.scan_usb")
    def test_scan_not_called_when_disabled(self, mock_scan):
        with patch("camera.picker._pick_gui", return_value="cognex"):
            from camera.picker import pick_source
            pick_source(scan_usb_cameras=False)
        mock_scan.assert_not_called()

    @patch("camera.picker._pick_gui")
    @patch("camera.picker.scan_usb")
    def test_fixed_sources_always_included(self, mock_scan, mock_gui):
        """cognex / d435 應始終出現在選項中"""
        mock_scan.return_value = []
        captured_options = []

        def _capture_options(options, title, allow_file):
            captured_options.extend(options)
            return "cognex"

        mock_gui.side_effect = _capture_options

        from camera.picker import pick_source
        pick_source()
        sources = [o[0] for o in captured_options]
        assert "cognex" in sources
        assert "d435"   in sources

    @patch("camera.picker._pick_gui")
    @patch("camera.picker.scan_usb")
    def test_usb_cameras_prepended_before_fixed(self, mock_scan, mock_gui):
        """USB 攝影機在清單最前面，fixed sources 在後"""
        mock_scan.return_value = [("usb", "USB cam-0"), ("usb:1", "USB cam-1")]
        captured = []

        def _capture(options, title, allow_file):
            captured.extend(options)
            return "usb"

        mock_gui.side_effect = _capture
        from camera.picker import pick_source
        pick_source()

        sources = [o[0] for o in captured]
        assert sources.index("usb") < sources.index("cognex")
        assert sources.index("usb:1") < sources.index("d435")
