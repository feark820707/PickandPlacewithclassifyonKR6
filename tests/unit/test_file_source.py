# =============================================================================
#  Unit Tests — camera/file_source.py (FileCamera)
# =============================================================================
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from camera.file_source import FileCamera


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def img_file(tmp_path) -> Path:
    """建立一張 100×80 BGR 測試圖（用 imencode 繞開 Windows 非 ASCII 路徑問題）"""
    p = tmp_path / "test.jpg"
    img = np.zeros((80, 100, 3), dtype=np.uint8)
    img[20:60, 30:70] = (128, 200, 50)     # 畫一個矩形
    _, buf = cv2.imencode(".jpg", img)
    p.write_bytes(buf.tobytes())
    return p


@pytest.fixture
def img_dir(tmp_path) -> Path:
    """建立含 3 張圖的目錄"""
    d = tmp_path / "imgs"
    d.mkdir()
    for i in range(3):
        img = np.full((60, 80, 3), i * 80, dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        (d / f"img_{i:02d}.jpg").write_bytes(buf.tobytes())
    return d


@pytest.fixture
def empty_dir(tmp_path) -> Path:
    d = tmp_path / "empty"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
#  單圖模式
# ---------------------------------------------------------------------------
class TestFileCameraSingleImage:

    def test_connect_single_file(self, img_file):
        cam = FileCamera(path=img_file)
        cam.connect()
        assert cam.is_connected
        assert not cam.is_dir_mode()
        assert cam.file_count == 1
        cam.disconnect()

    def test_get_frame_returns_image(self, img_file):
        cam = FileCamera(path=img_file)
        cam.connect()
        rgb, depth = cam.get_frame()
        assert isinstance(rgb, np.ndarray)
        assert rgb.shape == (80, 100, 3)
        assert depth is None          # FileCamera 永遠 2D
        cam.disconnect()

    def test_single_file_always_same_frame(self, img_file):
        """單圖模式：每次 get_frame 回傳同一張（不會 StopIteration）"""
        cam = FileCamera(path=img_file)
        cam.connect()
        frames = [cam.get_frame()[0] for _ in range(5)]
        cam.disconnect()
        for f in frames[1:]:
            np.testing.assert_array_equal(frames[0], f)

    def test_depth_mode_always_2d(self, img_file):
        """depth_mode 參數無論傳什麼，FileCamera 永遠 2D"""
        cam = FileCamera(path=img_file, depth_mode="3D")
        cam.connect()
        _, depth = cam.get_frame()
        assert depth is None
        cam.disconnect()

    def test_disconnect_clears_state(self, img_file):
        cam = FileCamera(path=img_file)
        cam.connect()
        cam.disconnect()
        assert not cam.is_connected

    def test_connect_invalid_path_raises(self, tmp_path):
        cam = FileCamera(path=tmp_path / "nonexistent.jpg")
        with pytest.raises(ConnectionError, match="路徑不存在"):
            cam.connect()

    def test_connect_corrupt_file_raises(self, tmp_path):
        p = tmp_path / "bad.jpg"
        p.write_bytes(b"not an image")
        cam = FileCamera(path=p)
        with pytest.raises(ConnectionError, match="無法讀取影像"):
            cam.connect()

    def test_context_manager(self, img_file):
        with FileCamera(path=img_file) as cam:
            assert cam.is_connected
            rgb, _ = cam.get_frame()
            assert rgb is not None
        assert not cam.is_connected

    def test_get_frame_not_connected_raises(self, img_file):
        cam = FileCamera(path=img_file)
        with pytest.raises(RuntimeError, match="未連線"):
            cam.get_frame()


# ---------------------------------------------------------------------------
#  目錄模式
# ---------------------------------------------------------------------------
class TestFileCameraDirectory:

    def test_connect_dir(self, img_dir):
        cam = FileCamera(path=img_dir)
        cam.connect()
        assert cam.is_connected
        assert cam.is_dir_mode()
        assert cam.file_count == 3
        cam.disconnect()

    def test_iterates_in_order(self, img_dir):
        """目錄模式：依字母順序逐張讀取"""
        cam = FileCamera(path=img_dir)
        cam.connect()
        frames = [cam.get_frame()[0] for _ in range(3)]
        cam.disconnect()
        # 每張圖的平均亮度應遞增（img_0=0, img_1=80, img_2=160）
        means = [f.mean() for f in frames]
        assert means[0] < means[1] < means[2]

    def test_loop_wraps_around(self, img_dir):
        """loop=True 時讀完後循環回第一張"""
        cam = FileCamera(path=img_dir, loop=True)
        cam.connect()
        # 讀 3 張（一輪）再讀第 4 張應等於第 1 張
        frame0 = cam.get_frame()[0]
        cam.get_frame(); cam.get_frame()
        frame3 = cam.get_frame()[0]   # 第 4 次 = 循環回第 1 張
        np.testing.assert_array_equal(frame0, frame3)
        cam.disconnect()

    def test_no_loop_raises_stop_iteration(self, img_dir):
        """loop=False 時超出邊界 → StopIteration"""
        cam = FileCamera(path=img_dir, loop=False)
        cam.connect()
        cam.get_frame(); cam.get_frame(); cam.get_frame()   # 讀完 3 張
        with pytest.raises(StopIteration):
            cam.get_frame()
        cam.disconnect()

    def test_advance(self, img_dir):
        cam = FileCamera(path=img_dir)
        cam.connect()
        frame0 = cam.get_frame()[0]    # index → 1
        frame1 = cam.advance()          # index → 2
        assert not np.array_equal(frame0, frame1)
        cam.disconnect()

    def test_go_prev(self, img_dir):
        cam = FileCamera(path=img_dir)
        cam.connect()
        frame0 = cam.get_frame()[0]    # 讀 img_00
        cam.get_frame()                # 讀 img_01
        prev   = cam.go_prev()         # 退回 img_00
        np.testing.assert_array_equal(frame0, prev)
        cam.disconnect()

    def test_reset(self, img_dir):
        cam = FileCamera(path=img_dir)
        cam.connect()
        cam.get_frame(); cam.get_frame()
        cam.reset()
        assert cam.current_index == 0
        cam.disconnect()

    def test_current_filename(self, img_dir):
        cam = FileCamera(path=img_dir)
        cam.connect()
        name = cam.current_filename
        assert name.endswith(".jpg")
        cam.disconnect()

    def test_empty_dir_raises(self, empty_dir):
        cam = FileCamera(path=empty_dir)
        with pytest.raises(ConnectionError, match="無支援"):
            cam.connect()

    def test_supported_extensions(self, tmp_path):
        """只掃描影像副檔名，忽略其他檔案"""
        d = tmp_path / "mixed"
        d.mkdir()
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        _, buf_jpg = cv2.imencode(".jpg", img)
        (d / "a.jpg").write_bytes(buf_jpg.tobytes())
        _, buf_png = cv2.imencode(".png", img)
        (d / "b.png").write_bytes(buf_png.tobytes())
        (d / "readme.txt").write_text("ignore me")
        (d / "data.yaml").write_text("ignore me too")
        cam = FileCamera(path=d)
        cam.connect()
        assert cam.file_count == 2
        cam.disconnect()
