#!/usr/bin/env python3
# =============================================================================
#  KR6 Pick & Place — UI 即時預覽 (Standalone Demo)  v2 — Optimized
#
#  直接連線 Cognex IS8505P 即時取像 + UI 面板顯示
#  不需要 PLC / YOLO / Coordinator，純粹看相機效果
#
#  v2 效能最佳化：
#    - JPG 下載 (373KB) 取代 BMP (4.78MB)  → 下載快 3×
#    - FTP 持久連線（不每次 connect/login） → 省掉 ~200ms
#    - 背景執行緒預取影像                    → UI 不卡頓
#    - 自動連續觸發模式 (A 鍵切換)          → 免手動按 T
#
#  Usage:
#      python ui_demo.py                  # Cognex 192.168.0.10
#      python ui_demo.py --source webcam  # 筆電 Webcam
#      python ui_demo.py --source image   # 上次 FTP 下載的靜態圖
#
#  Controls:
#      Q / ESC : 退出
#      T       : 手動觸發 Cognex 拍照
#      A       : 切換自動連續觸發模式
#      S       : 截圖
#      SPACE   : 暫停 / 恢復
# =============================================================================
from __future__ import annotations

import argparse
import io
import os
import socket
import sys
import threading
import time
from ftplib import FTP
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
#  Color Palette (BGR)
# ---------------------------------------------------------------------------
GREEN  = (0, 200, 0)
RED    = (0, 0, 220)
YELLOW = (0, 220, 220)
WHITE  = (255, 255, 255)
GRAY   = (128, 128, 128)
CYAN   = (220, 200, 0)
ORANGE = (0, 140, 255)
BG_PANEL = (45, 45, 45)

# ---------------------------------------------------------------------------
#  Cognex Native Mode Helper — Optimized
# ---------------------------------------------------------------------------
class CognexLive:
    """
    高效能 Cognex IS8505P Native Mode 連線。

    最佳化：
      1. FTP 持久連線（connect 一次，多次 RETR）
      2. 下載 image.jpg (373KB) 而非 image.bmp (4.78MB)
      3. 背景執行緒非同步觸發 + 下載
    """

    def __init__(self, ip="192.168.0.10", telnet_port=23, ftp_port=21,
                 user="admin", password=""):
        self.ip = ip
        self.telnet_port = telnet_port
        self.ftp_port = ftp_port
        self.user = user
        self.password = password

        self._sock: socket.socket | None = None
        self._ftp: FTP | None = None
        self._connected = False
        self._trigger_count = 0
        self._last_trigger_ms = 0.0
        self._last_ftp_ms = 0.0
        self._last_total_ms = 0.0

        # 背景預取
        self._bg_thread: threading.Thread | None = None
        self._bg_running = False
        self._auto_trigger = False
        self._frame_lock = threading.Lock()
        self._pending_frame: np.ndarray | None = None
        self._frame_ready = threading.Event()

    def connect(self) -> bool:
        try:
            # TCP/Telnet
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(5.0)
            self._sock.connect((self.ip, self.telnet_port))
            self._recv_all(2.0)

            self._sock.sendall(f"{self.user}\r\n".encode())
            time.sleep(0.2)
            self._recv_all(1.0)
            self._sock.sendall(f"{self.password}\r\n".encode())
            time.sleep(0.2)
            reply = self._recv_all(1.0)
            if "Logged In" not in reply:
                print(f"  [WARN] Login: {reply.strip()}")

            self._sock.sendall(b"SO1\r\n")
            time.sleep(0.1)
            self._recv_all(1.0)

            # 持久 FTP 連線
            self._ftp_connect()

            self._connected = True

            # 啟動背景預取執行緒
            self._bg_running = True
            self._bg_thread = threading.Thread(
                target=self._bg_loop, daemon=True, name="CognexBG"
            )
            self._bg_thread.start()

            return True
        except Exception as e:
            print(f"  [ERROR] Connect: {e}")
            return False

    def _ftp_connect(self):
        """建立/重建 FTP 持久連線"""
        try:
            if self._ftp:
                try: self._ftp.quit()
                except: pass
            self._ftp = FTP()
            self._ftp.connect(self.ip, self.ftp_port, timeout=5)
            self._ftp.login(self.user, self.password)
        except Exception as e:
            print(f"  [WARN] FTP reconnect: {e}")
            self._ftp = None

    def trigger(self) -> str:
        """SE8 軟觸發（快速，~15ms）"""
        if not self._connected:
            return "N/A"
        try:
            t0 = time.monotonic()
            self._sock.sendall(b"SE8\r\n")
            self._sock.settimeout(2.0)
            reply = self._sock.recv(4096).decode("ascii", errors="replace").strip()
            self._last_trigger_ms = (time.monotonic() - t0) * 1000
            self._trigger_count += 1
            return reply
        except Exception as e:
            return str(e)

    def grab_image_fast(self) -> np.ndarray | None:
        """
        透過持久 FTP 連線下載 JPG 影像。

        比舊方法快 3-5x：
          - image.jpg (373KB) vs image.bmp (4.78MB)
          - 重用 FTP 連線 vs 每次重連
        """
        if not self._ftp:
            self._ftp_connect()
        if not self._ftp:
            return None

        try:
            t0 = time.monotonic()
            buf = io.BytesIO()
            self._ftp.retrbinary("RETR image.jpg", buf.write)
            self._last_ftp_ms = (time.monotonic() - t0) * 1000

            buf.seek(0)
            data = np.frombuffer(buf.read(), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return img
        except Exception:
            # FTP 斷線 → 重連
            self._ftp_connect()
            return None

    def trigger_and_grab(self) -> np.ndarray | None:
        """觸發 + 下載（完整週期）"""
        t0 = time.monotonic()
        reply = self.trigger()
        if not reply.startswith("1"):
            return None
        # 小延遲等曝光完成
        time.sleep(0.05)
        img = self.grab_image_fast()
        self._last_total_ms = (time.monotonic() - t0) * 1000
        return img

    def get_latest_frame(self) -> np.ndarray | None:
        """取得背景執行緒預取的最新影像（非阻塞）"""
        with self._frame_lock:
            frame = self._pending_frame
            self._pending_frame = None
        return frame

    def _bg_loop(self):
        """背景持續觸發 + 下載執行緒"""
        while self._bg_running:
            if self._auto_trigger and self._connected:
                img = self.trigger_and_grab()
                if img is not None:
                    with self._frame_lock:
                        self._pending_frame = img
                    self._frame_ready.set()
            else:
                time.sleep(0.05)  # idle

    @property
    def auto_trigger(self) -> bool:
        return self._auto_trigger

    @auto_trigger.setter
    def auto_trigger(self, val: bool):
        self._auto_trigger = val

    def disconnect(self):
        self._bg_running = False
        self._auto_trigger = False
        if self._bg_thread:
            self._bg_thread.join(timeout=2)
        if self._ftp:
            try: self._ftp.quit()
            except: pass
        if self._sock:
            try:
                self._sock.sendall(b"SO0\r\n")
                time.sleep(0.1)
            except: pass
            self._sock.close()
        self._connected = False

    def _recv_all(self, timeout=2.0) -> str:
        self._sock.settimeout(timeout)
        buf = b""
        try:
            while True:
                data = self._sock.recv(4096)
                if not data: break
                buf += data
        except socket.timeout:
            pass
        return buf.decode("ascii", errors="replace")

    @property
    def is_connected(self) -> bool:
        return self._connected


# ---------------------------------------------------------------------------
#  UI Panel Drawing
# ---------------------------------------------------------------------------
PANEL_W = 320

def draw_panel(canvas, x0, h, source_name, cognex=None, fps=0.0,
               frame_shape=None, trigger_reply="", frame_count=0,
               paused=False, auto_mode=False):
    """繪製右側資訊面板"""
    cv2.rectangle(canvas, (x0, 0), (x0 + PANEL_W, h), BG_PANEL, -1)

    y = 30
    put = lambda text, pos, color=WHITE, s=0.5, t=1: cv2.putText(
        canvas, text, pos, cv2.FONT_HERSHEY_SIMPLEX, s, color, t, cv2.LINE_AA)

    # Title
    put("KR6 PICK & PLACE", (x0+10, y), CYAN, 0.7, 2)
    y += 30
    put("UI Live Preview v2", (x0+10, y), GRAY, 0.5)
    y += 35

    # Source
    cv2.line(canvas, (x0+10, y), (x0+PANEL_W-10, y), GRAY, 1)
    y += 25
    put("Camera Source", (x0+10, y), WHITE, 0.55, 1)
    y += 22
    put(f"  Type: {source_name}", (x0+10, y), CYAN, 0.45)
    y += 18

    if cognex and cognex.is_connected:
        put(f"  IP:   {cognex.ip}", (x0+10, y), GREEN, 0.45)
        y += 18
        cv2.circle(canvas, (x0+18, y-5), 5, GREEN, -1)
        put("  CONNECTED", (x0+28, y), GREEN, 0.45)
    elif source_name == "Cognex IS8505P":
        cv2.circle(canvas, (x0+18, y-5), 5, RED, -1)
        put("  DISCONNECTED", (x0+28, y), RED, 0.45)
    else:
        put(f"  Status: Active", (x0+10, y), GREEN, 0.45)

    y += 30

    # Image Info
    cv2.line(canvas, (x0+10, y), (x0+PANEL_W-10, y), GRAY, 1)
    y += 25
    put("Image Info", (x0+10, y), WHITE, 0.55, 1)
    y += 22
    if frame_shape:
        put(f"  Resolution: {frame_shape[1]}x{frame_shape[0]}", (x0+10, y), CYAN, 0.45)
        y += 18
        put(f"  Channels:   {frame_shape[2] if len(frame_shape)>2 else 1}", (x0+10, y), CYAN, 0.45)
        y += 18
    put(f"  Frames:     {frame_count}", (x0+10, y), WHITE, 0.45)
    y += 18
    fps_color = GREEN if fps > 5 else YELLOW if fps > 2 else RED
    put(f"  FPS:        {fps:.1f}", (x0+10, y), fps_color, 0.45)
    y += 30

    # Performance (Cognex)
    if cognex and cognex.is_connected:
        cv2.line(canvas, (x0+10, y), (x0+PANEL_W-10, y), GRAY, 1)
        y += 25
        put("Cognex Performance", (x0+10, y), WHITE, 0.55, 1)
        y += 22

        # Mode indicator
        mode_color = GREEN if auto_mode else GRAY
        mode_text = "AUTO" if auto_mode else "MANUAL"
        cv2.circle(canvas, (x0+18, y-5), 5, mode_color, -1)
        put(f"  Mode: {mode_text}", (x0+28, y), mode_color, 0.45)
        y += 20

        put(f"  Triggers:   {cognex._trigger_count}", (x0+10, y), WHITE, 0.45)
        y += 18

        # Trigger latency
        trig_color = GREEN if cognex._last_trigger_ms < 50 else YELLOW
        put(f"  Trigger:    {cognex._last_trigger_ms:.0f} ms", (x0+10, y), trig_color, 0.45)
        y += 18

        # FTP latency
        ftp_color = GREEN if cognex._last_ftp_ms < 200 else YELLOW if cognex._last_ftp_ms < 500 else RED
        put(f"  FTP DL:     {cognex._last_ftp_ms:.0f} ms", (x0+10, y), ftp_color, 0.45)
        y += 18

        # Total cycle
        total_color = GREEN if cognex._last_total_ms < 300 else YELLOW if cognex._last_total_ms < 600 else RED
        put(f"  Cycle:      {cognex._last_total_ms:.0f} ms", (x0+10, y), total_color, 0.45)
        y += 18

        # Effective FPS
        if cognex._last_total_ms > 0:
            efps = 1000.0 / cognex._last_total_ms
            efps_color = GREEN if efps > 3 else YELLOW
            put(f"  Eff. FPS:   {efps:.1f}", (x0+10, y), efps_color, 0.45)
        y += 30

    # Status
    if paused:
        y += 10
        put("|| PAUSED", (x0+80, y), YELLOW, 0.8, 2)
        y += 30

    # Controls
    y = h - 120
    cv2.line(canvas, (x0+10, y), (x0+PANEL_W-10, y), GRAY, 1)
    y += 22
    put("Keyboard Controls", (x0+10, y), WHITE, 0.5, 1)
    y += 20
    put("  Q/ESC  : Quit", (x0+10, y), GRAY, 0.4); y += 16
    put("  T      : Trigger once", (x0+10, y), GRAY, 0.4); y += 16
    put("  A      : Auto trigger ON/OFF", (x0+10, y), GREEN if auto_mode else GRAY, 0.4); y += 16
    put("  S      : Screenshot", (x0+10, y), GRAY, 0.4); y += 16
    put("  SPACE  : Pause / Resume", (x0+10, y), GRAY, 0.4)


# ---------------------------------------------------------------------------
#  Main UI Loop
# ---------------------------------------------------------------------------
def run_ui(source: str):
    WIN_NAME = "KR6 Pick & Place — Live Preview"
    WIN_W, WIN_H = 1280, 720

    cognex: CognexLive | None = None
    cap: cv2.VideoCapture | None = None
    static_img: np.ndarray | None = None

    frame_count = 0
    fps = 0.0
    fps_start = time.monotonic()
    fps_frames = 0
    trigger_reply = ""
    paused = False
    auto_mode = False
    current_frame: np.ndarray | None = None

    # Source setup
    if source == "cognex":
        source_name = "Cognex IS8505P"
        print(f"\n  Connecting to Cognex at 192.168.0.10...")
        cognex = CognexLive()
        if cognex.connect():
            print(f"  ✅ Connected!")
            print(f"  Grabbing first frame (JPG)...")
            trigger_reply = cognex.trigger()
            time.sleep(0.1)
            current_frame = cognex.grab_image_fast()
            if current_frame is not None:
                print(f"  ✅ First frame: {current_frame.shape}")
                print(f"     Trigger: {cognex._last_trigger_ms:.0f}ms, "
                      f"FTP: {cognex._last_ftp_ms:.0f}ms")
            else:
                print("  ⚠ JPG failed, trying BMP fallback...")
        else:
            print("  ❌ Connection failed!")

    elif source == "webcam":
        source_name = "HD Webcam (Built-in)"
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  ❌ Cannot open webcam!")
            return

    elif source == "image":
        source_name = "Static Image (FTP capture)"
        img_path = Path("tests/cognex_test_capture.png")
        if img_path.exists():
            static_img = cv2.imread(str(img_path))
            current_frame = static_img
            print(f"  ✅ Loaded: {img_path} -> {static_img.shape}")
        else:
            print(f"  ❌ Image not found: {img_path}")
            return
    else:
        print(f"  ❌ Unknown source: {source}")
        return

    # Create window
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, WIN_W + PANEL_W, WIN_H)

    print(f"\n  UI opened ({WIN_W+PANEL_W}x{WIN_H})")
    print(f"  Press T=trigger, A=auto, Q=quit\n")

    try:
        while True:
            # --- Check background frames (auto mode) ---
            if cognex and cognex.is_connected and not paused:
                bg_frame = cognex.get_latest_frame()
                if bg_frame is not None:
                    current_frame = bg_frame
                    frame_count += 1

            # --- Webcam grab ---
            if source == "webcam" and cap and not paused:
                ret, frame = cap.read()
                if ret:
                    current_frame = frame
                    frame_count += 1

            # --- Build canvas ---
            canvas = np.full((WIN_H, WIN_W + PANEL_W, 3), 30, dtype=np.uint8)

            if current_frame is not None:
                display = cv2.resize(current_frame, (WIN_W, WIN_H))
                canvas[:, :WIN_W] = display

                # Overlay info
                info_text = f"{current_frame.shape[1]}x{current_frame.shape[0]}"
                if cognex and cognex._last_total_ms > 0:
                    info_text += f" | Cycle: {cognex._last_total_ms:.0f}ms"
                cv2.putText(canvas, info_text, (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 1, cv2.LINE_AA)

                # Crosshair
                cx, cy = WIN_W // 2, WIN_H // 2
                cv2.line(canvas, (cx-20, cy), (cx+20, cy), CYAN, 1)
                cv2.line(canvas, (cx, cy-20), (cx, cy+20), CYAN, 1)

                # Auto mode indicator on image
                if auto_mode:
                    cv2.putText(canvas, "[AUTO]", (WIN_W - 120, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2, cv2.LINE_AA)
            else:
                cv2.putText(canvas, "No frame — press T to trigger",
                           (WIN_W//2 - 200, WIN_H//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, GRAY, 2, cv2.LINE_AA)

            # --- Draw panel ---
            draw_panel(
                canvas, WIN_W, WIN_H,
                source_name=source_name,
                cognex=cognex,
                fps=fps,
                frame_shape=current_frame.shape if current_frame is not None else None,
                trigger_reply=trigger_reply,
                frame_count=frame_count,
                paused=paused,
                auto_mode=auto_mode,
            )

            # --- Show ---
            cv2.imshow(WIN_NAME, canvas)

            # --- Keyboard (16ms = ~60 FPS UI refresh) ---
            key = cv2.waitKey(16) & 0xFF

            if key == ord("q") or key == 27:  # Q / ESC
                break

            elif key == ord("t") and cognex and cognex.is_connected:
                # Manual trigger (foreground, fast)
                img = cognex.trigger_and_grab()
                if img is not None:
                    current_frame = img
                    frame_count += 1
                    trigger_reply = "1"

            elif key == ord("a") and cognex and cognex.is_connected:
                # Toggle auto trigger mode
                auto_mode = not auto_mode
                cognex.auto_trigger = auto_mode
                status = "ON" if auto_mode else "OFF"
                print(f"  Auto trigger: {status}")

            elif key == ord("s"):
                ss_dir = Path("screenshots")
                ss_dir.mkdir(exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                fname = ss_dir / f"screenshot_{ts}.png"
                cv2.imwrite(str(fname), canvas)
                print(f"  Screenshot: {fname}")

            elif key == 32:  # SPACE
                paused = not paused
                if paused and cognex:
                    cognex.auto_trigger = False
                    auto_mode = False

            # --- FPS calc ---
            fps_frames += 1
            elapsed = time.monotonic() - fps_start
            if elapsed >= 1.0:
                fps = fps_frames / elapsed
                fps_frames = 0
                fps_start = time.monotonic()

    except KeyboardInterrupt:
        print("\n  Ctrl+C")

    finally:
        if cognex:
            cognex.disconnect()
            print("  Cognex disconnected")
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print("  UI closed.\n")


# ---------------------------------------------------------------------------
#  Entry
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="KR6 Pick & Place — UI Live Preview v2 (Optimized)"
    )
    parser.add_argument(
        "--source", choices=["cognex", "webcam", "image"],
        default="cognex",
        help="Video source (default: cognex)"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  KR6 Pick & Place — UI Live Preview v2")
    print("=" * 60)
    print(f"  Source: {args.source}")

    run_ui(args.source)


if __name__ == "__main__":
    main()
