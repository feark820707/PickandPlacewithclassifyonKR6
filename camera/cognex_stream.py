# =============================================================================
#  KR6 Pick & Place — Cognex IS8505MP In-Sight Native Mode 相機驅動
#  Ethernet RJ45 連線（TCP/Telnet + FTP），僅 RGB（無深度硬體）
#
#  ⚠ Cognex In-Sight 8505P 是「智慧相機」，使用 Cognex 專有的
#     In-Sight Native Mode 通訊協定，**不是** 標準 GigE Vision。
#     因此 harvesters + .cti (GenTL) 無法使用。
#
#  通訊方式：
#    1. TCP Socket (port 23, Telnet) → Native Mode 命令控制
#       - SO1  → 上線 (Set Online)
#       - SO0  → 離線
#       - SE8  → 軟觸發拍照
#       - GV*  → 讀取檢測結果
#       - SW8  → 設定觸發模式
#    2. FTP (port 21) → 下載影像 BMP/JPG (sendimage.bmp)
# =============================================================================
from __future__ import annotations

import io
import logging
import socket
import time
from ftplib import FTP
from typing import Optional

import cv2
import numpy as np

from camera.base import CameraBase

logger = logging.getLogger(__name__)

# In-Sight Native Mode 回覆前綴
_ACK = "1"   # 指令成功
_NAK = "0"   # 指令失敗


class CognexCamera(CameraBase):
    """
    Cognex IS8505MP-363-50 相機（In-Sight Native Mode）。

    - GigE Ethernet 連線（TCP/Telnet + FTP）
    - 僅 RGB 高解析度影像（內建光源/鏡頭）
    - 無深度硬體 → depth 永遠為 None
    - 僅支援 DEPTH_MODE="2D"

    通訊協定：
      - TCP Socket (Telnet, port 23) → Native Mode 命令
      - FTP (port 21) → 影像檔案下載

    Usage:
        cam = CognexCamera(
            ip="192.168.0.10",
            telnet_port=23,
            ftp_port=21,
            depth_mode="2D",
        )
        with cam:
            rgb, depth = cam.get_frame()  # depth=None
    """

    # 預設 Native Mode 通訊參數
    _RECV_BUF = 4096
    _CMD_TIMEOUT = 5.0       # 指令回覆等待秒數
    _TRIGGER_SETTLE = 0.3    # 觸發後等待影像穩定 (sec)
    _FTP_IMAGE_PATH = "image.bmp"      # IS8505P 實機確認的影像檔名

    def __init__(
        self,
        ip: str = "192.168.0.10",
        telnet_port: int = 23,
        ftp_port: int = 21,
        ftp_user: str = "admin",
        ftp_password: str = "",
        telnet_user: str = "admin",
        telnet_password: str = "",
        depth_mode: str = "2D",
        # 保留 legacy 參數以相容 factory.py（將被忽略）
        port: int = 3000,
        cti_path: str = "",
    ):
        if depth_mode == "3D":
            raise ValueError(
                "Cognex IS8505MP 無深度硬體，不支援 DEPTH_MODE='3D'"
            )
        super().__init__(depth_mode=depth_mode)
        self._ip = ip
        self._telnet_port = telnet_port
        self._ftp_port = ftp_port
        self._ftp_user = ftp_user
        self._ftp_password = ftp_password
        self._telnet_user = telnet_user
        self._telnet_password = telnet_password
        self._sock: Optional[socket.socket] = None

    # -----------------------------------------------------------------
    #  Native Mode TCP 通訊
    # -----------------------------------------------------------------
    def _send_cmd(self, cmd: str, timeout: Optional[float] = None) -> str:
        """
        發送 Native Mode 命令並等待回覆。

        Args:
            cmd: 指令字串（不含換行）
            timeout: 等待回覆超時 (秒)

        Returns:
            回覆字串（去除前後空白）

        Raises:
            ConnectionError: 通訊失敗
        """
        if self._sock is None:
            raise ConnectionError("Cognex 尚未建立 TCP 連線")

        timeout = timeout or self._CMD_TIMEOUT
        self._sock.settimeout(timeout)

        try:
            # Native Mode 命令以 \r\n 結尾
            self._sock.sendall(f"{cmd}\r\n".encode("ascii"))
            reply = self._sock.recv(self._RECV_BUF).decode("ascii").strip()
            logger.debug("Cognex CMD [%s] → [%s]", cmd, reply)
            return reply
        except socket.timeout:
            raise ConnectionError(
                f"Cognex 命令逾時 ({timeout}s): '{cmd}'"
            )
        except OSError as e:
            raise ConnectionError(f"Cognex 通訊錯誤: {e}")

    def _drain_welcome(self) -> None:
        """讀掉 Telnet 歡迎訊息（如果有的話）"""
        self._sock.settimeout(2.0)
        try:
            while True:
                data = self._sock.recv(self._RECV_BUF)
                if not data:
                    break
                logger.debug(
                    "Cognex welcome: %s",
                    data.decode("ascii", errors="replace"),
                )
        except socket.timeout:
            pass  # 沒有更多歡迎訊息

    def _login(self) -> None:
        """
        執行 In-Sight Telnet 登入程序。

        IS8505P 的 Telnet 連線需要先登入（User + Password）
        才能執行 Native Mode 命令。
        預設帳密: admin / (空白)
        """
        # 讀取歡迎訊息 + "User:" 提示
        self._drain_welcome()

        # 發送使用者名稱
        logger.debug("Cognex 登入 — User: %s", self._telnet_user)
        self._sock.settimeout(self._CMD_TIMEOUT)
        self._sock.sendall(f"{self._telnet_user}\r\n".encode("ascii"))
        time.sleep(0.3)

        # 讀取 "Password:" 提示
        try:
            self._sock.settimeout(2.0)
            reply = self._sock.recv(self._RECV_BUF).decode("ascii", errors="replace")
            logger.debug("Cognex login prompt: %s", reply.strip())
        except socket.timeout:
            pass

        # 發送密碼
        logger.debug("Cognex 登入 — Password: ***")
        self._sock.sendall(f"{self._telnet_password}\r\n".encode("ascii"))
        time.sleep(0.3)

        # 讀取登入結果
        try:
            self._sock.settimeout(2.0)
            reply = self._sock.recv(self._RECV_BUF).decode("ascii", errors="replace").strip()
            logger.info("Cognex 登入回覆: %s", reply)
            if "Logged In" in reply:
                logger.info("Cognex Telnet 登入成功")
            elif "Invalid" in reply:
                raise ConnectionError(
                    f"Cognex Telnet 登入失敗（帳密錯誤）: {reply}"
                )
            else:
                logger.warning("Cognex 登入回覆未預期: %s（嘗試繼續）", reply)
        except socket.timeout:
            logger.warning("Cognex 登入回覆超時（嘗試繼續）")

    # -----------------------------------------------------------------
    #  CameraBase 介面實作
    # -----------------------------------------------------------------
    def connect(self) -> None:
        """透過 In-Sight Native Mode TCP 連線到 Cognex 相機"""
        logger.info(
            "Cognex 連線中... (IP=%s, Telnet=%d, FTP=%d)",
            self._ip, self._telnet_port, self._ftp_port,
        )

        # 1. 建立 TCP Socket 連線 (Telnet port)
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(self._CMD_TIMEOUT)
            self._sock.connect((self._ip, self._telnet_port))
        except OSError as e:
            self._sock = None
            raise ConnectionError(
                f"無法連線到 Cognex ({self._ip}:{self._telnet_port}): {e}"
            )

        # 2. Telnet 登入 (User + Password)
        self._login()

        # 3. 上線 (Set Online) — SO1
        #    回覆: 1=成功, -5=已經是Online狀態（皆可接受）
        reply = self._send_cmd("SO1")
        if reply.startswith(_ACK) or reply == "-5":
            logger.info("Cognex SO1: %s", reply)
        else:
            logger.warning("Cognex SO1 回覆非預期: %s (繼續嘗試)", reply)

        # 4. 設定軟觸發模式 — SW8 (Software Trigger)
        reply = self._send_cmd("SW8")
        if not reply.startswith(_ACK):
            logger.warning("Cognex SW8 回覆非預期: %s", reply)

        # 5. 驗證連線 — GV 讀取版本或狀態
        try:
            reply = self._send_cmd("GV001")
            logger.info("Cognex GV001 回覆: %s", reply)
        except ConnectionError:
            logger.warning("Cognex GV001 查詢失敗，但 TCP 連線正常")

        self._connected = True
        logger.info("Cognex 連線成功 — Native Mode (IP=%s)", self._ip)

    def disconnect(self) -> None:
        """釋放 Native Mode TCP 連線"""
        if self._sock is not None:
            try:
                # 離線 — SO0
                self._send_cmd("SO0")
            except Exception as e:
                logger.warning("Cognex SO0 離線命令失敗: %s", e)

            try:
                self._sock.close()
            except Exception as e:
                logger.warning("Cognex socket 關閉失敗: %s", e)
            self._sock = None

        self._connected = False
        logger.info("Cognex 已斷線")

    def _capture(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        透過 Native Mode 觸發拍照 + FTP 下載影像。

        流程：
          1. SE8 → 軟觸發拍照
          2. 等待影像穩定
          3. FTP 下載 sendimage.bmp
          4. 解碼為 OpenCV BGR ndarray

        Returns:
            (rgb, None):
              - rgb: np.ndarray, shape (H, W, 3), dtype uint8, BGR
              - depth: 永遠為 None（Cognex 無深度硬體）
        """
        # 1. 軟觸發拍照
        reply = self._send_cmd("SE8")
        if not reply.startswith(_ACK):
            logger.warning("Cognex SE8 觸發回覆: %s", reply)

        # 2. 等待影像處理完成
        time.sleep(self._TRIGGER_SETTLE)

        # 3. 透過 FTP 下載影像
        rgb = self._download_image_ftp()

        return rgb, None  # Cognex 無深度

    def _download_image_ftp(self) -> np.ndarray:
        """
        從 Cognex FTP 伺服器下載影像。

        Returns:
            np.ndarray: BGR 格式影像, shape (H, W, 3), dtype uint8

        Raises:
            ConnectionError: FTP 連線失敗
            RuntimeError: 影像下載或解碼失敗
        """
        buf = io.BytesIO()

        try:
            ftp = FTP()
            ftp.connect(self._ip, self._ftp_port, timeout=self._CMD_TIMEOUT)
            ftp.login(self._ftp_user, self._ftp_password)

            # 切換到二進位模式下載影像
            ftp.retrbinary(
                f"RETR {self._FTP_IMAGE_PATH}",
                buf.write,
            )
            ftp.quit()
        except Exception as e:
            raise ConnectionError(
                f"Cognex FTP 影像下載失敗 ({self._ip}): {e}"
            )

        # 4. 解碼影像
        buf.seek(0)
        img_bytes = np.frombuffer(buf.read(), dtype=np.uint8)
        rgb = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        if rgb is None:
            raise RuntimeError(
                "Cognex 影像解碼失敗 — FTP 下載的檔案可能不是有效影像"
            )

        logger.debug(
            "Cognex 影像擷取成功: shape=%s", rgb.shape,
        )
        return rgb

    # -----------------------------------------------------------------
    #  額外方法
    # -----------------------------------------------------------------
    def get_inspection_results(self) -> str:
        """
        讀取 In-Sight 檢測結果 (GV*)。

        Returns:
            原始結果字串

        Usage:
            results = cam.get_inspection_results()
        """
        return self._send_cmd("GV*")

    def trigger_and_wait(self, max_wait: float = 5.0) -> bool:
        """
        觸發拍照並等待檢測完成。

        Args:
            max_wait: 最大等待秒數

        Returns:
            True 檢測完成，False 超時
        """
        reply = self._send_cmd("SE8")
        if not reply.startswith(_ACK):
            return False

        start = time.monotonic()
        while time.monotonic() - start < max_wait:
            try:
                reply = self._send_cmd("GV*")
                if reply.startswith(_ACK):
                    return True
            except ConnectionError:
                pass
            time.sleep(0.1)

        return False

    @property
    def ip(self) -> str:
        return self._ip

    @property
    def telnet_port(self) -> int:
        return self._telnet_port

    @property
    def ftp_port(self) -> int:
        return self._ftp_port
