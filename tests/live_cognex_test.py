#!/usr/bin/env python3
"""
即時測試 Cognex IS8505P Native Mode TCP + FTP 連線
使用方式: python tests/live_cognex_test.py
"""
import io
import socket
import time
from ftplib import FTP

IP = "192.168.0.10"
TELNET_PORT = 23
FTP_PORT = 21
USER = "admin"
PASSWORD = ""


def recv_all(sock, timeout=2.0):
    """讀取所有可用資料"""
    sock.settimeout(timeout)
    buf = b""
    try:
        while True:
            data = sock.recv(4096)
            if not data:
                break
            buf += data
    except socket.timeout:
        pass
    return buf.decode("ascii", errors="replace")


def test_native_mode():
    """測試 Native Mode TCP 通訊 (含登入)"""
    print("=" * 60)
    print("  [1/2] Cognex IS8505P Native Mode TCP 測試")
    print("=" * 60)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)

    try:
        sock.connect((IP, TELNET_PORT))
        print(f"[OK] TCP 連線 {IP}:{TELNET_PORT}")

        # Read welcome
        welcome = recv_all(sock, timeout=2.0)
        print(f"[<<] {welcome.strip()}")

        # Login - User
        sock.sendall(f"{USER}\r\n".encode("ascii"))
        time.sleep(0.5)
        reply = recv_all(sock, timeout=2.0)
        print(f"[>>] User={USER}  [<<] {reply.strip()}")

        # Login - Password
        sock.sendall(f"{PASSWORD}\r\n".encode("ascii"))
        time.sleep(0.5)
        reply = recv_all(sock, timeout=2.0)
        print(f"[>>] Pass=***    [<<] {reply.strip()}")

        # SO1
        sock.sendall(b"SO1\r\n")
        time.sleep(0.3)
        reply = recv_all(sock, timeout=2.0).strip()
        print(f"[>>] SO1         [<<] {reply}")

        # SE8
        sock.sendall(b"SE8\r\n")
        time.sleep(1.0)
        reply = recv_all(sock, timeout=2.0).strip()
        print(f"[>>] SE8         [<<] {reply}")

        # GV*
        sock.sendall(b"GV*\r\n")
        time.sleep(0.3)
        reply = recv_all(sock, timeout=2.0).strip()
        print(f"[>>] GV*         [<<] {reply}")

        # SO0
        sock.sendall(b"SO0\r\n")
        time.sleep(0.3)
        reply = recv_all(sock, timeout=2.0).strip()
        print(f"[>>] SO0         [<<] {reply}")

        return True

    except Exception as e:
        print(f"[FAIL] {e}")
        return False
    finally:
        sock.close()


def test_ftp_image():
    """測試 FTP 影像下載"""
    print()
    print("=" * 60)
    print("  [2/2] Cognex IS8505P FTP 影像下載測試")
    print("=" * 60)

    try:
        ftp = FTP()
        ftp.connect(IP, FTP_PORT, timeout=5)
        print(f"[OK] FTP 連線 {IP}:{FTP_PORT}")

        ftp.login(USER, PASSWORD)
        print(f"[OK] FTP 登入成功 (user={USER})")

        # 列出根目錄
        print("\n[>>] LIST (根目錄):")
        files = ftp.nlst()
        for f in files[:20]:  # 最多顯示 20 個
            print(f"     {f}")
        if len(files) > 20:
            print(f"     ... (共 {len(files)} 個檔案)")

        # 嘗試下載 sendimage.bmp
        for img_name in ["sendimage.bmp", "image.bmp", "capture.bmp"]:
            try:
                buf = io.BytesIO()
                ftp.retrbinary(f"RETR {img_name}", buf.write)
                size = buf.tell()
                print(f"\n[OK] 下載 {img_name}: {size:,} bytes")

                # 嘗試解碼
                try:
                    import cv2
                    import numpy as np
                    buf.seek(0)
                    img_data = np.frombuffer(buf.read(), dtype=np.uint8)
                    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                    if img is not None:
                        print(f"[OK] 影像解碼成功: shape={img.shape}, dtype={img.dtype}")
                        # 儲存到本地
                        cv2.imwrite("tests/cognex_test_capture.png", img)
                        print("[OK] 已儲存到 tests/cognex_test_capture.png")
                    else:
                        print("[WARN] cv2.imdecode 回傳 None")
                except ImportError:
                    print("[WARN] cv2 未安裝，跳過影像解碼")
                break  # 下載成功就不試其他名稱
            except Exception as e:
                print(f"[WARN] {img_name}: {e}")

        ftp.quit()
        return True

    except Exception as e:
        print(f"[FAIL] {e}")
        return False


def main():
    print()
    tcp_ok = test_native_mode()
    ftp_ok = test_ftp_image()

    print()
    print("=" * 60)
    print(f"  結果: TCP={'PASS' if tcp_ok else 'FAIL'}  FTP={'PASS' if ftp_ok else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
