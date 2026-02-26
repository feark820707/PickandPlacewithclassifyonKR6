"""Benchmark Cognex IS8505P: trigger + FTP + decode latency"""
import io, time, socket
from ftplib import FTP
import numpy as np
import cv2

IP = "192.168.0.10"

print("=" * 50)
print("  Cognex IS8505P Latency Benchmark")
print("=" * 50)

# --- 1. FTP BMP download ---
t0 = time.monotonic()
ftp = FTP()
ftp.connect(IP, 21, timeout=5)
ftp.login("admin", "")
buf = io.BytesIO()
ftp.retrbinary("RETR image.bmp", buf.write)
ftp.quit()
t_ftp = (time.monotonic() - t0) * 1000
size_mb = buf.tell() / 1024 / 1024
print(f"\n[1] FTP download (BMP):  {t_ftp:.0f} ms  ({size_mb:.2f} MB)")

# --- 2. Decode ---
t0 = time.monotonic()
buf.seek(0)
data = np.frombuffer(buf.read(), dtype=np.uint8)
img = cv2.imdecode(data, cv2.IMREAD_COLOR)
t_decode = (time.monotonic() - t0) * 1000
print(f"[2] cv2.imdecode:        {t_decode:.0f} ms  ({img.shape})")

# --- 3. List image files on FTP ---
ftp2 = FTP()
ftp2.connect(IP, 21, timeout=5)
ftp2.login("admin", "")
files = ftp2.nlst()
img_files = [f for f in files if any(f.lower().endswith(ext) for ext in (".bmp", ".jpg", ".png", ".jpeg"))]
print(f"\n[3] Image files on FTP:  {img_files}")

# --- 4. Try JPG download ---
for name in ["image.jpg", "sendimage.jpg", "capture.jpg", "image.jpeg"]:
    try:
        b = io.BytesIO()
        t0 = time.monotonic()
        ftp2.retrbinary(f"RETR {name}", b.write)
        t_jpg = (time.monotonic() - t0) * 1000
        print(f"    {name}: {b.tell():,} bytes ({t_jpg:.0f} ms)")
    except Exception as e:
        pass  # not available

ftp2.quit()

# --- 5. SE8 trigger ---
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(5)
sock.connect((IP, 23))

# login
sock.settimeout(2)
try:
    while sock.recv(4096): pass
except socket.timeout: pass
sock.sendall(b"admin\r\n")
time.sleep(0.3)
try: sock.recv(4096)
except socket.timeout: pass
sock.sendall(b"\r\n")
time.sleep(0.3)
try: sock.recv(4096)
except socket.timeout: pass

# SO1
sock.sendall(b"SO1\r\n")
time.sleep(0.2)
try: sock.recv(4096)
except socket.timeout: pass

# Trigger
t0 = time.monotonic()
sock.sendall(b"SE8\r\n")
sock.settimeout(3)
try:
    reply = sock.recv(4096).decode().strip()
except socket.timeout:
    reply = "TIMEOUT"
t_trig = (time.monotonic() - t0) * 1000
print(f"\n[5] SE8 trigger:         {t_trig:.0f} ms  (reply={reply})")

# --- 6. Full cycle: trigger + download + decode ---
time.sleep(0.3)
t_total_start = time.monotonic()

# trigger
sock.sendall(b"SE8\r\n")
sock.settimeout(3)
try: sock.recv(4096)
except: pass
t_after_trigger = time.monotonic()

# ftp
ftp3 = FTP()
ftp3.connect(IP, 21, timeout=5)
ftp3.login("admin", "")
buf3 = io.BytesIO()
ftp3.retrbinary("RETR image.bmp", buf3.write)
ftp3.quit()
t_after_ftp = time.monotonic()

# decode
buf3.seek(0)
d3 = np.frombuffer(buf3.read(), dtype=np.uint8)
img3 = cv2.imdecode(d3, cv2.IMREAD_COLOR)
t_after_decode = time.monotonic()

trig_ms = (t_after_trigger - t_total_start) * 1000
dl_ms = (t_after_ftp - t_after_trigger) * 1000
dec_ms = (t_after_decode - t_after_ftp) * 1000
total_ms = (t_after_decode - t_total_start) * 1000

sock.close()

print(f"\n[6] Full Cycle Breakdown:")
print(f"    SE8 trigger:   {trig_ms:6.0f} ms")
print(f"    FTP download:  {dl_ms:6.0f} ms")
print(f"    Decode:        {dec_ms:6.0f} ms")
print(f"    ─────────────────────")
print(f"    TOTAL:         {total_ms:6.0f} ms  ({1000/total_ms:.1f} FPS)")

print(f"\n{'=' * 50}")
