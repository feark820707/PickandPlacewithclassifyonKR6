# =============================================================================
#  KR6 Pick & Place — 相機來源選擇器 GUI
#
#  提供一個可重用的 tkinter 對話框，掃描可用裝置後讓用戶點選。
#  無 tkinter 時自動降級為 terminal 文字選單。
#
#  Usage（任何工具中）：
#      from camera.picker import pick_source
#      source = pick_source()          # 顯示 GUI 對話框
#      source = pick_source(title="選擇拍攝來源")
#      # 回傳 "usb", "usb:1", "cognex", "d435", 或路徑字串
#      # 用戶取消時回傳 None
# =============================================================================
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import cv2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  USB 掃描（與 demo.py 相同邏輯，集中在此）
# ---------------------------------------------------------------------------
def scan_usb(max_index: int = 8) -> list[tuple[str, str]]:
    """
    掃描可用 USB 攝影機。

    Returns:
        List of (source_str, display_label)
        e.g. [("usb", "USB cam-0  640×480"), ("usb:1", "USB cam-1  1280×720")]
    """
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    found: list[tuple[str, str]] = []

    # OpenCV C++ 警告走 fd 2（C 層 stderr），須用 os.dup2 抑制
    try:
        stderr_fd    = sys.stderr.fileno()
        saved_fd     = os.dup(stderr_fd)
        devnull_fd   = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, stderr_fd)
        os.close(devnull_fd)
        _redirected  = True
    except Exception:
        saved_fd     = -1
        _redirected  = False

    try:
        for i in range(max_index):
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                src   = "usb" if i == 0 else f"usb:{i}"
                label = f"USB cam-{i}   {w}×{h}   {fps:.0f} fps"
                found.append((src, label))
    finally:
        if _redirected:
            os.dup2(saved_fd, stderr_fd)
            os.close(saved_fd)
    return found


# ---------------------------------------------------------------------------
#  固定來源清單（USB 掃描結果之外）
# ---------------------------------------------------------------------------
_FIXED_SOURCES: list[tuple[str, str]] = [
    ("cognex", "Cognex IS8505P   (Telnet/FTP)"),
    ("d435",   "Intel RealSense D435   (USB 深度)"),
]


# ---------------------------------------------------------------------------
#  GUI 選擇器（tkinter）
# ---------------------------------------------------------------------------
def _pick_gui(
    options: list[tuple[str, str]],
    title: str,
    allow_file: bool,
) -> Optional[str]:
    """tkinter 對話框，回傳選中的 source 字串或 None（取消）。"""
    import tkinter as tk
    from tkinter import filedialog, ttk

    result: list[Optional[str]] = [None]

    root = tk.Tk()
    root.title(title)
    root.resizable(False, False)
    root.attributes("-topmost", True)

    # ── 標題列 ─────────────────────────────────────────────────────
    tk.Label(root, text=title, font=("Arial", 11, "bold"),
             pady=8).pack(fill="x")
    ttk.Separator(root, orient="horizontal").pack(fill="x")

    # ── 來源清單（Listbox）──────────────────────────────────────────
    frame = tk.Frame(root, padx=12, pady=8)
    frame.pack(fill="both")

    scrollbar = tk.Scrollbar(frame, orient="vertical")
    listbox   = tk.Listbox(frame, width=52, height=min(len(options) + 1, 10),
                           yscrollcommand=scrollbar.set,
                           selectmode="single", font=("Consolas", 10),
                           activestyle="dotbox")
    scrollbar.config(command=listbox.yview)
    scrollbar.pack(side="right", fill="y")
    listbox.pack(side="left", fill="both", expand=True)

    for _, label in options:
        listbox.insert("end", f"  {label}")
    if allow_file:
        listbox.insert("end", "  📂  瀏覽圖檔 / 目錄…")

    listbox.selection_set(0)   # 預設選第一項

    # ── 按鈕列 ─────────────────────────────────────────────────────
    btn_frame = tk.Frame(root, padx=12, pady=8)
    btn_frame.pack(fill="x")

    def _confirm():
        sel = listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if allow_file and idx == len(options):
            # 瀏覽檔案 / 目錄
            path = filedialog.askopenfilename(
                title="選擇圖檔",
                filetypes=[("影像檔", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                           ("所有檔案", "*.*")],
            )
            if not path:
                path = filedialog.askdirectory(title="或選擇目錄")
            if path:
                result[0] = path
                root.destroy()
        else:
            result[0] = options[idx][0]
            root.destroy()

    def _cancel():
        root.destroy()

    listbox.bind("<Double-Button-1>", lambda _e: _confirm())
    listbox.bind("<Return>",          lambda _e: _confirm())

    tk.Button(btn_frame, text="確認", width=10, command=_confirm,
              default="active").pack(side="right", padx=4)
    tk.Button(btn_frame, text="取消", width=10, command=_cancel
              ).pack(side="right")

    # 置中螢幕
    root.update_idletasks()
    x = (root.winfo_screenwidth()  - root.winfo_reqwidth())  // 2
    y = (root.winfo_screenheight() - root.winfo_reqheight()) // 2
    root.geometry(f"+{x}+{y}")

    root.mainloop()
    return result[0]


# ---------------------------------------------------------------------------
#  文字降級選單（無 tkinter 時）
# ---------------------------------------------------------------------------
def _pick_terminal(
    options: list[tuple[str, str]],
    title: str,
    allow_file: bool,
) -> Optional[str]:
    print(f"\n{'═'*55}")
    print(f"  {title}")
    print(f"{'═'*55}")
    for i, (_, label) in enumerate(options, 1):
        print(f"  [{i}] {label}")
    if allow_file:
        print(f"  [{len(options)+1}] 瀏覽圖檔 / 目錄（輸入路徑）")
    print()
    while True:
        raw = input("  輸入編號（或直接輸入路徑 / q 取消）：").strip()
        if raw.lower() == "q":
            return None
        if raw.isdigit():
            idx = int(raw) - 1
            if allow_file and idx == len(options):
                return input("  請輸入圖檔或目錄路徑：").strip() or None
            if 0 <= idx < len(options):
                return options[idx][0]
        elif raw:
            return raw


# ---------------------------------------------------------------------------
#  公開 API
# ---------------------------------------------------------------------------
def pick_source(
    title: str = "選擇影像來源",
    allow_file: bool = True,
    scan_usb_cameras: bool = True,
) -> Optional[str]:
    """
    顯示來源選擇對話框（GUI 優先，降級為 terminal）。

    Args:
        title:             對話框標題
        allow_file:        是否顯示「瀏覽圖檔/目錄」選項
        scan_usb_cameras:  是否掃描 USB 攝影機

    Returns:
        source 字串（"usb", "usb:1", "cognex", "d435", 路徑）
        用戶取消時回傳 None
    """
    options: list[tuple[str, str]] = []

    if scan_usb_cameras:
        logger.debug("掃描 USB 攝影機…")
        options.extend(scan_usb())

    options.extend(_FIXED_SOURCES)

    try:
        import tkinter  # noqa: F401
        return _pick_gui(options, title, allow_file)
    except Exception as e:
        logger.debug("tkinter 不可用，降級為 terminal 選單: %s", e)
        return _pick_terminal(options, title, allow_file)
