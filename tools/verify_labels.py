# =============================================================================
#  KR6 Pick & Place — YOLO-OBB 標注驗證工具
#
#  功能：
#    - 掃描 labeled/ 或 train|val 目錄
#    - 驗證每個 .txt 格式是否符合 YOLO-OBB（每行 9 個值：class + 8 座標）
#    - 統計各品類數量、孤立影像（無 label）、孤立 label（無影像）
#    - 顯示座標範圍警告（超出 [0,1] 表示標注錯誤）
#
#  Usage:
#      python tools/verify_labels.py                    # 掃描 datasets/labeled/
#      python tools/verify_labels.py -d datasets/train  # 掃描指定目錄
# =============================================================================
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("verify_labels")

CLASSES = ["classA", "classB", "classC", "classD"]


# ---------------------------------------------------------------------------
#  驗證單一 .txt 檔
# ---------------------------------------------------------------------------
def _verify_txt(txt_path: Path) -> list[str]:
    """
    回傳錯誤列表（空 = 無誤）。

    YOLO-OBB 格式（每行）：
        class_id  x1 y1  x2 y2  x3 y3  x4 y4
        （共 9 個數值，座標均為 0~1 正規化）
    """
    errors = []
    try:
        lines = txt_path.read_text(encoding="utf-8").strip().splitlines()
    except Exception as e:
        return [f"無法讀取: {e}"]

    if not lines:
        return ["空檔案（無任何標注）"]

    for i, line in enumerate(lines, 1):
        parts = line.strip().split()
        if len(parts) != 9:
            errors.append(f"行 {i}: 欄位數 {len(parts)}（需 9：class + 8座標）")
            continue

        try:
            cls_id = int(parts[0])
        except ValueError:
            errors.append(f"行 {i}: class_id '{parts[0]}' 非整數")
            continue

        if cls_id < 0 or cls_id >= len(CLASSES):
            errors.append(f"行 {i}: class_id={cls_id} 超出範圍（0~{len(CLASSES)-1}）")

        try:
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            errors.append(f"行 {i}: 座標含非數值")
            continue

        out_of_range = [f"{v:.3f}" for v in coords if not (0.0 <= v <= 1.0)]
        if out_of_range:
            errors.append(f"行 {i}: 座標超出[0,1]: {out_of_range}")

    return errors


# ---------------------------------------------------------------------------
#  掃描目錄
# ---------------------------------------------------------------------------
def _scan_dir(root: Path) -> dict:
    """
    遞迴掃描 root 下所有 jpg + txt，回傳統計資訊。
    """
    jpgs = {p.stem: p for p in root.rglob("*.jpg")}
    txts = {p.stem: p for p in root.rglob("*.txt")}

    paired   = set(jpgs) & set(txts)
    img_only = set(jpgs) - set(txts)   # 有圖無標注
    lbl_only = set(txts) - set(jpgs)   # 有標注無圖

    class_counts: dict[str, int] = defaultdict(int)
    errors_by_file: dict[str, list[str]] = {}

    for stem in sorted(paired):
        txt_errors = _verify_txt(txts[stem])
        if txt_errors:
            errors_by_file[txts[stem].name] = txt_errors
        else:
            # 計算品類分布
            for line in txts[stem].read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if parts:
                    try:
                        cls_id = int(parts[0])
                        cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"cls{cls_id}"
                        class_counts[cls_name] += 1
                    except (ValueError, IndexError):
                        pass

    return {
        "paired":          len(paired),
        "img_only":        sorted(img_only),
        "lbl_only":        sorted(lbl_only),
        "class_counts":    dict(class_counts),
        "errors_by_file":  errors_by_file,
    }


# ---------------------------------------------------------------------------
#  主流程
# ---------------------------------------------------------------------------
def run(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    scan_root = Path(args.directory)
    if not scan_root.exists():
        logger.error("目錄不存在: %s", scan_root)
        sys.exit(1)

    logger.info("掃描目錄: %s", scan_root.resolve())
    result = _scan_dir(scan_root)

    # ── 摘要 ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 50)
    print(f"  YOLO-OBB 標注驗證報告")
    print("═" * 50)
    print(f"  配對 (image+label): {result['paired']}")
    print(f"  孤立影像 (無label) : {len(result['img_only'])}")
    print(f"  孤立label (無影像) : {len(result['lbl_only'])}")

    # 品類分布
    print("\n  品類標注數量：")
    total_annotations = sum(result["class_counts"].values())
    for cls in CLASSES:
        cnt = result["class_counts"].get(cls, 0)
        bar = "█" * min(cnt // 2, 30)
        print(f"    {cls:<8} {cnt:>5}  {bar}")
    print(f"    {'合計':<8} {total_annotations:>5}")

    # 孤立檔案
    if result["img_only"]:
        print(f"\n  ⚠ 無標注影像（前 10）：")
        for name in result["img_only"][:10]:
            print(f"    {name}.jpg")

    if result["lbl_only"]:
        print(f"\n  ⚠ 無對應影像的標注（前 10）：")
        for name in result["lbl_only"][:10]:
            print(f"    {name}.txt")

    # 格式錯誤
    if result["errors_by_file"]:
        print(f"\n  ✗ 格式錯誤 ({len(result['errors_by_file'])} 個檔案)：")
        for fname, errs in list(result["errors_by_file"].items())[:20]:
            print(f"    [{fname}]")
            for e in errs:
                print(f"      • {e}")
    else:
        print(f"\n  ✓ 所有標注格式正確")

    print("═" * 50 + "\n")

    # 回傳碼：有錯誤則非零
    has_issues = bool(result["errors_by_file"] or result["img_only"] or result["lbl_only"])
    sys.exit(1 if has_issues else 0)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="YOLO-OBB 標注驗證工具",
    )
    parser.add_argument(
        "-d", "--directory",
        default="datasets/labeled",
        help="掃描目錄（預設 datasets/labeled）",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
