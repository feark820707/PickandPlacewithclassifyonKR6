# =============================================================================
#  KR6 Pick & Place — YOLO 資料集 train/val 分割工具
#
#  前提：標注完成後，影像與 label 放在同一品類目錄下：
#      datasets/labeled/
#        classA/  classA_0001.jpg  classA_0001.txt  ...
#        classB/  ...
#
#  執行：
#      python tools/split_dataset.py                    # 預設 80/20 分割
#      python tools/split_dataset.py --val-ratio 0.15  # 85/15 分割
#      python tools/split_dataset.py --seed 99         # 固定亂數種子
#
#  輸出：
#      datasets/
#        train/
#          images/  *.jpg
#          labels/  *.txt
#        val/
#          images/  *.jpg
#          labels/  *.txt
# =============================================================================
from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("split_dataset")

CLASSES = ["classA", "classB", "classC", "classD"]


# ---------------------------------------------------------------------------
#  核心：分割單一品類
# ---------------------------------------------------------------------------
def _split_class(
    class_dir: Path,
    train_img: Path, train_lbl: Path,
    val_img: Path,   val_lbl: Path,
    val_ratio: float,
    seed: int,
) -> tuple[int, int]:
    """
    將 class_dir 內的 jpg+txt 配對分割到 train/val。

    Returns:
        (train_count, val_count)
    """
    jpgs = sorted(class_dir.glob("*.jpg"))
    pairs = []
    for jpg in jpgs:
        txt = jpg.with_suffix(".txt")
        if txt.exists():
            pairs.append((jpg, txt))
        else:
            logger.warning("無對應 label，跳過: %s", jpg.name)

    if not pairs:
        logger.warning("[%s] 無有效 (image, label) 配對", class_dir.name)
        return 0, 0

    random.seed(seed)
    random.shuffle(pairs)

    n_val = max(1, round(len(pairs) * val_ratio))
    val_pairs   = pairs[:n_val]
    train_pairs = pairs[n_val:]

    for img, lbl in train_pairs:
        shutil.copy2(img, train_img / img.name)
        shutil.copy2(lbl, train_lbl / lbl.name)

    for img, lbl in val_pairs:
        shutil.copy2(img, val_img / img.name)
        shutil.copy2(lbl, val_lbl / lbl.name)

    return len(train_pairs), len(val_pairs)


# ---------------------------------------------------------------------------
#  主流程
# ---------------------------------------------------------------------------
def run(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    labeled_root = Path(args.labeled)
    datasets_root = Path(args.output)

    # 建立輸出目錄
    train_img = datasets_root / "train" / "images"
    train_lbl = datasets_root / "train" / "labels"
    val_img   = datasets_root / "val"   / "images"
    val_lbl   = datasets_root / "val"   / "labels"
    for d in (train_img, train_lbl, val_img, val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    total_train = total_val = 0

    for cls in CLASSES:
        class_dir = labeled_root / cls
        if not class_dir.exists():
            logger.info("[%s] 目錄不存在，跳過", cls)
            continue

        t, v = _split_class(
            class_dir,
            train_img, train_lbl,
            val_img, val_lbl,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        logger.info("[%s] train=%d  val=%d", cls, t, v)
        total_train += t
        total_val   += v

    logger.info("═" * 40)
    logger.info("分割完成 — train: %d  val: %d  合計: %d",
                total_train, total_val, total_train + total_val)
    logger.info("輸出: %s", datasets_root.resolve())
    logger.info("═" * 40)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="YOLO 資料集 train/val 分割工具",
    )
    parser.add_argument(
        "-l", "--labeled",
        default="datasets/labeled",
        help="標注完成的根目錄（預設 datasets/labeled）",
    )
    parser.add_argument(
        "-o", "--output",
        default="datasets",
        help="輸出根目錄（預設 datasets，會建立 train/ val/）",
    )
    parser.add_argument(
        "--val-ratio",
        default=0.2, type=float,
        help="val 比例（預設 0.2 = 20%%）",
    )
    parser.add_argument(
        "--seed",
        default=42, type=int,
        help="亂數種子（預設 42）",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
