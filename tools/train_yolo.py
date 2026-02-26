# =============================================================================
#  KR6 Pick & Place — YOLO-OBB 訓練啟動器
#
#  Usage:
#      python tools/train_yolo.py                   # 預設參數
#      python tools/train_yolo.py --epochs 200      # 自訂 epoch
#      python tools/train_yolo.py --model yolov8s-obb.pt  # 小模型
#      python tools/train_yolo.py --resume          # 從上次中斷繼續
#
#  訓練完成後：
#      runs/obb/train/weights/best.pt  → 複製到 assets/best_obb.pt
#      python tools/train_yolo.py --export           # 自動複製 best.pt
# =============================================================================
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("train_yolo")

# ---------------------------------------------------------------------------
#  訓練預設值
# ---------------------------------------------------------------------------
DEFAULTS = {
    "data":    "datasets/dataset.yaml",
    "model":   "yolo11m-obb.pt",       # YOLO11 中型 OBB，預訓練於 DOTA，比 yolov8m 精度更高
    "epochs":  100,
    "imgsz":   640,
    "batch":   16,
    "device":  "0",                    # GPU 0（RTX 4070）
    "workers": 4,
    "patience": 30,                    # early stopping
    "project":  "runs/obb",
    "name":     "train",
    "exist_ok": True,
}

ASSETS_DIR  = PROJECT_ROOT / "assets"
OUTPUT_PT   = ASSETS_DIR / "best_obb.pt"


# ---------------------------------------------------------------------------
#  驗證前置條件
# ---------------------------------------------------------------------------
def _check_prerequisites(args) -> bool:
    ok = True

    data_yaml = PROJECT_ROOT / args.data
    if not data_yaml.exists():
        logger.error("dataset.yaml 不存在: %s", data_yaml)
        ok = False

    train_dir = PROJECT_ROOT / "datasets" / "train" / "images"
    val_dir   = PROJECT_ROOT / "datasets" / "val"   / "images"
    if not train_dir.exists() or not any(train_dir.glob("*.jpg")):
        logger.error("train/images 不存在或無影像。請先執行 split_dataset.py")
        ok = False
    if not val_dir.exists() or not any(val_dir.glob("*.jpg")):
        logger.error("val/images 不存在或無影像。請先執行 split_dataset.py")
        ok = False

    try:
        import torch
        gpu_ok = torch.cuda.is_available()
        logger.info("CUDA 可用: %s  (device=%s)", gpu_ok,
                    torch.cuda.get_device_name(0) if gpu_ok else "N/A")
        if not gpu_ok and args.device != "cpu":
            logger.warning("GPU 不可用，改用 CPU（速度較慢）")
            args.device = "cpu"
    except ImportError:
        logger.warning("torch 未安裝，跳過 GPU 檢查")

    return ok


# ---------------------------------------------------------------------------
#  執行訓練
# ---------------------------------------------------------------------------
def run_train(args):
    cmd = [
        sys.executable, "-m", "ultralytics",
        "obb", "train",
        f"data={args.data}",
        f"model={args.model}",
        f"epochs={args.epochs}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"device={args.device}",
        f"workers={args.workers}",
        f"patience={args.patience}",
        f"project={DEFAULTS['project']}",
        f"name={DEFAULTS['name']}",
        "exist_ok=True",
    ]

    if args.resume:
        cmd.append("resume=True")

    logger.info("執行命令: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


# ---------------------------------------------------------------------------
#  匯出 best.pt → assets/best_obb.pt
# ---------------------------------------------------------------------------
def export_best(run_name: str = "train") -> bool:
    best_pt = PROJECT_ROOT / "runs" / "obb" / run_name / "weights" / "best.pt"
    if not best_pt.exists():
        logger.error("找不到 best.pt: %s", best_pt)
        return False

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, OUTPUT_PT)
    logger.info("best.pt 已複製到: %s", OUTPUT_PT)
    return True


# ---------------------------------------------------------------------------
#  主流程
# ---------------------------------------------------------------------------
def run(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # 只做 export
    if args.export_only:
        ok = export_best()
        sys.exit(0 if ok else 1)

    # 前置檢查
    if not _check_prerequisites(args):
        logger.error("前置條件不符，請修正後重試")
        sys.exit(1)

    # 訓練
    logger.info("開始訓練 YOLO-OBB — model=%s  epochs=%d  imgsz=%d",
                args.model, args.epochs, args.imgsz)
    rc = run_train(args)

    if rc != 0:
        logger.error("訓練失敗（return code=%d）", rc)
        sys.exit(rc)

    # 自動複製 best.pt
    logger.info("訓練完成，複製 best.pt → assets/best_obb.pt")
    export_best()

    logger.info("完成！可執行 main.py --site B 開始推論")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="YOLO-OBB 訓練啟動器",
    )
    parser.add_argument("--data",        default=DEFAULTS["data"],    help="dataset.yaml 路徑")
    parser.add_argument("--model",       default=DEFAULTS["model"],   help="基礎模型（yolov8n/s/m/l-obb.pt）")
    parser.add_argument("--epochs",      default=DEFAULTS["epochs"],  type=int)
    parser.add_argument("--imgsz",       default=DEFAULTS["imgsz"],   type=int)
    parser.add_argument("--batch",       default=DEFAULTS["batch"],   type=int)
    parser.add_argument("--device",      default=DEFAULTS["device"],  help="GPU id 或 'cpu'")
    parser.add_argument("--workers",     default=DEFAULTS["workers"], type=int)
    parser.add_argument("--patience",    default=DEFAULTS["patience"],type=int, help="early stopping patience")
    parser.add_argument("--resume",      action="store_true",         help="從上次中斷繼續訓練")
    parser.add_argument("--export-only", action="store_true",         dest="export_only",
                        help="只複製 best.pt，不重新訓練")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
