# KR6 Pick & Place 視覺系統 — 專案規格書

## 硬體環境
| 項目 | 規格 |
|------|------|
| 機器手臂 | KUKA KR6（含 KRC4 控制器） |
| 相機（二選一） | **選項A**：Intel RealSense D435（USB 3.0，RGB+Depth）|
|                | **選項B**：Cognex IS8505MP-363-50（GigE Ethernet RJ45，RGB only） |
| 運算主機 | i7-13620H（10C/16T）、48GB RAM、RTX 4070 Laptop 8GB、Win11 Pro |
| PLC | Siemens S7-1515 |
| 網路 | PROFINET（PLC ↔ KRC4）、OPC-UA（PC ↔ PLC）、GigE（PC ↔ Cognex，選項B時） |
| 末端執行器 | 真空吸盤（平面物件，**Rz 可旋轉對齊物件角度**） |
| USB | USB 3.1 Gen1 xHCI 1.2（選項A所需） |
| 有線網路 | Realtek PCIe GbE 1Gbps（OPC-UA + 選項B GigE Vision 共用或分流） |

### 相機選項比較
| 比較項 | RealSense D435（選項A） | Cognex IS8505MP-363-50（選項B） |
|--------|------------------------|--------------------------------|
| 介面 | USB 3.0 | GigE (Ethernet RJ45) |
| 影像輸出 | RGB 1280×720 + Depth 同步 | RGB 高解析度（內建光源/鏡頭） |
| 深度能力 | ✅ 內建結構光深度感測 | ❌ 無深度硬體 |
| 可用模式 | **3D**（用深度）或 **2D**（忽略深度） | 僅 **2D** |
| SDK | `pyrealsense2` | `cognex_dataman` / `harvesters`（GigE Vision） |
| 適用場景 | 多種厚度物件混合（3D）/ 固定厚度（2D） | 固定厚度或少量品類，高精度 2D 定位 |

### 運作模式組合（CAMERA_TYPE × DEPTH_MODE）
| 組合 | CAMERA_TYPE | DEPTH_MODE | Z 軸策略 | 適用場景 |
|------|-------------|------------|----------|----------|
| ① D435 + 3D | `"d435"` | `"3D"` | 每物件即時深度量測 → 動態 Z | 多種厚度混合，最高精度 |
| ② D435 + 2D | `"d435"` | `"2D"` | 忽略深度，用 `THICKNESS_MAP` 查表 → 固定 Z | D435 已裝但物件厚度單一，省運算 |
| ③ Cognex + 2D | `"cognex"` | `"2D"` | 無深度硬體，用 `THICKNESS_MAP` 查表 → 固定 Z | 高精度 2D 定位，固定厚度 |
| ④ Cognex + 3D | `"cognex"` | `"3D"` | ❌ **不允許**（啟動時報錯） | — |

> **規則**：`DEPTH_MODE="3D"` 需要相機提供深度資料；Cognex 無深度硬體，故組合④啟動時會拋出 `ConfigError`。

---

## 設計原則

1. **分類 與 抓取 完全解耦**：YOLO 決定「是什麼、放哪裡」；幾何分析決定「怎麼抓、抓哪點」，兩者獨立運算後合併。
2. **吸盤姿態 Rz 可旋轉**：
   - 機器人末端 **Rx=0, Ry=0 固定**（吸盤永遠朝下），**Rz 由視覺計算**，讓吸盤對齊物件長軸方向。
   - YOLO 使用 **OBB（Oriented Bounding Box）** 偵測物件旋轉角 θ → 傳給機器人作為 Rz。
3. **鏡頭−吸盤不同軸偏移補償**：
   - 相機光軸與吸盤中心存在固定偏移 `(OFFSET_X, OFFSET_Y)`（手眼標定量測）。
   - 當 Rz ≠ 0 時，偏移向量隨旋轉角 θ 旋轉，需在 XY 座標上做旋轉補償：
     ```
     compensated_x = pick_x + OFFSET_X·cos(θ) - OFFSET_Y·sin(θ)
     compensated_y = pick_y + OFFSET_X·sin(θ) + OFFSET_Y·cos(θ)
     ```
   - 若 Rz=0 且偏移為零，公式退化為無補償（向下相容）。
4. **CAMERA_TYPE 與 DEPTH_MODE 雙軸設計**：
   - `CAMERA_TYPE`（`"d435"` | `"cognex"`）→ 決定「用哪台相機取像」。
   - `DEPTH_MODE`（`"3D"` | `"2D"`）→ 決定「Z 軸怎麼算」。
   - **3D 模式**：使用相機深度資料即時量測物件表面高度 → 動態 Z 補償（僅 D435 支援）。
   - **2D 模式**：不使用深度資料，依 `THICKNESS_MAP[label]` 查表 → 固定 Z 補償（D435 / Cognex 皆可）。
   - 啟動時自動驗證組合合法性（Cognex + 3D → 報錯）。
5. **CPU/GPU 明確分工**：YOLO 跑 GPU、幾何跑 CPU ProcessPool、UI/通訊/協調各綁定獨立核心。
6. **相機抽象層（CameraBase）**：統一介面 `get_frame() → (rgb, depth_or_none)`，3D 模式回傳深度圖，2D 模式回傳 `None`，下游模組依 `DEPTH_MODE` 決定 Z 計算方式。
7. **YOLO OBB → θ → Rz 完整鏈路**：YOLO-OBB 輸出旋轉角 θ → GeometryWorker 計算偏移補償 → Coordinator 組裝 `rz=θ` → OPC-UA 寫入 PLC → KR6 執行帶旋轉的 Pick。
8. **可觀測性（Observability）**：每個模組透過統一 Logger 輸出結構化日誌（JSON），關鍵事件附帶 cycle_id 追蹤；Coordinator 維護系統狀態機與健康心跳，異常時自動降級或安全停機。
9. **容錯與恢復（Fault Tolerance）**：每層通訊設置 timeout + retry + circuit breaker 模式；Queue 滿載時丟棄最舊幀（背壓控制）；PLC 通訊中斷時進入 SAFE_STOP 狀態，恢復後自動重連繼續。

---

## 系統架構

```
CAMERA_TYPE="d435"              CAMERA_TYPE="cognex"
D435 (USB, RGB+Depth)           Cognex IS8505MP (GigE, RGB)
         │                              │
         └────────────┬─────────────────┘
                      ▼
                CameraBase（抽象層）
                 get_frame() → (rgb, depth_or_none)
                      │
              ┌───────┴───────┐
              │               │
     DEPTH_MODE="3D"   DEPTH_MODE="2D"
     (depth 有值)       (depth=None)
              │               │
              └───────┬───────┘
                      ▼
┌──────────────────────────────────────────────────────┐
│      Windows 筆電 (i7-13620H / RTX 4070 / 48GB)      │
│                                                      │
│  [GPU]  YOLOWorker (OBB) ──→ Queue_A                 │
│         (rgb → label, obb_bbox, θ, place_pos)         │
│                            │                         │
│  [CPU Pool] GeometryPool ◀─┘                         │
│         θ → Rz + 偏移補償(OFFSET_X/Y)                │
│         3D: depth → 動態 robot_z                      │
│         2D: THICKNESS_MAP[label] → 固定 robot_z       │
│                  │                                   │
│                  ▼ Queue_B                           │
│  [Core 12] Coordinator（任務指派）                    │
│                  │                                   │
│         ┌────────┴────────┐                          │
│         ▼                 ▼                          │
│  [Core 13] OPC-UA    [Core 14] UI                   │
└──────────┬───────────────────────────────────────────┘
           │ OPC-UA TCP
           ▼
     S7-1515 (PLC)
           │ PROFINET
           ▼
       KRC4 / KR6
```

---

## 資料流

```
1. CameraBase.get_frame() 擷取 (rgb, depth_or_none)
      DEPTH_MODE="3D" + D435  ：rgb + aligned depth map
      DEPTH_MODE="2D" + D435  ：rgb + None（忽略深度硬體）
      DEPTH_MODE="2D" + Cognex：rgb + None（無深度硬體）
2. RGB → YOLOWorker OBB (GPU)
      → 輸出：label, obb_bbox(旋轉框), θ(物件角度°), place_pos, confidence
3. (obb_bbox + θ + depth_or_none + label) → GeometryPool (CPU)
      a. 輪廓 → 中心點像素座標 → H矩陣 → 機器人基座標 (pick_x_raw, pick_y_raw)
      b. Z 軸：
         DEPTH_MODE="3D"：depth → 即時量測 → 動態 robot_z
         DEPTH_MODE="2D"：THICKNESS_MAP[label] → 查表 → 固定 robot_z
      c. Rz = θ（物件旋轉角直接作為機器人 Rz）
      d. 鏡頭−吸盤偏移補償（OFFSET_X, OFFSET_Y ≠ 0 時）：
         pick_x = pick_x_raw + OFFSET_X·cos(θ) - OFFSET_Y·sin(θ)
         pick_y = pick_y_raw + OFFSET_X·sin(θ) + OFFSET_Y·cos(θ)
      → 輸出：robot_x, robot_y, robot_z, rz=θ
4. Coordinator 合併：
      → { label, pick:{x,y,z,rx=0,ry=0,rz=θ}, place:{x,y,z}, theta_deg }
5. OPC-UA 寫入 S7-1515 DB1
6. PLC SCL 讀取 → 轉發給 KRC4 (PROFINET)
7. KR6 執行：移動→旋轉Rz→下降→真空→抬起→旋轉回→放置
8. KR6 完成 → PLC done bit → OPC-UA → PC → 下一循環
```

---

## 系統維護架構

### 一、狀態機（System State Machine）

Coordinator 以有限狀態機管理系統生命週期，所有狀態轉換皆寫入日誌：

```
                    ┌──────────────────────────────┐
                    │         INITIALIZING          │
                    │  (validate_config, 載入模型,  │
                    │   連線相機/PLC)               │
                    └──────────┬───────────────────┘
                               │ 全部成功
                               ▼
              ┌────────────────────────────────┐
              │            READY               │
              │  (等待 PLC cmd 或手動觸發)      │
              └───────────┬────────────────────┘
                          │ 收到 cmd=1 / 偵測到物件
                          ▼
              ┌────────────────────────────────┐
         ┌──▶ │           RUNNING              │ ◀─┐
         │    │  (擷取→推論→幾何→寫入 PLC)      │    │
         │    └──┬────────────┬────────────────┘    │
         │       │            │                     │
         │  正常完成      發生錯誤                連續成功
         │       │            │                  恢復後
         │       ▼            ▼                     │
         │   ┌────────┐  ┌──────────────┐           │
         └── │ READY  │  │   ERROR      │ ──────────┘
             └────────┘  │ (分級處理)   │
                         └──────┬───────┘
                                │ 不可恢復 / 手動停機
                                ▼
                         ┌──────────────┐
                         │  SAFE_STOP   │
                         │ (真空關閉,    │
                         │  手臂歸位)    │
                         └──────────────┘
```

```python
# coordinator/state_machine.py
from enum import Enum, auto

class SystemState(Enum):
    INITIALIZING = auto()  # 啟動中：載入設定、連線
    READY        = auto()  # 就緒：等待任務
    RUNNING      = auto()  # 執行中：pick & place 循環
    ERROR        = auto()  # 錯誤：依嚴重等級處理
    SAFE_STOP    = auto()  # 安全停機：真空關閉、手臂歸位

class ErrorLevel(Enum):
    WARN     = auto()  # 警告：跳過此物件，繼續下一個
    RETRY    = auto()  # 重試：重新擷取/推論（最多 3 次）
    CRITICAL = auto()  # 嚴重：進入 SAFE_STOP
```

### 二、結構化日誌系統（Structured Logging）

所有模組使用統一 Logger，輸出 **JSON 格式** 日誌，方便日後用 ELK/Grafana 分析：

```python
# utils/logger.py
import logging, json, time
from pathlib import Path

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level":     record.levelname,
            "module":    record.name,
            "cycle_id":  getattr(record, "cycle_id", None),
            "message":   record.getMessage(),
            "data":      getattr(record, "data", {}),
        }
        return json.dumps(log_entry, ensure_ascii=False)

def setup_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    Path(log_dir).mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # 檔案：全部等級，每日輪替
    from logging.handlers import TimedRotatingFileHandler
    fh = TimedRotatingFileHandler(
        f"{log_dir}/{name}.jsonl", when="midnight", backupCount=30
    )
    fh.setFormatter(JsonFormatter())
    logger.addHandler(fh)
    
    # Console：INFO 以上
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(JsonFormatter())
    logger.addHandler(ch)
    
    return logger

# 各模組使用範例：
# logger = setup_logger("yolo_worker")
# logger.info("偵測完成", extra={"cycle_id": 42, "data": {"label": "classA", "θ": 15.3}})
```

### 三、每個 Cycle 的追蹤鏈（Tracing）

每次 Pick & Place 循環分配一個 `cycle_id`（遞增整數），從擷取到完成全程攜帶：

```
cycle_id=42:
  [Camera]  INFO  擷取完成, fps=30
  [YOLO]    INFO  偵測: classA, θ=15.3°, conf=0.92
  [Geom]    INFO  pick=(312.5, 108.2, -72.0), rz=15.3
  [OPC-UA]  INFO  寫入 DB1 完成
  [PLC]     INFO  cmd=1 已發送
  [PLC]     INFO  cmd=2 完成回報, 耗時 2.3s
  [Coord]   INFO  cycle_id=42 完成, 總耗時 3.1s
```

### 四、健康監控與心跳（Health Monitor）

```python
# utils/health_monitor.py
import time, threading

class HealthMonitor:
    """各模組定期回報心跳，超時則觸發告警"""
    
    def __init__(self, timeout_sec: float = 5.0):
        self._heartbeats: dict[str, float] = {}
        self._timeout = timeout_sec
        self._lock = threading.Lock()
    
    def beat(self, module_name: str):
        """模組回報存活"""
        with self._lock:
            self._heartbeats[module_name] = time.time()
    
    def check_all(self) -> dict[str, str]:
        """回傳各模組健康狀態"""
        now = time.time()
        status = {}
        with self._lock:
            for name, last in self._heartbeats.items():
                if now - last > self._timeout:
                    status[name] = "TIMEOUT"  # → 觸發 ERROR state
                else:
                    status[name] = "OK"
        return status

# 監控項目：
# - Camera:  幀率 < 閾值 → WARN
# - YOLO:    推論時間 > 閾值 → WARN
# - OPC-UA:  連線中斷 → RETRY → CRITICAL
# - PLC:     cmd=2 超時 → RETRY → SAFE_STOP
```

### 五、錯誤恢復策略（Fault Recovery）

| 錯誤場景 | 等級 | 恢復策略 |
|---------|------|----------|
| YOLO 單幀推論失敗 | WARN | 跳過此幀，繼續下一幀 |
| 深度值異常（NaN/超限） | WARN | 使用上次有效 Z 值，或跳過 |
| OPC-UA 連線中斷 | RETRY | 自動重連（3次，間隔2s） |
| PLC cmd=2 超時（>10s） | RETRY | 重發 cmd=1（最多2次） |
| 相機離線 | CRITICAL | → SAFE_STOP，等待人工介入 |
| YOLO 連續 N 幀無偵測 | WARN | 記錄日誌，等待物件進入 |
| GeometryPool worker 崩潰 | RETRY | ProcessPool 自動重啟 worker |
| Queue 積壓超過上限 | WARN | 丟棄最舊幀（背壓控制） |

```python
# utils/retry.py
import time, functools

def retry(max_retries: int = 3, delay: float = 2.0, on_fail=None):
    """通用重試裝飾器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"{func.__name__} 第{attempt}次失敗: {e}")
                    if attempt < max_retries:
                        time.sleep(delay)
                    else:
                        if on_fail:
                            on_fail(e)
                        raise
        return wrapper
    return decorator

# 使用範例：
# @retry(max_retries=3, delay=2.0, on_fail=lambda e: enter_safe_stop())
# def write_to_plc(data): ...
```

### 六、效能指標收集（Metrics）

```python
# utils/metrics.py
import time
from collections import deque

class CycleMetrics:
    """收集每個 cycle 的效能指標，供 UI 顯示與日誌記錄"""
    
    def __init__(self, window_size: int = 100):
        self.capture_ms   = deque(maxlen=window_size)
        self.yolo_ms      = deque(maxlen=window_size)
        self.geometry_ms   = deque(maxlen=window_size)
        self.opcua_ms     = deque(maxlen=window_size)
        self.total_ms     = deque(maxlen=window_size)
        self.cycle_count  = 0
        self.error_count  = 0
    
    def record(self, stage: str, elapsed_ms: float):
        getattr(self, f"{stage}_ms").append(elapsed_ms)
    
    def summary(self) -> dict:
        def avg(d): return sum(d)/len(d) if d else 0
        return {
            "cycles":       self.cycle_count,
            "errors":       self.error_count,
            "avg_capture":  f"{avg(self.capture_ms):.1f}ms",
            "avg_yolo":     f"{avg(self.yolo_ms):.1f}ms",
            "avg_geometry": f"{avg(self.geometry_ms):.1f}ms",
            "avg_opcua":    f"{avg(self.opcua_ms):.1f}ms",
            "avg_total":    f"{avg(self.total_ms):.1f}ms",
            "throughput":   f"{self.cycle_count / max(sum(self.total_ms)/1000, 0.001):.1f} picks/s",
        }
```

### 七、設定檔管理（Configuration Management）

將 `config.py` 硬編碼改為 **YAML 分層設定**，支援多環境切換：

```
configs/
├── default.yaml       # 所有預設值（版本控制）
├── site_A.yaml        # 現場A 覆蓋值（相機=D435, 3D模式）
├── site_B.yaml        # 現場B 覆蓋值（相機=Cognex, 2D模式）
└── local.yaml         # 開發者本機覆蓋（.gitignore）
```

```python
# config.py 載入邏輯
import yaml
from pathlib import Path

def load_config(site: str = None) -> dict:
    """分層載入：default → site → local（後者覆蓋前者）"""
    cfg = {}
    for name in ["default", site, "local"]:
        path = Path(f"configs/{name}.yaml")
        if path.exists():
            with open(path) as f:
                layer = yaml.safe_load(f)
                cfg = deep_merge(cfg, layer)
    validate_config(cfg)
    return cfg
```

```yaml
# configs/default.yaml 範例
camera_type: "d435"
depth_mode: "3D"

plc:
  ip: "192.168.0.10"
  opc_port: 4840

mechanical:
  suction_length_mm: 80.0
  safety_margin_mm: 5.0
  offset_x: 0.0
  offset_y: 0.0

yolo:
  model: "assets/best_obb.pt"
  confidence: 0.5

logging:
  level: "INFO"
  dir: "logs"
  retention_days: 30

health:
  heartbeat_timeout_sec: 5.0
  queue_max_size: 10
  plc_cmd_timeout_sec: 10.0
```

### 八、測試策略（Testing）

```
tests/
├── unit/
│   ├── test_geometry.py       # calc_z_3d, calc_z_2d, compensate_offset 純計算測試
│   ├── test_config.py         # validate_config 合法/非法組合
│   └── test_state_machine.py  # 狀態轉換邏輯
├── integration/
│   ├── test_camera_mock.py    # Mock 相機 → YOLO → 幾何完整管線
│   └── test_opcua_sim.py      # OPC-UA → PLCSIM 讀寫驗證
└── e2e/
    └── test_full_cycle.py     # 含實體相機 + PLC 的端對端測試
```

```python
# tests/unit/test_geometry.py 範例
import pytest, math

def test_compensate_offset_zero_angle():
    """θ=0 時只做平移"""
    x, y = compensate_offset(100, 200, theta_deg=0)
    assert x == pytest.approx(100 + OFFSET_X)
    assert y == pytest.approx(200 + OFFSET_Y)

def test_compensate_offset_90_deg():
    """θ=90° 時偏移向量旋轉 90°"""
    OFFSET_X, OFFSET_Y = 5.0, 3.0
    x, y = compensate_offset(100, 200, theta_deg=90)
    assert x == pytest.approx(100 - 3.0, abs=0.01)
    assert y == pytest.approx(200 + 5.0, abs=0.01)

def test_calc_z_3d():
    z = calc_z_3d(surface_depth_mm=795)
    assert z == pytest.approx(-80.0)

def test_config_cognex_3d_raises():
    with pytest.raises(ConfigError):
        validate_config({"camera_type": "cognex", "depth_mode": "3D"})
```

### 九、版本管理與部署

```
.gitignore 建議排除：
  logs/
  configs/local.yaml
  assets/best_obb.pt     # 模型太大，用 Git LFS 或 DVC 管理
  assets/H.npy           # 現場標定產物，不入版控
  __pycache__/
  *.pyc

Git 分支策略（Git Flow 簡化版）：
  main     → 穩定版本，部署到現場
  develop  → 開發整合
  feature/ → 功能開發（如 feature/cognex-support）
  hotfix/  → 現場緊急修復
```

---

## 物件角度偵測（YOLO-OBB）

### 為什麼用 OBB 而非 AABB？
- 傳統 YOLO 輸出 **AABB（Axis-Aligned Bounding Box）**，只有 `x, y, w, h`，無法知道物件旋轉角度。
- **YOLO-OBB** 輸出 **旋轉邊界框** `(cx, cy, w, h, θ)`，其中 θ 為物件長軸相對水平的旋轉角（度）。
- θ 直接對應機器人的 **Rz**，讓吸盤旋轉對齊物件後再吸取。

### YOLO-OBB 訓練要點
```
# 標註格式（DOTA/OBB 格式）：
# class_id  x1 y1  x2 y2  x3 y3  x4 y4
# 四個角點座標，訓練時自動計算旋轉角

# ultralytics 訓練指令：
yolo obb train data=dataset.yaml model=yolov8m-obb.pt epochs=100
```

### 推論輸出
```python
# yolo_worker.py 中
results = model(rgb_frame)
for obb in results[0].obb:
    cx, cy, w, h, theta = obb.xywhr[0]   # θ in radians
    label = model.names[int(obb.cls)]
    conf  = float(obb.conf)
    theta_deg = math.degrees(theta)        # 轉為度
    # → 傳入 Queue_A: {label, cx, cy, w, h, theta_deg, conf, place_pos}
```

---

## 鏡頭−吸盤偏移補償（XY Offset Compensation）

### 問題描述
相機光軸（拍照中心）與吸盤中心通常 **不在同一軸線上**，存在固定偏移 `(OFFSET_X, OFFSET_Y)`：

```
       相機光軸
          ●
          │← OFFSET_X →│
          │    ┌───┐    │
          │    │   │ OFFSET_Y
          │    └───┘    │
          │        ●    │
               吸盤中心
```

### 補償公式
當機器人末端旋轉 Rz=θ 後，偏移向量也跟著旋轉，必須做旋轉矩陣補償：

```python
import math

def compensate_offset(pick_x_raw: float, pick_y_raw: float,
                      theta_deg: float) -> tuple[float, float]:
    """
    補償鏡頭−吸盤不同軸造成的 XY 偏移。
    
    Args:
        pick_x_raw: H 矩陣轉換後的原始機器人 X
        pick_y_raw: H 矩陣轉換後的原始機器人 Y
        theta_deg:  物件旋轉角度（度）= 機器人 Rz
    
    Returns:
        (compensated_x, compensated_y)
    """
    θ = math.radians(theta_deg)
    comp_x = pick_x_raw + OFFSET_X * math.cos(θ) - OFFSET_Y * math.sin(θ)
    comp_y = pick_y_raw + OFFSET_X * math.sin(θ) + OFFSET_Y * math.cos(θ)
    return comp_x, comp_y

# 特殊情況：
#   θ=0°, OFFSET=(0,0)  → 無補償（向下相容原設計）
#   θ=0°, OFFSET=(5,3)  → comp_x = raw_x + 5, comp_y = raw_y + 3（平移）
#   θ=90°, OFFSET=(5,3) → comp_x = raw_x - 3, comp_y = raw_y + 5（旋轉後偏移）
```

### 參數量測方法
`OFFSET_X`, `OFFSET_Y` 在手眼標定時一併量測：
1. 在標定板上標記一個已知點 P
2. 用相機拍照，計算 P 的影像座標
3. 移動機器人讓吸盤中心對準 P，記錄機器人座標
4. 差值即為 `(OFFSET_X, OFFSET_Y)`

---

## Z 軸補償計算

### DEPTH_MODE = "3D"（動態深度補償）
> 需搭配 D435（depth 有值），每個物件獨立量測表面高度

```python
# geometry_worker.py 中的 Z 計算
def calc_z_3d(surface_depth_mm: float) -> float:
    return (CAMERA_HEIGHT_MM
            - surface_depth_mm
            - SUCTION_LENGTH_MM
            - SAFETY_MARGIN_MM)

# 範例（相機高800mm，吸盤長80mm，安全餘量5mm）：
#   薄件表面深度 795mm → robot_z = 800 - 795 - 80 - 5 = -80mm
#   厚件表面深度 780mm → robot_z = 800 - 780 - 80 - 5 = -65mm
```

### DEPTH_MODE = "2D"（固定 Z 查表補償）
> D435 或 Cognex 皆可使用，不讀取深度資料，依 label 查 THICKNESS_MAP

```python
# geometry_worker.py 中的 Z 計算
def calc_z_2d(label: str) -> float:
    thickness = THICKNESS_MAP.get(label, 0.0)
    return (WORKTABLE_Z_MM
            - SUCTION_LENGTH_MM
            - SAFETY_MARGIN_MM
            + thickness)

# THICKNESS_MAP = {"classA": 5.0, "classB": 10.0, "classC": 20.0}  # mm
#
# 範例（台面 Z=0mm，吸盤長80mm，安全餘量5mm）：
#   classA 厚5mm  → robot_z = 0 - 80 - 5 + 5  = -80mm
#   classC 厚20mm → robot_z = 0 - 80 - 5 + 20 = -65mm
```

### GeometryWorker 統一入口
```python
def compute_pick_pose(obb_result: dict, depth_or_none, H: np.ndarray) -> dict:
    """
    完整計算抓取姿態：XY(含偏移補償) + Z(3D/2D) + Rz(=θ)
    
    Args:
        obb_result: {cx, cy, w, h, theta_deg, label, place_pos}
        depth_or_none: 深度圖或 None
        H: Homography 矩陣
    Returns:
        {pick_x, pick_y, pick_z, rx=0, ry=0, rz, place_pos, label}
    """
    # 1. 像素 → 機器人座標
    pick_x_raw, pick_y_raw = pixel_to_robot(obb_result["cx"], obb_result["cy"], H)
    
    # 2. Z 軸
    if DEPTH_MODE == "3D":
        assert depth_or_none is not None, "3D 模式需要深度資料"
        surface_z = extract_surface_depth(depth_or_none, obb_result)
        pick_z = calc_z_3d(surface_z)
    else:
        pick_z = calc_z_2d(obb_result["label"])
    
    # 3. Rz = 物件旋轉角
    rz = obb_result["theta_deg"]
    
    # 4. 鏡頭−吸盤偏移補償
    pick_x, pick_y = compensate_offset(pick_x_raw, pick_y_raw, rz)
    
    return {
        "pick_x": pick_x, "pick_y": pick_y, "pick_z": pick_z,
        "rx": 0.0, "ry": 0.0, "rz": rz,
        "place_pos": obb_result["place_pos"],
        "label": obb_result["label"]
    }
```

---

## 執行緒/進程資源分配

> **本機 CPU**：i7-13620H = 6P-core (Thread 0~11) + 4E-core (Thread 12~15)，共 16 邏輯處理器

| 模組 | 類別 | CPU Core | 說明 |
|------|------|----------|------|
| YOLOWorker | Thread | GPU (RTX 4070 8GB) | 持續推論，不佔 CPU |
| GeometryPool | ProcessPool | Core 6, 7, 8, 9 | P-core 幾何+深度，CPU密集並行 |
| Coordinator | Thread | Core 12 | E-core 低負載，任務調度 |
| PLCBridge | Thread | Core 13 | E-core I/O等待為主 |
| UIMonitor | Thread | Core 14 | E-core 限速顯示，綁定1核 |

---

## 專案檔案結構

```
kr6_pick_place/
│
├── main.py                  # 入口：載入 config → validate → 啟動狀態機
├── config.py                # 分層載入 YAML → 驗證 → 匯出全域設定
│
├── configs/                 # ★ 分層設定檔（YAML）
│   ├── default.yaml         #   所有預設值（入版控）
│   ├── site_A.yaml          #   現場A 覆蓋（D435+3D）
│   ├── site_B.yaml          #   現場B 覆蓋（Cognex+2D）
│   └── local.yaml           #   開發者本機覆蓋（.gitignore）
│
├── camera/
│   ├── base.py              # CameraBase 抽象類別：get_frame() → (rgb, depth|None)
│   ├── d435_stream.py       # 選項A：D435 USB RGB+Depth 擷取
│   ├── cognex_stream.py     # 選項B：Cognex IS8505MP GigE Vision 擷取
│   └── factory.py           # 工廠函式：依 CAMERA_TYPE 建立對應相機實例
│
├── vision/
│   ├── yolo_worker.py       # GPU Thread：YOLO-OBB 推論 → label + θ + place_map
│   └── geometry_worker.py   # CPU ProcessPool：OBB→抓取點 + Rz + 偏移補償 + Z(3D/2D)
│
├── coordinator/
│   ├── coordinator.py       # 任務指派、Queue 管理、結果合併
│   └── state_machine.py     # ★ 系統狀態機（INIT→READY→RUNNING→ERROR→SAFE_STOP）
│
├── communication/
│   └── opcua_bridge.py      # OPC-UA Thread：寫入 S7-1515（含重連機制）
│
├── ui/
│   └── ui_monitor.py        # UI Thread：即時顯示 + 健康儀表板 + 效能指標
│
├── utils/                   # ★ 共用工具模組
│   ├── logger.py            #   結構化 JSON 日誌（TimedRotating, cycle_id 追蹤）
│   ├── health_monitor.py    #   心跳監控：各模組 timeout 偵測
│   ├── metrics.py           #   效能指標收集（各階段耗時、吞吐量）
│   └── retry.py             #   通用重試裝飾器（retry + circuit breaker）
│
├── calibration/
│   ├── hand_eye_calib.py    # 手眼標定：產生 H.npy + 量測 OFFSET_X/Y
│   └── camera_height.py     # 相機安裝高度量測
│
├── tests/                   # ★ 測試套件
│   ├── unit/
│   │   ├── test_geometry.py     # 幾何計算：Z補償、偏移補償
│   │   ├── test_config.py       # 設定驗證：合法/非法組合
│   │   └── test_state_machine.py# 狀態轉換邏輯
│   ├── integration/
│   │   ├── test_camera_mock.py  # Mock相機→YOLO→幾何管線
│   │   └── test_opcua_sim.py    # OPC-UA + PLCSIM 讀寫
│   └── e2e/
│       └── test_full_cycle.py   # 端對端（實體相機+PLC）
│
├── assets/
│   ├── H.npy                # Homography 矩陣（標定後產生，.gitignore）
│   └── best_obb.pt          # YOLO-OBB 權重（Git LFS 或 DVC）
│
├── logs/                    # ★ 日誌輸出目錄（.gitignore）
│   ├── coordinator.jsonl
│   ├── yolo_worker.jsonl
│   ├── geometry_worker.jsonl
│   └── opcua_bridge.jsonl
│
├── requirements.txt
├── pytest.ini               # ★ 測試設定
└── .gitignore
```

---

## requirements.txt

```
# 共用 — 核心功能
opencv-python
ultralytics
torch
torchvision
opcua
numpy
psutil

# 共用 — 維護基礎設施
pyyaml                # YAML 設定檔載入
pytest                # 單元/整合測試
pytest-cov            # 測試覆蓋率

# 選項A：Intel RealSense D435
pyrealsense2

# 選項B：Cognex IS8505MP（GigE Vision）
harvesters            # GigE Vision GenTL consumer
# 需另安裝 Cognex GigE Vision Driver（提供 .cti GenTL producer）
```

---

## config.py 關鍵參數

```python
# ============================================================
#  核心雙開關：CAMERA_TYPE × DEPTH_MODE
# ============================================================
CAMERA_TYPE = "d435"      # "d435" | "cognex"
DEPTH_MODE  = "3D"        # "3D"  | "2D"
#
# 合法組合：
#   d435  + 3D  → 即時深度量測（推薦：多厚度混合場景）
#   d435  + 2D  → 忽略深度，查表（D435 已裝但物件厚度單一）
#   cognex + 2D → 無深度硬體，查表（高精度 2D 場景）
#   cognex + 3D → ❌ 啟動時拋出 ConfigError

# 網路
PLC_IP   = "192.168.0.10"
OPC_PORT = 4840
OPC_NS   = "urn:siemens:s71500"

# ---------- D435 參數（CAMERA_TYPE="d435" 時使用）----------
D435_WIDTH  = 1280
D435_HEIGHT = 720
D435_FPS    = 30

# ---------- Cognex 參數（CAMERA_TYPE="cognex" 時使用）----------
COGNEX_IP     = "192.168.0.50"      # Cognex 相機 IP
COGNEX_PORT   = 3000                # 通訊埠（依實際設定）
COGNEX_CTI    = r"C:\Program Files\Cognex\GigE Vision\bin\CognexGigEVision.cti"

# ============================================================
#  機械參數（需依實際量測填入）
# ============================================================
# ---- DEPTH_MODE="3D" 時使用 ----
CAMERA_HEIGHT_MM  = 800.0   # 相機到工作台距離（3D 深度換算用）

# ---- DEPTH_MODE="2D" 時使用 ----
WORKTABLE_Z_MM    = 0.0     # 工作台面在機器人座標系的 Z 值
THICKNESS_MAP = {                   # 各品類物件厚度（mm）
    "classA": 5.0,
    "classB": 10.0,
    "classC": 20.0,
}

# ---- 共用 ----
SUCTION_LENGTH_MM = 80.0    # 吸盤長度
SAFETY_MARGIN_MM  = 5.0     # 安全餘量

# ---- 鏡頭−吸盤偏移（手眼標定時量測，單位 mm）----
OFFSET_X = 0.0    # 相機光軸→吸盤中心 X 方向偏移
OFFSET_Y = 0.0    # 相機光軸→吸盤中心 Y 方向偏移
# 若 OFFSET_X=OFFSET_Y=0 → 不補償（同軸設計）

# ============================================================
#  CPU 核心分配（i7-13620H：6P+4E = 16 Thread）
# ============================================================
CORE_UI          = [14]            # E-core
CORE_COORDINATOR = [12]            # E-core
CORE_OPCUA       = [13]            # E-core
CORE_GEOMETRY    = [6, 7, 8, 9]    # P-core × 4

# YOLO-OBB（RTX 4070 Laptop GPU 8GB VRAM）
YOLO_MODEL  = "assets/best_obb.pt"   # OBB 模型權重（非 AABB）
YOLO_CONF   = 0.5

# 放置位置對應（單位 mm，依實際場地填入）
PLACE_MAP = {
    "classA": {"x": 400.0, "y": 100.0, "z": -50.0},
    "classB": {"x": 400.0, "y": 200.0, "z": -50.0},
    "classC": {"x": 400.0, "y": 300.0, "z": -50.0},
}

# ============================================================
#  啟動驗證（main.py 啟動時呼叫）
# ============================================================
def validate_config():
    if CAMERA_TYPE == "cognex" and DEPTH_MODE == "3D":
        raise ConfigError(
            "Cognex 無深度硬體，不支援 DEPTH_MODE='3D'。"
            "請改為 DEPTH_MODE='2D' 或切換 CAMERA_TYPE='d435'。"
        )
    if DEPTH_MODE == "2D":
        missing = [k for k in PLACE_MAP if k not in THICKNESS_MAP]
        if missing:
            raise ConfigError(
                f"DEPTH_MODE='2D' 但 THICKNESS_MAP 缺少類別：{missing}"
            )
    if OFFSET_X == 0.0 and OFFSET_Y == 0.0:
        import warnings
        warnings.warn(
            "OFFSET_X 與 OFFSET_Y 皆為 0，若鏡頭與吸盤不同軸請量測後填入。"
        )
```

---

## S7-1515 DB1 資料塊結構（TIA Portal）

```
DB1
  pick_x   : Real   // 抓取點 X (mm)，含鏡頭−吸盤偏移補償
  pick_y   : Real   // 抓取點 Y (mm)，含鏡頭−吸盤偏移補償
  pick_z   : Real   // 抓取點 Z (mm)，3D動態 或 2D查表
  rx       : Real   // 固定 0.0（吸盤永遠朝下）
  ry       : Real   // 固定 0.0（吸盤永遠朝下）
  rz       : Real   // 物件旋轉角 θ（°），由 YOLO-OBB 計算
  place_x  : Real   // 放置點 X
  place_y  : Real   // 放置點 Y
  place_z  : Real   // 放置點 Z
  cmd      : Int    // 0=idle, 1=執行, 2=完成
```

---

## 待辦（開始寫程式前需確認）

### 通用（不論相機與模式）
- [x] 確認筆電 CPU 核心數 → i7-13620H 10C/16T，已調整核心分配
- [ ] 確認 S7-1515 IP 位址
- [ ] 確認 KRC4 ↔ S7-1515 PROFINET 已接線設定
- [ ] 量測吸盤實際長度（SUCTION_LENGTH_MM）
- [ ] 準備 YOLO-OBB 訓練資料集（**旋轉框標註**，DOTA 格式四角點，物件類別與 PLACE_MAP 對應）
- [ ] 訓練 YOLO-OBB 模型（`yolo obb train`），產生 `best_obb.pt`
- [ ] 執行手眼標定產生 H.npy
- [ ] **量測鏡頭−吸盤偏移量**（OFFSET_X, OFFSET_Y），填入 config.py
- [ ] **決定 CAMERA_TYPE**（`"d435"` / `"cognex"`）
- [ ] **決定 DEPTH_MODE**（`"3D"` / `"2D"`）
- [ ] 建立 `configs/default.yaml` 並依現場需求建立 `site_X.yaml`
- [ ] 撰寫 `tests/unit/` 單元測試（幾何、設定驗證、狀態機）
- [ ] 設定 `.gitignore`（logs/, local.yaml, H.npy, __pycache__/）

### CAMERA_TYPE = "d435" 時
- [ ] 確認 D435 USB 3.0 連線正常
- [ ] 安裝 `pyrealsense2`

### CAMERA_TYPE = "cognex" 時
- [ ] 確認 Cognex 相機 IP 與 GigE 網路設定
- [ ] 安裝 Cognex GigE Vision Driver（.cti 路徑填入 config）
- [ ] 安裝 `harvesters`

### DEPTH_MODE = "3D" 時（僅限 D435）
- [ ] 量測相機實際安裝高度（CAMERA_HEIGHT_MM）
- [ ] 驗證深度精度：D435 深度誤差是否在可接受範圍

### DEPTH_MODE = "2D" 時（D435 或 Cognex 皆可）
- [ ] 量測或登記每個品類的物件厚度（填入 THICKNESS_MAP）
- [ ] 確認 THICKNESS_MAP 涵蓋所有 PLACE_MAP 中的類別
- [ ] 量測工作台面在機器人座標的 Z 值（WORKTABLE_Z_MM）

---

## VSCode 開發建議

建議開發順序：

**Phase 1 — 基礎設施（先建後用）**
1. `configs/default.yaml` → YAML 設定檔，定義所有參數預設值
2. `config.py` → 分層載入邏輯 + `validate_config()`
3. `utils/logger.py` → 結構化 JSON 日誌
4. `utils/retry.py` → 重試裝飾器
5. `utils/health_monitor.py` → 心跳監控
6. `utils/metrics.py` → 效能指標收集
7. `coordinator/state_machine.py` → 系統狀態機
8. `tests/unit/test_config.py` → 設定驗證測試 ← **先有測試再開發功能**

**Phase 2 — 相機與視覺**
9. `camera/base.py` → 定義 CameraBase 抽象介面
10. `camera/d435_stream.py` → 選項A：驗證 D435 影像與深度
11. `camera/cognex_stream.py` → 選項B：驗證 Cognex GigE 影像擷取
12. `camera/factory.py` → 工廠函式：依 CAMERA_TYPE 建立相機
13. `calibration/hand_eye_calib.py` → 產生 H.npy + 量測 OFFSET_X/Y

**Phase 3 — 推論與幾何**
14. `vision/yolo_worker.py` → **YOLO-OBB** 推論，驗證 θ 角度輸出
15. `vision/geometry_worker.py` → OBB→XY + 偏移補償 + Rz + Z(3D/2D)
16. `tests/unit/test_geometry.py` → 幾何計算單元測試

**Phase 4 — 通訊與整合**
17. `communication/opcua_bridge.py` → 與 PLC 通訊測試（含重連機制）
18. `coordinator/coordinator.py` → 整合狀態機 + Queue + 結果合併
19. `tests/integration/test_opcua_sim.py` → PLCSIM 整合測試

**Phase 5 — UI 與全系統**
20. `ui/ui_monitor.py` → 即時顯示 + 健康儀表板 + 效能指標
21. `main.py` → 全系統整合
22. `tests/e2e/test_full_cycle.py` → 端對端測試
