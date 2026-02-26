# KR6 Pick & Place 視覺系統 — 專案規格書

> **專案狀態：Phase 1–6 全部完成 ✅ | 172/172 測試通過**
> 最後更新：2026-02-26（依據實際程式碼同步）

## 硬體環境
| 項目 | 規格 |
|------|------|
| 機器手臂 | KUKA KR6（含 KRC4 控制器） |
| 相機 | **選項A**：Intel RealSense D435（USB 3.0，RGB+Depth）|
|      | **選項B**：Cognex IS8505P（GigE Ethernet RJ45，RGB only）|
|      | **選項C**：通用 USB/RTSP 攝影機（cv2.VideoCapture，開發測試用）|
| 運算主機 | i7-13620H（10C/16T）、48GB RAM、RTX 4070 Laptop 8GB、Win11 Pro |
| PLC | Siemens S7-1515 |
| 網路 | PROFINET（PLC ↔ KRC4）、OPC-UA（PC ↔ PLC）、TCP/Telnet+FTP（PC ↔ Cognex，選項B時） |
| 末端執行器 | 真空吸盤（平面物件，**Rz 可旋轉對齊物件角度**） |
| USB | USB 3.1 Gen1 xHCI 1.2（選項A所需） |
| 有線網路 | Realtek PCIe GbE 1Gbps（OPC-UA + 選項B TCP 共用或分流） |

### 相機選項比較
| 比較項 | RealSense D435（選項A） | Cognex IS8505P（選項B） |
|--------|------------------------|--------------------------------|
| 介面 | USB 3.0 | GigE Ethernet RJ45 |
| 影像輸出 | RGB 1280×720 + Depth 同步 | RGB 高解析度（內建光源/鏡頭） |
| 深度能力 | ✅ 內建結構光深度感測 | ❌ 無深度硬體 |
| 可用模式 | **3D**（用深度）或 **2D**（忽略深度） | 僅 **2D** |
| 通訊協定 | `pyrealsense2` USB SDK | **In-Sight Native Mode**：TCP/Telnet (port 23) + FTP (port 21)（標準庫） |
| 適用場景 | 多種厚度物件混合（3D）/ 固定厚度（2D） | 固定厚度或少量品類，高精度 2D 定位 |

> **Cognex 通訊說明**：IS8505P 使用 **In-Sight Native Mode**（非標準 GigE Vision），不需 `harvesters` / GenTL Producer。命令控制走 Telnet (port 23)，影像擷取走 FTP (port 21)，均使用 Python 標準庫（`socket`, `ftplib`）。

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
6. **相機抽象層（CameraBase）+ 插件登錄機制**：統一介面 `get_frame() → (rgb, depth_or_none)`；新相機只需繼承 `CameraBase` 並加上 `@register_camera(...)` 裝飾器，系統自動識別，無需修改 factory 或驗證邏輯。
7. **YOLO OBB → θ → Rz 完整鏈路**：YOLO-OBB 輸出旋轉角 θ → GeometryWorker 計算偏移補償 → Coordinator 組裝 `rz=θ` → OPC-UA 寫入 PLC → KR6 執行帶旋轉的 Pick。
8. **可觀測性（Observability）**：每個模組透過統一 Logger 輸出結構化日誌（JSON），關鍵事件附帶 cycle_id 追蹤；Coordinator 維護系統狀態機與健康心跳，異常時自動降級或安全停機。
9. **容錯與恢復（Fault Tolerance）**：每層通訊設置 timeout + retry（含指數退避）+ circuit breaker 模式；Queue 滿載時丟棄最舊幀（背壓控制）；PLC 通訊中斷時進入 SAFE_STOP 狀態，恢復後自動重連繼續。

---

## 系統架構

```
CAMERA_TYPE="d435"              CAMERA_TYPE="cognex"
D435 (USB, RGB+Depth)           Cognex IS8505P (TCP/Telnet+FTP, RGB)
         │                              │
         └────────────┬─────────────────┘
                      ▼
                CameraBase（抽象層 + @register_camera 插件機制）
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
           │ OPC-UA TCP 4840
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
      DEPTH_MODE="2D" + Cognex：rgb + None（TCP/Telnet 取像，無深度）
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

**有效轉換路徑（已實作並測試）：**
```
INITIALIZING → {READY, ERROR, SAFE_STOP}
READY        → {RUNNING, ERROR, SAFE_STOP}
RUNNING      → {READY, ERROR, SAFE_STOP}
ERROR        → {RUNNING, READY, SAFE_STOP}
SAFE_STOP    → {INITIALIZING}
```

### 二、結構化日誌系統（Structured Logging）

所有模組使用統一 Logger，輸出 **JSON 格式** 日誌，方便日後用 ELK/Grafana 分析：

```python
# utils/logger.py
def setup_logger(
    name: str,
    log_dir: str = "logs",
    level: str = "DEBUG",
    retention_days: int = 30,
    console_level: str = "INFO",
    use_json: bool = True,
) -> logging.Logger:
    """
    建立並回傳一個已配置的 Logger。
    - 檔案：TimedRotatingFileHandler（每日輪替，保留 retention_days 天）
    - Console：INFO 以上
    - 格式：JSON（use_json=True）或可讀文字（use_json=False）
    """

# JSON 輸出格式：
# {"timestamp": "...", "level": "INFO", "module": "yolo_worker",
#  "cycle_id": 42, "message": "偵測完成", "data": {"label": "classA", "θ": 15.3}}

# 各模組使用範例：
# logger = setup_logger("yolo_worker", log_dir="logs", level="DEBUG")
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
class HealthMonitor:
    """各模組定期回報心跳，超時則觸發告警"""

    def beat(self, module_name: str):
        """模組回報存活"""

    def check_all(self) -> dict[str, HealthStatus]:
        """回傳各模組健康狀態：OK / WARN / TIMEOUT / UNKNOWN"""

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
| OPC-UA 連線中斷 | RETRY | 自動重連（3次，指數退避）|
| PLC cmd=2 超時（>10s） | RETRY | 重發 cmd=1（最多2次） |
| 相機離線 | CRITICAL | → SAFE_STOP，等待人工介入 |
| YOLO 連續 N 幀無偵測 | WARN | 記錄日誌，等待物件進入 |
| GeometryPool worker 崩潰 | RETRY | ProcessPool 自動重啟 worker |
| Queue 積壓超過上限 | WARN | 丟棄最舊幀（背壓控制） |

```python
# utils/retry.py — 通用重試裝飾器（含指數退避）+ Circuit Breaker

def retry(
    max_retries: int = 3,
    delay: float = 2.0,
    backoff_factor: float = 1.0,   # 1.5 = 指數退避
    exceptions: tuple = (Exception,),
    on_fail=None,
):
    """
    通用重試裝飾器。
    @retry(max_retries=3, delay=2.0, backoff_factor=1.5,
           on_fail=lambda e: enter_safe_stop())
    def write_to_plc(data): ...
    """

class CircuitBreaker:
    """
    Circuit Breaker 模式 — 防止連續失敗導致系統雪崩。
    連續失敗達閾值 → OPEN（斷路）
    冷卻期後 → HALF_OPEN（探測）→ CLOSED（恢復）或繼續 OPEN
    """
    # 狀態：CLOSED / OPEN / HALF_OPEN
```

### 六、效能指標收集（Metrics）

```python
# utils/metrics.py
class CycleMetrics:
    """收集每個 cycle 的效能指標，供 UI 顯示與日誌記錄"""

    STAGES = ("capture", "yolo", "geometry", "opcua", "total")

    # Context Manager 形式量測
    def measure(self, stage: str):
        """with metrics.measure("yolo"): ..."""

    def start_cycle(self):   # 標記 cycle 開始
    def complete_cycle(self): # 自動記錄 total 耗時並計數
    def record_error(self):   # 錯誤計數
    def record_skip(self):    # 跳過計數（空幀、無偵測）
    def reset(self):          # 重置所有指標

    def summary(self) -> dict:
        # 回傳格式：
        # {
        #   "cycles": 142, "errors": 2, "skips": 5,
        #   "avg_capture_ms": 12.3, "avg_yolo_ms": 25.1,
        #   "avg_geometry_ms": 3.2, "avg_opcua_ms": 8.5,
        #   "avg_total_ms": 55.6,
        #   "throughput_picks_per_sec": 2.1,
        #   "p95_total_ms": 72.3,
        # }
```

### 七、設定檔管理（Configuration Management）

分層 YAML 設定，支援多環境切換：

```
configs/
├── default.yaml       # 所有預設值（版本控制）
├── site_A.yaml        # 現場A 覆蓋值（相機=D435, 3D模式）
├── site_B.yaml        # 現場B 覆蓋值（相機=Cognex, 2D模式）
└── local.yaml         # 開發者本機覆蓋（.gitignore）
```

```python
# config.py 載入邏輯
def load_config(site: str = None, config_dir: str = "configs") -> dict:
    """分層載入：default → site_{name} → local（後者覆蓋前者）"""

def validate_config(cfg: dict) -> None:
    """驗證規則：
    - Cognex + 3D        → ConfigError
    - 2D 模式缺 thickness → ConfigError
    - camera_type 未在 registry → ConfigError
    - OFFSET=(0,0)       → Warning（提醒手眼標定）
    """

class AppConfig:
    """Config 存取器，40+ 屬性，含型別轉換"""
```

```yaml
# configs/default.yaml（關鍵預設值摘要）
camera_type: "d435"
depth_mode: "3D"

plc:
  ip: "192.168.0.1"      # S7-1515 PLC
  opc_port: 4840

cognex:
  ip: "192.168.0.10"     # Cognex IS8505P（實機偵測確認）
  telnet_port: 23        # In-Sight Native Mode 命令控制
  ftp_port: 21           # 影像下載
  ftp_user: "admin"
  ftp_password: ""       # IS8505P 預設空白

mechanical:
  suction_length_mm: 80.0
  safety_margin_mm: 5.0
  offset_x: 0.0
  offset_y: 0.0

yolo:
  model: "assets/best_obb.pt"
  confidence: 0.5
  device: "cuda:0"

logging:
  level: "INFO"
  dir: "logs"
  retention_days: 30
  format: "json"

health:
  heartbeat_timeout_sec: 5.0
  queue_max_size: 10

retry:
  max_retries: 3
  delay_sec: 2.0
  backoff_factor: 1.5    # 指數退避倍數
```

### 八、測試策略（Testing）

```
tests/
├── unit/
│   ├── test_geometry.py        # calc_z_3d, calc_z_2d, compensate_offset 純計算（22 tests）
│   ├── test_config.py          # validate_config 合法/非法組合（26 tests）
│   ├── test_state_machine.py   # 狀態轉換邏輯（28 tests）
│   ├── test_file_source.py     # FileCamera 單圖/目錄/導航/錯誤路徑（19 tests）  ★新增
│   ├── test_usb_cam.py         # UsbCamera mock VideoCapture / 後端選擇（21 tests）★新增
│   ├── test_camera_picker.py   # scan_usb / _pick_terminal / pick_source（17 tests）★新增
│   └── test_exception_traps.py # 例外陷阱：邊界條件 / 降級路徑（15 tests）       ★新增
├── integration/
│   ├── test_camera_mock.py     # Mock 相機 → YOLO → 幾何完整管線（8 tests）
│   └── test_opcua_sim.py       # OPC-UA Mock 讀寫驗證（9 tests）
└── e2e/
    └── test_full_cycle.py      # 完整 Pick & Place 循環端對端測試（7 tests）

總計：172 tests，100% 通過 ✅
```

```python
# tests/unit/test_geometry.py 範例（實際已通過）
def test_compensate_offset_zero_angle():
    """θ=0 時只做平移"""
    x, y = compensate_offset(100, 200, theta_deg=0, offset_x=5.0, offset_y=3.0)
    assert x == pytest.approx(105.0)
    assert y == pytest.approx(203.0)

def test_compensate_offset_90_deg():
    """θ=90° 時偏移向量旋轉 90°"""
    x, y = compensate_offset(100, 200, theta_deg=90, offset_x=5.0, offset_y=3.0)
    assert x == pytest.approx(97.0, abs=0.01)   # 100 - 3
    assert y == pytest.approx(205.0, abs=0.01)  # 200 + 5

def test_calc_z_3d():
    z = calc_z_3d(surface_depth_mm=795, camera_height_mm=800,
                  suction_length_mm=80, safety_margin_mm=5)
    assert z == pytest.approx(-80.0)

def test_config_cognex_3d_raises():
    with pytest.raises(ConfigError):
        validate_config({"camera_type": "cognex", "depth_mode": "3D", ...})
```

### 九、版本管理與部署

```
.gitignore 排除：
  logs/
  configs/local.yaml
  assets/best_obb.pt     # 模型太大，用 Git LFS 或 DVC 管理
  assets/H.npy           # 現場標定產物，不入版控
  __pycache__/
  *.pyc
  screenshots/

Git 分支策略（Git Flow 簡化版）：
  main     → 穩定版本，部署到現場
  develop  → 開發整合
  feature/ → 功能開發
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
# vision/yolo_worker.py 中
results = model(rgb_frame)
for obb in results[0].obb:
    cx, cy, w, h, theta = obb.xywhr[0]   # θ in radians
    label = model.names[int(obb.cls)]
    conf  = float(obb.conf)
    theta_deg = math.degrees(theta)        # 轉為度
    # → 封裝為 OBBDetection dataclass，放入 Queue_A
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
# vision/geometry_worker.py
import math

def compensate_offset(
    pick_x_raw: float, pick_y_raw: float,
    theta_deg: float,
    offset_x: float, offset_y: float,
) -> tuple[float, float]:
    θ = math.radians(theta_deg)
    comp_x = pick_x_raw + offset_x * math.cos(θ) - offset_y * math.sin(θ)
    comp_y = pick_y_raw + offset_x * math.sin(θ) + offset_y * math.cos(θ)
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
# vision/geometry_worker.py
def calc_z_3d(
    surface_depth_mm: float,
    camera_height_mm: float,
    suction_length_mm: float,
    safety_margin_mm: float,
) -> float:
    return camera_height_mm - surface_depth_mm - suction_length_mm - safety_margin_mm

# 範例（相機高800mm，吸盤長80mm，安全餘量5mm）：
#   薄件表面深度 795mm → robot_z = 800 - 795 - 80 - 5 = -80mm
#   厚件表面深度 780mm → robot_z = 800 - 780 - 80 - 5 = -65mm
```

### DEPTH_MODE = "2D"（固定 Z 查表補償）
> D435 或 Cognex 皆可使用，不讀取深度資料，依 label 查 THICKNESS_MAP

```python
# vision/geometry_worker.py
def calc_z_2d(
    label: str,
    thickness_map: dict,
    worktable_z_mm: float,
    suction_length_mm: float,
    safety_margin_mm: float,
) -> float:
    thickness = thickness_map.get(label, 0.0)
    return worktable_z_mm - suction_length_mm - safety_margin_mm + thickness

# THICKNESS_MAP = {"classA": 5.0, "classB": 10.0, "classC": 20.0}  # mm
#
# 範例（台面 Z=0mm，吸盤長80mm，安全餘量5mm）：
#   classA 厚5mm  → robot_z = 0 - 80 - 5 + 5  = -80mm
#   classC 厚20mm → robot_z = 0 - 80 - 5 + 20 = -65mm
```

### GeometryWorker 統一入口
```python
# vision/geometry_worker.py
def compute_pick_pose(
    obb_result: dict,
    depth_or_none,
    H: np.ndarray,
    cfg: AppConfig,
) -> PickPose:
    """
    完整計算抓取姿態：XY(含偏移補償) + Z(3D/2D) + Rz(=θ)

    Returns:
        PickPose(pick_x, pick_y, pick_z, rx=0, ry=0, rz, label, place_pos)
    """
    # 1. 像素 → 機器人座標（Homography）
    pick_x_raw, pick_y_raw = pixel_to_robot(obb_result["cx"], obb_result["cy"], H)

    # 2. Z 軸（3D 或 2D）
    if cfg.depth_mode == "3D":
        surface_z = extract_surface_depth(depth_or_none, obb_result["cx"], obb_result["cy"])
        pick_z = calc_z_3d(surface_z, cfg.camera_height_mm, ...)
    else:
        pick_z = calc_z_2d(obb_result["label"], cfg.thickness_map, ...)

    # 3. Rz = 物件旋轉角
    rz = obb_result["theta_deg"]

    # 4. 鏡頭−吸盤偏移補償
    pick_x, pick_y = compensate_offset(pick_x_raw, pick_y_raw, rz,
                                       cfg.offset_x, cfg.offset_y)

    return PickPose(pick_x=pick_x, pick_y=pick_y, pick_z=pick_z,
                    rx=0.0, ry=0.0, rz=rz,
                    label=obb_result["label"],
                    place_pos=obb_result["place_pos"])
```

---

## 執行緒/進程資源分配

> **本機 CPU**：i7-13620H = 6P-core (Thread 0~11) + 4E-core (Thread 12~15)，共 16 邏輯處理器

| 模組 | 類別 | CPU Core | 說明 |
|------|------|----------|------|
| YOLOWorker | Thread | GPU (RTX 4070 8GB) | 持續推論，不佔 CPU |
| GeometryPool | ProcessPool | Core 6, 7, 8, 9 | P-core 幾何+深度，CPU密集並行 |
| Coordinator | Thread | Core 12 | E-core 低負載，任務調度 |
| PLCBridge (OPC-UA) | Thread | Core 13 | E-core I/O等待為主 |
| UIMonitor | Thread | Core 14 | E-core 限速顯示，綁定1核 |

---

## 專案檔案結構

```
kr6_pick_place/
│
├── main.py                  # 入口：CLI 參數 → 載入 config → 狀態機 → 主迴圈
├── config.py                # 分層載入 YAML → 驗證 → AppConfig 存取器
│
├── configs/                 # ★ 分層設定檔（YAML）
│   ├── default.yaml         #   所有預設值（版本控制）
│   ├── site_A.yaml          #   現場A 覆蓋（D435+3D）
│   ├── site_B.yaml          #   現場B 覆蓋（Cognex+2D）
│   └── local.yaml           #   開發者本機覆蓋（.gitignore）
│
├── camera/
│   ├── base.py              # CameraBase 抽象類別 + @register_camera 插件機制
│   │                        #   CAMERA_REGISTRY: dict[str, CameraInfo] 全域註冊表
│   ├── d435_stream.py       # 選項A：D435 USB RGB+Depth（@register_camera("d435")）
│   ├── cognex_stream.py     # 選項B：Cognex IS8505P In-Sight Native Mode
│   │                        #   TCP/Telnet (port 23) + FTP (port 21)
│   │                        #   （@register_camera("cognex", supported_depth_modes={"2D"})）
│   ├── usb_cam.py           # 選項C：通用 USB/RTSP（@register_camera("usb")）★新增
│   │                        #   Windows DSHOW 後端 / Linux CAP_ANY 自動切換
│   ├── file_source.py       # 本地影像來源（@register_camera("file")）★新增
│   │                        #   單張圖檔（循環）/ 目錄輪播，Unicode 路徑安全讀寫
│   ├── picker.py            # 影像來源選擇器 GUI ★新增
│   │                        #   scan_usb() + tkinter ListBox + terminal 降級
│   │                        #   pick_source() 公開 API，demo.py / 其他工具呼叫
│   └── factory.py           # 工廠函式：依 CAMERA_REGISTRY 建立相機實例
│
├── vision/
│   ├── yolo_worker.py       # GPU Thread：YOLO-OBB 推論 → OBBDetection + θ
│   └── geometry_worker.py   # CPU ProcessPool：OBB→抓取點 + Rz + 偏移補償 + Z(3D/2D)
│
├── coordinator/
│   ├── coordinator.py       # 任務指派、Queue 管理、結果合併
│   └── state_machine.py     # ★ 系統狀態機（INIT→READY→RUNNING→ERROR→SAFE_STOP）
│
├── communication/
│   └── opcua_bridge.py      # OPC-UA Thread：寫入 S7-1515 DB1（含重連機制）
│
├── ui/
│   └── ui_monitor.py        # UI Thread：即時顯示 + 健康儀表板 + 效能指標
│
├── utils/                   # ★ 共用工具模組
│   ├── logger.py            #   結構化 JSON 日誌（TimedRotating, cycle_id 追蹤）
│   ├── health_monitor.py    #   心跳監控：各模組 timeout 偵測
│   ├── metrics.py           #   效能指標收集（Context Manager 量測，P95，吞吐量）
│   └── retry.py             #   通用重試裝飾器（指數退避）+ Circuit Breaker
│
├── calibration/
│   ├── hand_eye_calib.py    # 手眼標定：產生 H.npy + 量測 OFFSET_X/Y
│   └── camera_height.py     # 相機安裝高度量測
│
├── tools/                   # ★ 開發輔助工具
│   ├── demo.py              #   YOLO-OBB 即時推論 demo（--source 彈性選擇）
│   ├── collect_images.py    #   採集訓練圖
│   ├── annotate.py          #   OBB 標註工具
│   ├── calibrate_camera.py  #   相機內參標定
│   ├── calibrate_hand_eye.py#   手眼標定（產生 H.npy + OFFSET_X/Y）
│   ├── adjust_exposure.py   #   曝光調整
│   ├── preview_labels.py    #   預覽標籤結果
│   ├── split_dataset.py     #   資料集分割 train/val
│   ├── train_yolo.py        #   YOLO-OBB 訓練啟動
│   └── verify_labels.py     #   標籤格式驗證
│
├── tests/                   # ★ 測試套件（172/172 通過）
│   ├── unit/
│   │   ├── test_geometry.py        # 幾何計算（22 tests）
│   │   ├── test_config.py          # 設定驗證（26 tests）
│   │   ├── test_state_machine.py   # 狀態轉換（28 tests）
│   │   ├── test_file_source.py     # FileCamera（19 tests）★新增
│   │   ├── test_usb_cam.py         # UsbCamera（21 tests）★新增
│   │   ├── test_camera_picker.py   # picker GUI/terminal（17 tests）★新增
│   │   └── test_exception_traps.py # 例外陷阱（15 tests）★新增
│   ├── integration/
│   │   ├── test_camera_mock.py     # Mock相機管線（8 tests）
│   │   └── test_opcua_sim.py       # OPC-UA Mock（9 tests）
│   └── e2e/
│       └── test_full_cycle.py      # 端對端（7 tests）
│
├── assets/
│   ├── H.npy                # Homography 矩陣（標定後產生，.gitignore）
│   └── best_obb.pt          # YOLO-OBB 權重（Git LFS 或 DVC）
│
├── logs/                    # ★ 日誌輸出目錄（.gitignore）
│
├── requirements.txt
├── pytest.ini
└── .gitignore
```

---

## requirements.txt

```
# ── 共用：核心功能 ──
opencv-python>=4.8
ultralytics>=8.1        # YOLO-OBB
torch>=2.1
torchvision>=0.16
opcua>=0.98             # OPC-UA client (python-opcua)
numpy>=1.24
psutil>=5.9             # CPU affinity / 系統監控

# ── 共用：維護基礎設施 ──
pyyaml>=6.0             # YAML 設定檔載入
pytest>=7.0             # 單元 / 整合測試
pytest-cov>=4.0         # 測試覆蓋率

# ── 選項 A：Intel RealSense D435 ──
pyrealsense2>=2.50      # D435 USB RGB + Depth

# ── 選項 B：Cognex IS8505P（In-Sight Native Mode）──
# ⚠ 使用 TCP/Telnet + FTP，依賴 Python 標準庫（socket, ftplib）
# 無需安裝第三方套件或 GigE Vision Driver
```

---

## config.py 關鍵參數

```python
# ============================================================
#  核心雙開關：CAMERA_TYPE × DEPTH_MODE
# ============================================================
CAMERA_TYPE = "d435"      # "d435" | "cognex"（及所有已 @register_camera 的類型）
DEPTH_MODE  = "3D"        # "3D"  | "2D"
#
# 合法組合：
#   d435  + 3D  → 即時深度量測（推薦：多厚度混合場景）
#   d435  + 2D  → 忽略深度，查表
#   cognex + 2D → TCP/Telnet 取像，查表
#   cognex + 3D → ❌ 啟動時拋出 ConfigError

# 網路
PLC_IP   = "192.168.0.1"         # S7-1515 PLC（依現場調整）
OPC_PORT = 4840
OPC_NS   = "urn:siemens:s71500"

# ---------- D435 參數（CAMERA_TYPE="d435" 時使用）----------
D435_WIDTH  = 1280
D435_HEIGHT = 720
D435_FPS    = 30

# ---------- Cognex 參數（CAMERA_TYPE="cognex" 時使用）----------
# ⚠ IS8505P 使用 In-Sight Native Mode（非標準 GigE Vision）
COGNEX_IP           = "192.168.0.10"   # Cognex IS8505P（實機偵測確認）
COGNEX_TELNET_PORT  = 23               # Native Mode 命令控制
COGNEX_FTP_PORT     = 21               # 影像下載
COGNEX_FTP_USER     = "admin"          # 預設帳號
COGNEX_FTP_PASSWORD = ""               # 預設空白

# ============================================================
#  機械參數（需依實際量測填入）
# ============================================================
# ---- DEPTH_MODE="3D" 時使用 ----
CAMERA_HEIGHT_MM  = 800.0   # 相機到工作台距離（3D 深度換算用）

# ---- DEPTH_MODE="2D" 時使用 ----
WORKTABLE_Z_MM    = 0.0     # 工作台面在機器人座標系的 Z 值
THICKNESS_MAP = {
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

# ============================================================
#  CPU 核心分配（i7-13620H：6P+4E = 16 Thread）
# ============================================================
CORE_UI          = [14]            # E-core
CORE_COORDINATOR = [12]            # E-core
CORE_OPCUA       = [13]            # E-core
CORE_GEOMETRY    = [6, 7, 8, 9]    # P-core × 4

# YOLO-OBB（RTX 4070 Laptop GPU 8GB VRAM）
YOLO_MODEL  = "assets/best_obb.pt"
YOLO_CONF   = 0.5
YOLO_DEVICE = "cuda:0"

# 放置位置對應（單位 mm，依實際場地填入）
PLACE_MAP = {
    "classA": {"x": 400.0, "y": 100.0, "z": -50.0},
    "classB": {"x": 400.0, "y": 200.0, "z": -50.0},
    "classC": {"x": 400.0, "y": 300.0, "z": -50.0},
}
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

## 待辦與現場確認清單

### 通用（Phase 1–5 程式碼完成，以下為現場部署項目）

- [x] 確認筆電 CPU 核心數 → i7-13620H 10C/16T，核心分配已設定
- [x] 建立 `configs/default.yaml` 及 `site_A.yaml` / `site_B.yaml`
- [x] 撰寫 `tests/unit/` 單元測試（幾何、設定驗證、狀態機）—— 100/100 通過
- [x] 設定 `.gitignore`
- [ ] 確認 S7-1515 IP 位址（預設 `192.168.0.1`，填入 site_X.yaml）
- [ ] 確認 KRC4 ↔ S7-1515 PROFINET 已接線設定
- [ ] 量測吸盤實際長度（SUCTION_LENGTH_MM），填入 site_X.yaml
- [ ] 準備 YOLO-OBB 訓練資料集（DOTA 格式四角點標註）
- [ ] 訓練 YOLO-OBB 模型（`yolo obb train`），產生 `best_obb.pt`
- [ ] 執行手眼標定，產生 `assets/H.npy`
- [ ] **量測鏡頭−吸盤偏移量**（OFFSET_X, OFFSET_Y），填入 site_X.yaml

### CAMERA_TYPE = "d435" 時（選項A）
- [ ] 確認 D435 USB 3.0 連線正常
- [ ] 安裝 `pyrealsense2`
- [ ] 量測相機實際安裝高度（CAMERA_HEIGHT_MM），填入 site_A.yaml
- [ ] 驗證深度精度是否在可接受範圍

### CAMERA_TYPE = "cognex" 時（選項B）
- [x] 確認 Cognex IS8505P IP → `192.168.0.10`（實機偵測確認）
- [x] 驗證 Telnet port 23 + FTP port 21 可正常連線（`User Logged In` 確認）
- [x] PC 有線介面（USB-LAN）設定靜態 IP `192.168.0.101`，ping 封鎖但 TCP 正常
- [ ] 確認 FTP 帳號密碼（預設 admin / 空白）
- [ ] 驗證 SE8 軟體觸發 → GV* 讀值 → FTP 取圖完整流程

### DEPTH_MODE = "2D" 時（D435 或 Cognex）
- [ ] 量測或登記每個品類的物件厚度（填入 THICKNESS_MAP）
- [ ] 確認 THICKNESS_MAP 涵蓋所有 PLACE_MAP 中的類別
- [ ] 量測工作台面在機器人座標的 Z 值（WORKTABLE_Z_MM）

---

## VSCode 開發建議

所有 Phase 均已完成，以下記錄開發順序供參考：

**Phase 1 — 基礎設施** ✅
1. `configs/default.yaml` → YAML 設定檔
2. `config.py` → 分層載入邏輯 + `validate_config()`
3. `utils/logger.py` → JSON 結構化日誌
4. `utils/retry.py` → 重試裝飾器 + Circuit Breaker
5. `utils/health_monitor.py` → 心跳監控
6. `utils/metrics.py` → Context Manager 形式效能計時
7. `coordinator/state_machine.py` → 狀態機
8. `tests/unit/test_config.py` → 設定驗證測試

**Phase 2 — 相機與視覺** ✅
9. `camera/base.py` → CameraBase 抽象介面 + `@register_camera` 插件機制
10. `camera/d435_stream.py` → D435 USB RGB+Depth
11. `camera/cognex_stream.py` → Cognex IS8505P In-Sight Native Mode（Telnet+FTP）
12. `camera/factory.py` → 依 CAMERA_REGISTRY 建立相機
13. `calibration/hand_eye_calib.py` → 產生 H.npy + OFFSET_X/Y

**Phase 3 — 推論與幾何** ✅
14. `vision/yolo_worker.py` → YOLO-OBB 推論，θ 角度輸出
15. `vision/geometry_worker.py` → XY + 偏移補償 + Rz + Z(3D/2D)
16. `tests/unit/test_geometry.py` → 幾何計算單元測試

**Phase 4 — 通訊與整合** ✅
17. `communication/opcua_bridge.py` → OPC-UA PLC 通訊（含重連）
18. `coordinator/coordinator.py` → 整合狀態機 + Queue + 結果合併
19. `tests/integration/test_opcua_sim.py` → OPC-UA Mock 整合測試

**Phase 5 — UI 與全系統** ✅
20. `ui/ui_monitor.py` → 即時顯示 + 健康儀表板 + 效能指標
21. `main.py` → 全系統整合（CLI：--site, --dry-run, --calibrate, --no-ui, --status）
22. `tests/e2e/test_full_cycle.py` → 端對端測試

**Phase 6 — 相機彈性擴充** ✅
23. `camera/file_source.py` → FileCamera 插件（單圖/目錄輪播，Unicode 路徑安全）
24. `camera/usb_cam.py` → UsbCamera 插件（Windows DSHOW / Linux CAP_ANY 自動切換）
25. `camera/picker.py` → pick_source() 影像來源選擇器（tkinter GUI + terminal 降級）
26. `tools/demo.py` 重構 → `--source` 統一參數，namedWindow 移至首幀後避免雙視窗
27. `tests/unit/test_file_source.py` → 19 tests（含 Unicode 路徑修正：imencode/fromfile）
28. `tests/unit/test_usb_cam.py` → 21 tests（含 Windows DSHOW 後端驗證）
29. `tests/unit/test_camera_picker.py` → 17 tests（scan_usb / GUI 降級 / terminal）
30. `tests/unit/test_exception_traps.py` → 15 tests（例外陷阱邊界條件）

> **目前狀態**：所有程式碼已完成，172/172 測試通過。下一步為現場部署：訓練資料採集、YOLO 訓練、手眼標定、PLC/KRC4 接線確認。
