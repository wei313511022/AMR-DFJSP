# Phase III 自有模型 I/O 契約（乾淨房版）

> 情境：**您的模型程式碼不能使用、不能 import 這個專案的任何程式碼**（程式碼屬於他人）。
> 您要在自己的 repo 獨立設計、獨立訓練；未來整合時，對方只需要
> **讀取您的權重檔 + 寫一個約 30 行的轉接層（adapter）**，就能把您的模型接進
> Phase III，取代 `ga_evolve()`。
>
> 本文件定義的就是那條邊界：您的模型該吃什麼（純資料）、該吐什麼（純資料）、
> 權重檔該長什麼樣。只要遵守這份契約，您的模型與此專案**零程式碼耦合**。

## 1. 設計總原則

```
┌─────────── 對方的專案（不能碰） ───────────┐   ┌───── 您的 repo（完全自主） ─────┐
│ GridEnv.step(action==1) 觸發                │   │                                  │
│   ga_jobs, init_state                       │   │                                  │
│        │                                    │   │                                  │
│        ▼                                    │   │                                  │
│  [Adapter 輸入端] ──── scene(純JSON) ─────────────▶ predict(scene) ──▶ plan(純JSON)│
│        │                                    │   │   ▲                              │
│  [Adapter 輸出端] ◀─── plan(純JSON) ◀─────────────┘                               │
│        │                                    │   │  模型由 checkpoint 檔完整還原，  │
│        ▼                                    │   │  不依賴任何訓練期程式碼           │
│  Individual(order, amr_assignment)          │   │                                  │
│  → local_improve → assign_schedules         │   └──────────────────────────────────┘
└─────────────────────────────────────────────┘
```

三條鐵律：

1. **邊界上只有純資料**：JSON-serializable 的 dict/list/數字/字串。您的模型不認識
   `GA.Job`、`Individual`、`init_state` 這些類別/變數名，只認識本文件定義的 schema。
2. **座標取代名稱**：scene 裡用 (x, y) 座標，不用 `"station3"`、`"dock1"`、`"AMR2"`
   這類專案內部命名——命名轉換是 adapter 的責任，不是模型的責任。
3. **權重檔自我描述**：checkpoint 內含重建模型所需的全部設定（架構超參數、
   正規化常數、schema 版本），讀檔即可推論，不需要您的訓練腳本在場。

## 2. 環境事實（設計模型時必須對齊的物理常數）

這些是從對方專案讀出的環境參數（`GA.py` 頂部常數區），您訓練資料的分佈
必須與它一致，否則權重接上去會失效：

| 項目 | 值 |
|---|---|
| 地圖 | 10×10 格點（x: 0–9, y: 0–9） |
| AMR | 5 台，起始/停靠點在 x=2 直行：(2,9) (2,7) (2,5) (2,3) (2,1) |
| 加工站 | 5 站，x=9 直行：(9,9) (9,7) (9,5) (9,3) (9,1) |
| 進料 dock | 5 個，x=0 直行：(0,9) (0,7) (0,5) (0,3) (0,1) |
| 物料種類 | A / B / C，對應加工時間 5 / 10 / 15 |
| AMR 載貨上限 | 每種物料合計 3 單位 |
| 移動時間 | ≈ 曼哈頓距離 × 1 tick/格（快速估計；實際模擬含 A* 繞路與碰撞） |
| 取料耗時 | 在 dock 取料本身耗時 = 該 job 的 duration |
| 加工耗時 | 在站點卸貨+加工耗時 = 該 job 的 duration |
| 資源互斥 | 每個 dock、每個站點同一時間只能服務一台 AMR |

> 建議：把這張表做成您 repo 裡的 `env_spec.json`，訓練環境從它讀參數。
> 未來若對方環境改版（例如 AMR 增為 8 台），只要換這個檔重訓，契約不變。

## 3. 模型輸入規格：`scene`

這是 Phase III 觸發當下，架構**所能提供的全部資訊**——請勿設計需要更多資訊的模型
（例如「每台 AMR 佇列中已排定但未執行的 job 清單」在現行介面拿不到）。

```jsonc
{
  "schema_version": "1.0",
  "time": 1234.0,                      // 目前模擬時鐘（tick）
  "amrs": [                            // 固定順序，索引 0..M-1 即模型的 AMR 編號
    {
      "position": [2, 9],              // 目前座標
      "available_at": 1250.0,          // 預計何時可接新工作（絕對時間，>= time）
      "inventory": {"A": 1, "B": 0, "C": 0}   // 車上各物料庫存
    }
    // ... 共 M 台（現行 M=5）
  ],
  "jobs": [                            // 變動長度 N，索引 0..N-1 即模型的 job 編號
    {
      "material": "A",                 // A/B/C
      "duration": 5.0,                 // 加工時間（=取料時間）
      "station_xy": [9, 5],            // 目的加工站座標
      "dock_xy": [0, 9],               // 進料 dock 座標
      "arrival_time": 1200.0           // 到達時間（<= time，已到達才會出現在此清單）
    }
    // ... 只含「尚未被任何 AMR 開始執行」的 job
  ]
}
```

語義注意事項：

- **已在執行中的 job 不會出現在 `jobs` 裡**；它們對系統的影響已隱含在
  `amrs[i].available_at`（那台 AMR 要晚一點才有空）與 `inventory` 中。
- `available_at` 是**絕對時間**；模型內部建議轉成相對值（`available_at - time`）
  再正規化，避免絕對時鐘數值無限增長破壞泛化。
- N 是變動的（實務上約 1–50）。**模型架構必須對 N 不敏感**
  （attention / GNN / 逐候選打分皆可；不能用攤平接全連接層的固定維度設計）。
- M（AMR 數）目前固定 5，建議也當作可變維度設計，成本很低、未來相容性高。

## 4. 模型輸出規格：`plan`

```jsonc
{
  "schema_version": "1.0",
  "assignment": [2, 0, 2, 4, 1],       // assignment[j] = 負責 job j 的 AMR 索引（0..M-1）
  "order": [                           // 全域執行順序：每個元素 = [job索引, 操作]
    [1, "pickup"],
    [0, "pickup"],
    [1, "unload"],
    [0, "unload"],
    [2, "pickup"]
    // ... 長度恰為 2N
  ]
}
```

**硬性約束**（adapter 端會驗證，違反即整包拒收）：

1. `len(assignment) == N`，每個值 ∈ `[0, M)`。
2. `order` 長度 = 2N；每個 job 恰好出現 `pickup` 一次、`unload` 一次。
3. 同一 job 的 `pickup` 必須排在 `unload` 之前。
4. （語義上）job 的 unload 由 pickup 它的同一台 AMR 執行——由 `assignment` 單值表示，
   天然滿足，模型不必額外處理。

> `order` 的意義：對方系統會把它依 `assignment` 分流到各 AMR 的執行佇列，
> **同一台 AMR 的 job 依 order 中的相對先後執行**。跨 AMR 的相對順序也會影響
> dock/站點資源的競爭結果，所以這是全域序，不是各 AMR 各自的序。

## 5. 推論介面：一個函式、確定性、無狀態

您的 repo 對外只需要暴露一個入口（打包成 pip package 或單一 `.py` 皆可）：

```python
def load_model(checkpoint_path: str) -> "Scheduler":
    """只憑 checkpoint 檔還原模型（見第6節），不依賴訓練程式碼。"""

class Scheduler:
    def predict(self, scene: dict) -> dict:
        """
        scene: 第3節格式。回傳: 第4節格式的 plan。
        - 必須確定性（同 scene 同輸出）：整合方需要可重現性
        - 無狀態：每次呼叫獨立，不得依賴前次呼叫的內部記憶
        - 單次呼叫延遲目標：< 1 秒（這是取代 GA 的主要賣點之一，
          GA 現行要跑 200個體×150代 + 3000 次局部搜尋）
        """
```

模型內部是一次性輸出（one-shot）還是自迴歸逐步建構（每步選一個 (job, AMR, 操作)），
**完全是您的自由**——契約只管 `predict` 的進出格式。自迴歸式較容易保證第 4 節的
約束（用遮罩逐步排除不合法選項），one-shot 式則需要後處理修復，請自行取捨。

## 6. 權重檔（checkpoint）規格：自我描述

「未來直接讀權重檔就能用」的關鍵是 checkpoint 必須攜帶重建模型的全部資訊：

```python
torch.save({
    "format_version": "1.0",
    "io_schema_version": "1.0",          # 對應第3、4節的 schema 版本
    "state_dict": model.state_dict(),
    "arch_config": {                     # 足以重建網路結構的全部超參數
        "model_class": "MySchedulerNet",  # 您 package 內的類別名
        "embed_dim": 128,
        "num_layers": 3,
        "num_heads": 8,
        "amr_feat_dim": 8,
        "job_feat_dim": 12,
        # ... 凡是 __init__ 需要的參數全部列出
    },
    "feature_config": {                  # 特徵工程的常數，推論端必須用同一組
        "normalize": {
            "position_scale": 10.0,
            "duration_scale": 15.0,
            "time_horizon_scale": 100.0
        },
        "materials": ["A", "B", "C"],    # one-hot 順序
        "relative_time": true            # available_at 是否轉相對時間
    },
    "env_spec": { ... },                 # 第2節的環境常數快照（訓練時所假設的環境）
    "metrics": {"val_makespan_mean": ..., "baseline_ga_ratio": ...}   # 選填，可追溯性
}, "my_scheduler_v1.pth")
```

`load_model()` 的實作：讀 `arch_config` → 建網路 → load `state_dict` →
用 `feature_config` 建特徵前處理器。**任何常數都不准寫死在推論程式裡**，
一律從 checkpoint 讀——這樣換一個權重檔就是換一個模型，包括未來環境改版重訓的版本。

## 7. 整合時的 adapter（寫在對方專案側，供未來參考）

這段程式碼**屬於整合階段、寫在對方專案裡**（或由對方同意的膠水檔案），
不屬於您的模型 repo。列出來是為了證明契約可落地、以及讓您知道欄位如何對應：

```python
# === 輸入端：對方的 (ga_jobs, init_state) → 您的 scene ===
AMR_IDS = ["AMR1", "AMR2", "AMR3", "AMR4", "AMR5"]   # 索引順序即契約中的 AMR 編號

scene = {
    "schema_version": "1.0",
    "time": init_state["time"],
    "amrs": [{
        "position": list(init_state["positions"][a]),
        "available_at": init_state["availability"][a],
        "inventory": init_state["inventory"][a],
    } for a in AMR_IDS],
    "jobs": [{
        "material": j.type_,
        "duration": j.duration,
        "station_xy": list(STATIONS[j.station]),          # 名稱→座標在這裡轉
        "dock_xy": list(INBOUND_DOCK_LOCATIONS[dock_key_from_value(j.inbound_dock)]),
        "arrival_time": j.arrival_time,
    } for j in ga_jobs],
}

# === 推論 ===
plan = scheduler.predict(scene)          # scheduler = load_model("my_scheduler_v1.pth")

# === 輸出端：您的 plan → 對方的 Individual ===
best_ind = Individual(
    order=[Operation(job_idx, kind) for job_idx, kind in plan["order"]],
    amr_assignment=[AMR_IDS[k] for k in plan["assignment"]],
)
# 之後照舊：local_improve(...) → assign_schedules(...)，一行都不用改
```

對應關係一覽：

| 對方專案 | 契約欄位 | 方向 |
|---|---|---|
| `init_state["time"]` | `scene.time` | → 模型 |
| `init_state["positions"][amr]` | `scene.amrs[i].position` | → 模型 |
| `init_state["availability"][amr]` | `scene.amrs[i].available_at` | → 模型 |
| `init_state["inventory"][amr]` | `scene.amrs[i].inventory` | → 模型 |
| `ga_jobs[j].type_ / duration / arrival_time` | `scene.jobs[j].material / duration / arrival_time` | → 模型 |
| `STATIONS[ga_jobs[j].station]`（名→座標） | `scene.jobs[j].station_xy` | → 模型 |
| `plan.assignment[j]`（索引→`"AMR{k}"`） | `Individual.amr_assignment[j]` | 模型 → |
| `plan.order`（`[j,kind]`→`Operation`） | `Individual.order` | 模型 → |

## 8. 獨立訓練時的注意事項（與契約直接相關者）

1. **自建模擬器**：您需要在自己 repo 依第 2 節環境事實實作訓練用模擬器
   （曼哈頓移動、dock/站點互斥、取料/加工耗時、庫存上限）。這是乾淨房重寫，
   不是抄程式碼——第 2 節的表就是完整的行為規格。
2. **訓練資料分佈**：`scene` 要涵蓋「冷啟動」（全部 AMR 在基地、庫存 0、
   `available_at == time`）與「中途狀態」（AMR 散佈各處、`available_at` 參差、
   有庫存）兩類，因為部署時模型看到的絕大多數是後者。
3. **優化目標對齊**：對方系統的評估目標是
   `makespan + 0.001 × Σ(各AMR完工時間)`（主要 makespan、次要負載平衡）。
   您的 reward / loss 對齊這個目標，接上去的表現才會與 GA 可比。
4. **輸出永遠合法**：第 4 節的約束建議在模型端就結構性保證（自迴歸+遮罩），
   而不是依賴 adapter 拒收重試——整合方不會幫您重試。
5. **版本欄位要認真維護**：`io_schema_version` 變了就是 breaking change，
   adapter 端會據此拒絕不相容的權重檔。

## 9. 交付驗收清單

未來交付給整合方的東西：

- [ ] `my_scheduler_v1.pth`（第 6 節格式，自我描述）
- [ ] 推論 package / 單檔（只含 `load_model` + `Scheduler.predict`，依賴僅 torch/numpy）
- [ ] 本契約文件（第 3、4 節 schema + 範例 scene/plan JSON 各一份）
- [ ] 驗收腳本：讀一個範例 `scene.json` → `predict` → 檢查 plan 通過第 4 節全部約束
- [ ] 效能報告：在驗證集上 vs 對方 GA 的 makespan 比值、單次推論延遲
