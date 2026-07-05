# AMR-DFJSP — DDQN 排程模型(Phase III 契約對齊版)

以 Rainbow/DDQN 訓練 AMR 派工策略,並依照 `docs/Phase3_Model_IO_Contract.md`
的 I/O 契約輸出**自描述 checkpoint**,讓訓練完的模型可以直接接進對方架構
(取代 `ga_evolve()`),整合方只需 `load_model()` + `predict(scene)`。

### 代辦事項
[*] 訓練資料似乎有誤，不是用Generate_training_data.py
[] 相關目標函數確定 
[] 修改main 將訓練與測試分開
[] 超參數調整
[] object functionn 設計
[] 了解rainbow ddqn設計
[] 目前 action space 已改為 dock-per-job??

## 資料夾結構

```
├── main.py                  # 訓練/測試入口(python main.py)
├── configs/
│   └── env_spec.json        # 場域常數(Route_Map:12×12、5 AMR、6 站、MA/MB/MC、T)
├── core/                    # 環境、模型、特徵(推論期也依賴,僅 torch/numpy)
│   ├── env.py               #   模擬器:dock/站點互斥、避碰、init_state 暖啟動、批次取料
│   ├── model.py             #   QNetwork(classic / Rainbow)
│   ├── features.py          #   動作空間與特徵 (travel, station_wait, proc, replenish_add)
│   └── data_io.py
├── training/                # 只在訓練期使用
│   ├── trainer.py, replay.py, rollout.py, evaluator.py
├── inference/               # ★ 交付給整合方的推論套件(契約 §5、§6)
│   ├── scheduler.py         #   load_model(ckpt) -> Scheduler.predict(scene) -> plan
│   └── checkpoint_io.py     #   export_contract_checkpoint(自描述權重檔)
├── viz/                     # 視覺化(matplotlib / plotly / route map)
├── scripts/
│   ├── validate_contract.py # 契約 §9 驗收腳本
│   ├── random_job_gen.py    # 訓練資料產生器(含 dock 欄位)
│   ├── live_job_feeder.py
│   └── Generate_training_data.py   # 經典 FJSSP 資料集(見文末)
├── data/                    # jsonl 資料(訓練/測試情境)
├── checkpoints/             # ddqn_policy.pt(續訓用)/ my_scheduler_v1.pth(交付用)
├── docs/                    # 契約、參數說明、範例 scene/plan JSON
├── notebooks/               # 舊 notebook(使用搬移前的扁平 import,僅供參考)
└── results/                 # 訓練曲線、圖表、影片
```

## 工作流程

```bash
# 1. 訓練(自動產生訓練資料;結束時同時輸出兩種 checkpoint)
python main.py

# 2. 契約驗收:load_model -> predict -> 檢查 §4 約束 / 確定性 / <1s 延遲
python scripts/validate_contract.py                  # 用 checkpoints/my_scheduler_v1.pth
python scripts/validate_contract.py --init-random    # 未訓練權重的管線煙霧測試
```

整合方拿到 `my_scheduler_v1.pth` 後:

```python
from inference import load_model
scheduler = load_model("my_scheduler_v1.pth")
plan = scheduler.predict(scene)   # scene: 契約 §3;plan: 契約 §4
```

`predict()` 為確定性、無狀態、單次 < 1 秒;plan 回傳前會先驗證 §4 全部硬約束。
範例 scene/plan 見 `docs/examples/`(由驗收腳本產生)。

## 與契約的對應

| 契約項目 | 本 repo 實作 |
|---|---|
| §2 環境事實 | `configs/env_spec.json`(env 由此讀參數,改版換檔即可) |
| §3 scene 輸入 | `inference/scheduler.py: Scheduler._scene_to_episode`(絕對時間→相對時間) |
| §4 plan 輸出 | 自迴歸 rollout 逐步派工,結構性滿足約束;`validate_plan` 再驗一次 |
| §5 推論介面 | `inference.load_model()` / `Scheduler.predict()` |
| §6 自描述權重檔 | `inference/checkpoint_io.py: export_contract_checkpoint` |
| §8 訓練注意事項 | `env.reset(scenario, init_state=...)` 支援中途狀態訓練;dock 取料耗時=duration、dock 互斥已入模擬器 |

## 重要語義(批次取料 batch pickup)

- **取料工序(op0 / 單工序 job)**:動作 = 選 task + 選「這趟在料倉取幾份該材料」:
  - 車上庫存 0 → 必取 1~3 份(容量上限每種 3);
  - 車上庫存 >0 → 可 `add=0` 直接用存貨送站(**不進料倉**),或順路補到滿(proactive)。
  - 取 N 份耗時 = N × 材料 duration(A/B/C = 5/10/15),期間佔用料倉(互斥)。
- **搬運工序(FJSSP op>0)**:到前一站取在製品(耗時 0),單一動作、不批次。
- **動作評分** `Score(i) = Q(i) + cover_bonus + load_bonus + wait_bonus`
  (features.py `select_action_index`;權重隨 checkpoint 的 `selection_bias` 交付,
  推論端 `predict()` 使用與訓練相同的權重)。
- **契約 plan 的批次對應**:每份材料入庫時記下料倉造訪的子區間(N 份 = N 段
  duration),job 消耗哪份(FIFO)其 `pickup` 就歸屬那段 → 同車連續 pickup 在
  `order` 中相鄰,整合方重演時自然形成批次取料;§4 約束仍結構性成立。
- **推論端保守規則**:scene 給的初始庫存不拿來抵扣(`consume_initial_inventory=False`),
  因為整合方會重演每個 job 自己的 pickup;訓練端則可完整使用庫存。
  另外若 scene 中同一材料來自多個 dock,批次取料無法保真,`predict()` 會自動
  降級為「每次進倉只取一份」(材料↔料倉一對一的場域不受影響)。
- **目標函數**:reward 對齊契約評估目標 `makespan + 0.001×Σ(各AMR完工時間)`
  (權重見 `env.objective_load_balance_weight`)。
- dock 等待與取料以「路徑停留步」寫進 transport path,避碰預約自然涵蓋 dock 佔用。

> **效能報告注意**:訓練與推論用的是快速估計(曼哈頓距離、無避碰),
> 對方系統會用 A*+避碰重演 plan——因此 makespan 數字**必須以對方模擬器重演
> 的結果為準**,本 repo 內部數字只能當相對比較用。

---

### 附:經典 FJSSP 資料集(`scripts/Generate_training_data.py`)

6 stations;10/15/20/25/30 jobs per instance;4~8 operations per job;
1~6 feasible machines per operation;processing time 10~99。
ref: https://github.com/SchedulingLab/fjsp-instances/tree/main


