# AMR-DFJSP — Rainbow DDQN 排程模型（FJSSP + AMR 搬運 + 出貨）

以 Rainbow/DDQN 求解**含 AMR 物料搬運的彈性零工式排程問題（FJSSP-AMR）**：
模型在每個決策點替空閒 AMR 決定「派哪個 job 的哪道工序、選哪台可行機台、
順路批次取幾份材料」，機台獨立加工，job 最後還要由 AMR 送到出貨口 T 才算完成，
目標是最小化 makespan。訓練資料由 `scripts/Generate_training_data.py`
（**不可修改**）產生；另依 `docs/Phase3_Model_IO_Contract.md` 的 I/O 契約
輸出自描述 checkpoint，供整合方 `load_model()` + `predict(scene)` 直接使用。

> **現況注意**：排程語義已改為「機台獨立加工＋送 T」（2026-07），
> `checkpoints/` 內既有權重是舊語義訓練的，**需要重新訓練**（`python main.py`）
> 才能得到有效的效能數字。

### 代辦事項
[*] 訓練資料似乎有誤，不是用Generate_training_data.py
[] 相關目標函數確定
[] 超參數調整
[] object function 設計
[] 了解rainbow ddqn設計
[] 目前 action space 已改為 dock-per-job??
[*] 出貨點job（最後工序完工後送 T，見 §1／§4.3）
[] 以新語義重新訓練模型（舊 checkpoint 已不適用）

---

## 1. 問題定義

- **場域**：12×12 網格（含障礙物），5 台 AMR、6 個加工站（S1–S6）、
  3 個進料料倉（MA/MB/MC，材料 A/B/C 一對一）、1 個出貨口 T (11,5)。
  佈局定義於 `configs/env_spec.json`，對應 `results/Route_Map.png`。
- **工件（job）**：每個 job 隨機帶一種材料（A/B/C，取料時間 5/10/15）。
  - **經典 FJSSP 多工序 job**（主要格式）：每道工序（operation）有多台可行機台
    與各自加工時間，**工序順序固定**——工序 k 完成才釋放工序 k+1
    （`scripts/Generate_training_data.py` 格式）。
  - **單工序派工 job**（legacy）：指定站點與加工時間（`scripts/random_job_gen.py`
    格式），可含動態到達（dispatch_time）。
- **決策**：事件驅動——每當「有 AMR 空閒且有可派工序」即為一個決策點，
  模型替該 AMR 選擇 (工序＋機台) 以及「這趟在料倉批次取幾份材料」。
- **FJSSP 語義（機台獨立加工）**：AMR 只負責搬運——把工件送到選定機台、
  等機台空出後**交件（handover）即釋放**，可立刻接下一個任務；機台自行加工，
  加工完成才釋放該 job 的下一道工序。**機台選擇由模型決定**：工序有多台可行
  機台時，每台機台是一個候選動作，模型依 Q 值（以整體目標函數為 reward 訓練）
  挑選對 makespan 最有利的機台。
- **出貨（delivery to T）**：job 的最後一道工序完工後，會釋放一個「送出貨口」
  任務——AMR 到最後加工站取成品（耗時 0）、運往出貨口 **T**；
  **job 到達 T 才算完成**。**Makespan ＝ 最後一個 job 送達 T 的時間**
  （env 旗標 `deliver_finished_to_output`，契約推論路徑會關閉）。
- **限制**：工序先後順序不可變動（op k 完成才能派 op k+1）、料倉/站點互斥
  （同一時間只服務一台 AMR）、AMR 每種材料載貨上限 3、
  可選的時間感知避碰（time-aware A*）。
- **目標函數**：`makespan + 0.001 × Σ(各AMR完工時間)`（主要 makespan、
  次要負載平衡；與契約評估目標一致，權重見 `env.objective_load_balance_weight`）。

## 2. 系統架構與資料流

```
 scripts/Generate_training_data.py（不可改）      data/test_data/*.jsonl
 （FJSSP 訓練資料，每行一個 instance）             （測試資料，一行一筆測資）
                  │                                        │
                  ▼                                        ▼
 main.py ──► training/trainer.py: train_ddqn()   training/test_runner.py
      │       每個決策點：                          evaluate_test_folder()
      │        build_actions_for_tasks()  ← 動作空間（工序×機台×取料量）
      │        env.action_features()      ← 動作特徵 (travel, wait, proc, add)
      │        QNetwork（Rainbow）        ← Q 值        │
      │        select_action_index()      ← Score=Q+bonus│ 每行：計算時間+makespan
      │        env.step()                 ← 交件即釋放、 │ → summary.csv
      │                                     機台加工、送T│ → 三面板影片(.mp4/.gif)
      ▼                                                  ▼
 checkpoints/ddqn_policy.pt（續訓）              results/test_runs/
 checkpoints/my_scheduler_v1.pth（契約交付）
      │
      ▼
 inference/load_model() ──► Scheduler.predict(scene) ──► plan
 （契約 §5：確定性、無狀態、<1s；scripts/validate_contract.py 驗收）
```

## 3. 資料夾結構

```
├── main.py                  # 訓練/測試入口（python main.py），所有超參數集中在此
├── configs/
│   ├── env_spec.json                # 現行場域常數（12×12、5 AMR、6 站、MA/MB/MC、T）
│   └── env_spec_phase3_contract.json# 契約 §2 原始 10×10 場域（保留參考）
├── core/                    # 環境、模型、特徵（推論期也依賴，僅 torch/numpy）
│   ├── env.py               #   事件驅動模擬器（本專案核心，約 1300 行）
│   ├── model.py             #   QNetwork（classic MLP / Rainbow 雙模式）
│   ├── features.py          #   動作空間建構、批次 Q 值、動作評分
│   └── data_io.py           #   JSONL/JSON 讀檔與 live 檔案輪詢
├── training/                # 只在訓練期使用
│   ├── trainer.py           #   train_ddqn 主迴圈 + 資料準備 + 訓練監控圖
│   ├── replay.py            #   PrioritizedReplayBuffer、NStepAccumulator
│   ├── rollout.py           #   greedy 測試回合（靜態/即時動畫/live stream）
│   ├── evaluator.py         #   demo 測試+繪圖入口、批次評估
│   └── test_runner.py       #   ★ 測試資料夾評估（逐筆計時+makespan）+ 影片輸出
├── inference/               # ★ 交付給整合方的推論套件（契約 §5、§6）
│   ├── scheduler.py         #   load_model(ckpt) -> Scheduler.predict(scene) -> plan
│   └── checkpoint_io.py     #   export_contract_checkpoint（自描述權重檔）
├── viz/                     # 視覺化（機台/AMR 甘特、路線圖、Plotly 互動）
├── scripts/
│   ├── Generate_training_data.py   # FJSSP 訓練資料產生器（不可修改；data/ 內有原始對照檔）
│   ├── random_job_gen.py    #   動態派工批次產生器（legacy）
│   ├── live_job_feeder.py   #   live 模式的 job 注入器
│   └── validate_contract.py #   契約 §9 驗收腳本
├── data/
│   ├── test_data/           #   ★ 測試資料夾（test_dataset.jsonl：一行一筆測資）
│   ├── data_README.md       #   FJSSP 資料集格式說明（fjsp-instances）
│   ├── sample_abz5.json     #   原始格式範例（job 直接是 operation list）
│   ├── Generate_training_data.py  # 產生器原始對照檔（訓練 import 的是 scripts/ 那份）
│   └── fjssp_training_dataset.jsonl # 訓練資料（main.py 自動產生/讀取）
├── checkpoints/             # ddqn_policy.pt（續訓）/ my_scheduler_v1.pth（交付）
├── docs/                    # 契約、參數說明（PARAMETER_GUIDE.md）、範例 scene/plan
├── notebooks/               # 舊 notebook（使用搬移前的扁平 import，僅供參考）
└── results/
    └── test_runs/           #   ★ 測試輸出（summary.csv + 每筆測資影片）
```

## 4. 各模組詳細說明

### 4.1 `main.py` — 訓練/測試入口

單一 `main()` 函式，前半段是所有可調參數（資料來源、DDQN/Rainbow 超參數、
測試/影片/視覺化開關），後半段依序：

1. **載入訓練資料**：`use_fjssp_dataset=True`（預設）時呼叫
   `scripts/Generate_training_data.create_jsonl_dataset()` 產生/讀取
   `data/fjssp_training_dataset.jsonl`，每行一個 instance 即一個訓練情境；
   `False` 時走 `training/trainer.prepare_scenarios()` 的動態派工批次路線。
2. **建環境**：`TaskSchedulingEnv()`，套用 proactive replenish 權重與避碰開關
   （訓練預設關避碰——time-aware A* 佔 episode 時間 1000 倍以上而 GPU 閒置，
   且契約的移動模型本來就是快速估計）。
3. **建網路**：`state_dim` 由 `env.reset([])` 的回傳長度自動推得（現行場域為 46），
   `input_dim = state_dim + action_dim(4)`；policy/target 兩個 `QNetwork`。
4. **訓練**（`do_train`）：呼叫 `train_ddqn()`，結束後輸出兩種 checkpoint：
   - `checkpoints/ddqn_policy.pt`：policy/target/optimizer 完整狀態，續訓用。
   - `checkpoints/my_scheduler_v1.pth`：契約 §6 自描述格式，交付用
     （由 `export_contract_checkpoint()` 產生，`selection_bias` 權重一併封裝）。
5. **測試**（`do_test`）：先對 `test_data_dir`（預設 `data/test_data/`）做
   **逐筆測資評估**——印出每筆的模型計算時間與 makespan、寫 `summary.csv`、
   輸出三面板影片（見 4.10）；再跑 `run_test_and_plot()`（demo 回合＋各種圖），
   最後 `print_batch_results()` 對前 N 個訓練情境快速批次評估。

### 4.2 `configs/env_spec.json` — 場域規格（契約 §2）

環境的一切物理常數都從這個檔讀：網格大小與障礙物、AMR 數量/起始位置/載貨上限、
站點座標（S1–S6）、料倉座標（MA/MB/MC）、材料→料倉對應（`material_dock_map`）、
出貨口 T、材料取料時間（A/B/C = 5/10/15）、資源互斥旗標。
**場域改版只要換檔重訓，程式不用改**；checkpoint 也會夾帶當時的 env_spec 快照。
`env_spec_phase3_contract.json` 是契約文件原始的 10×10 無障礙版本，保留對照用。

### 4.3 `core/env.py` — 事件驅動模擬器（核心）

`TaskSchedulingEnv` 同時服務訓練與推論，只依賴 numpy。重點機制：

- **任務模型**：`_jobs_to_tasks()` 把 job 轉成 available_tasks。
  - 多工序 job（含 `operations`）：`_make_op_tasks()` 為每道工序的**每台可行機台**
    建一個 task（FJSSP 彈性＝選 task 即選機台），同工序的兄弟 task 共用 `op_uid`，
    派出其中一個即移除其餘；工序完成時間到才釋放下一工序（`_pending_ops`）。
  - 工序 0 到**該 job 材料對應的料倉**取料（服務時間＝材料 duration）；
    工序 >0 到前一站取在製品（服務時間 0，不佔料倉互斥）。
  - 最後一道工序完工時，`_make_delivery_task()` 釋放「送出貨口」任務
    （取成品→運往 T，無機台選擇、無加工），到達 T 即完成該 job 並計入 makespan。
  - 也接受 `data/data_README.md` 的原始格式（job 直接是 operation list、無材料欄位，
    如 `data/sample_abz5.json`）：材料以 jid 輪替 A/B/C 補上，讓工序 0 有料倉可取。
- **決策點推進**：`_advance_to_decision_point()` 依「最早空閒 AMR ＋ 有可派 task」
  的事件驅動規則快轉模擬時鐘；`_release_until()` 依 dispatch_time 釋放動態到達的 job。
- **動作執行**：`step((action_index, replenish_plan))`：
  1. `_estimate_action_plan()` 估計時間軸：
     `transport（含料倉等待/取料的停留步）→ station wait → process`。
     料倉等待與取料以「路徑停留步」寫進 transport_path，
     使避碰預約自然涵蓋料倉佔用。
  2. 庫存記帳：批次取 N 份會建立 N 個 FIFO 單位事件，各自記住料倉造訪的
     子區間（第 k 份 = 第 k 段 duration）；job 消耗哪份（FIFO），其契約 `pickup`
     就歸屬那段——同車連續 pickup 在 plan `order` 中相鄰，整合方重演時
     自然形成批次取料。
  3. **交件即釋放**：AMR 等機台空出、於 `process_start` 交件後即釋放
     （`robot_free_times = 交件時間`），機台獨立加工至 `process_end` 才佔用站點
     （`station_busy_until`）並釋放下一工序；makespan 以工序/送貨完成時間追蹤
     （`_max_completion`）。更新後寫入 `trace`（每筆含 op_index、num_ops、
     is_delivery、segments、transport_path、pickup 歸屬區間等）。
  4. **Dense reward**：`-(Δmakespan) - w×(Δ Σ交件時間)`，逐步累加後
     episode 總 reward ＝ 負的最終目標函數值。
- **避碰（可開關）**：`_build_dynamic_reservations()` 把其他 AMR 已承諾的
  路徑/等待/閒置區間與機台加工佔用轉成 (格點, 時間) 點/區間/邊預約；
  `_plan_path_time_aware()` 用時間擴展 A*（允許原地等待）找無衝突路徑；
  `_post_process_position()` 讓交件後的 AMR 讓出站點格。
  關閉時走 BFS 最短路徑快速估計（訓練與推論預設）。
- **狀態向量** `_get_state()`（`M + 2 + 6M + S + D` 維，現行場域 M=5/S=6/D=3 → 46）：
  current_robot one-hot、可派 task 數、時間 t、
  每台 AMR 的 [free_time, x, y, invA, invB, invC]、
  各站/各料倉的 busy_until（時間皆為 episode 相對值）。
- **暖啟動**：`reset(scenario, init_state=...)` 支援契約「中途狀態」scene
  （AMR 位置/可用時間/車上庫存），供訓練中途狀態分佈與推論還原現場。
- **動態注入**：`enqueue_jobs()` 供 live stream 模式在模擬中途加入新 job。

### 4.4 `core/features.py` — 動作空間與動作評分

- `build_actions_for_tasks()`：組出 (task_idx, replenish_plan) 動作列表。
  - 搬運/送貨工序（op>0）：單一動作、不批次。
  - 料倉工序（op0）：庫存 0 → 必取 1..cap_i；庫存 >0 → 可 add=0 直接用車上庫存
    送站（不進料倉），或（開啟 proactive 時）順路補貨 1..cap_i。
    cap_i 由「目前可見的同材料需求」封頂，避免 episode 尾端載死庫存；
    `max_add=1` 供推論端在批次不保真時降級。
- `q_values_batch()`：一個 state 對 K 個動作特徵批次算 Q。
- `select_action_index()`：**Score(i) = Q(i) + cover_bonus + load_bonus + wait_bonus**。
  cover＝補貨能多覆蓋幾個未來同材料 job、load＝載貨率、wait＝站點反正要等就
  順便補貨。bonus 在**同一 task 的數量選項間**零基化，不影響跨 task 排序
  （跨 task 純由 Q 決定）；平手時偏好較大補貨量。權重隨 checkpoint 的
  `selection_bias` 交付，推論端與訓練端使用同一組。

### 4.5 `core/model.py` — Q 網路

`QNetwork` 包裝兩種骨幹，`load_state_dict()` 會自動從權重推斷是哪種（舊檔相容）：

- **classic**：state+action 串接 → 深層 MLP → 純量 Q。
- **rainbow**（預設）：Dueling 架構——state encoder (46→256→128) 與
  action encoder (4→64→64) 分流；Value 流與 Advantage 流皆為 `NoisyLinear`
  （factorized Gaussian，取代 ε-greedy 探索）；輸出 C51 分佈
  （51 個 atom，支撐 [-10000, 0]，涵蓋 n-step 折扣目標範圍），
  Q ＝ 分佈期望值。`action_values()` 對單一 state 高效評分整批候選動作
  （advantage 在候選集合內做均值中心化）。

### 4.6 `core/data_io.py` — 資料讀寫

`load_records()` 相容 JSONL / JSON array / 單一 JSON object；
`record_to_jobs()` 拆 dispatch_time 與 jobs；`poll_live_job_file()`
增量讀取 live 檔案的新行（供 live stream 模式）。

### 4.7 `training/trainer.py` — 訓練主迴圈

- `prepare_scenarios()`：legacy 資料路線——呼叫 `random_job_gen` 產生
  多條派工串流（`dispatch_batches_{i}.jsonl`），每條串流是一個訓練情境。
- `train_ddqn()`：每個 episode 隨機抽一個情境（可 full/window/subset 取樣），
  逐決策點：建動作空間 → 算動作特徵 → NoisyNet（或 ε-greedy）選動作 →
  `env.step()` → 存入 `NStepAccumulator`（n=3）→ 到齊後進 PER。
  每 `train_every_steps` 步從 PER 取批次更新：
  - Rainbow 路徑：C51 分佈投影（Double DQN——policy net 選 a*、target net 給分佈），
    cross-entropy loss × 重要性權重，loss 值回寫 priority。
  - classic 路徑：Double DQN 的 MSE TD loss。
  - 梯度裁剪、每 `target_sync_steps` 步（或每 5 episodes）同步 target net。
  - 內建即時訓練監控四宮格（makespan / loss / epsilon / 正規化 makespan，
    mk/proc 以「各工序最快機台時間總和」為分母）與可選的即時 Gantt/路線圖
    （長訓建議關閉），並輸出各階段耗時 profiling。
- `training/replay.py`：`PrioritizedReplayBuffer`（proportional PER，α/β 退火、
  重要性抽樣權重）與 `NStepAccumulator`（n-step 報酬摺疊，done 時 flush）。

### 4.8 `training/rollout.py` — 測試回合

- `run_greedy_episode()`：eval 模式（噪音關閉）跑一整個 episode，回傳 makespan。
- `run_greedy_episode_live()`：同上但逐步繪製動畫（派工佇列/Gantt/輸入佇列），
  可錄影格 PNG、輸出 GIF。
- `run_greedy_episode_live_stream()`：線上模式——輪詢 `data/live_jobs.jsonl`，
  新 job 用 `env.enqueue_jobs()` 動態注入，與 `scripts/live_job_feeder.py` 搭配
  可展示即時派工。

### 4.9 `training/evaluator.py` — demo 測試與批次評估

- `run_test_and_plot()`：讀 `data/test_scenario_one_time.jsonl` 跑 demo 回合，
  依開關輸出：matplotlib 互動排程、路線圖回放、Plotly 互動 Gantt、
  以及 `results/` 下的靜態 PNG（dispatch_queue / machine_schedule /
  amr_schedule / input_queue）。
- `print_batch_results()`：對情境列表逐一跑 greedy 回合印出 makespan；
  預設關避碰且只取前 10 個（全資料集開避碰評估要數小時）。

### 4.10 `training/test_runner.py` — 測試資料夾評估＋影片輸出 ★

把 `data/test_data/` 內的測試資料整批評估，**逐筆記錄模型計算時間與 makespan**，
並可為每筆輸出排程過程影片：

- `discover_test_scenarios(dir)`：掃描資料夾內全部 `.jsonl/.json`，
  依內容自動判斷——記錄含 `dispatch_time` → 整個檔案是一條動態派工串流情境；
  否則**一行（一筆記錄）＝一個獨立測資**（FJSSP instance，命名
  `test_dataset_000`、`test_dataset_001`…）。
- `evaluate_test_folder(...)`：對每筆測資跑 greedy 回合，記錄
  `jobs / makespan / finish_sum / objective / eval_seconds`
  （eval_seconds＝模型排程該筆的完整計算時間，含所有派工、機台選擇、送 T 決策），
  寫入 `results/test_runs/summary.csv` 並印出平均/最小/最大 makespan。
  預設 `collision_avoidance=False`（快速估計＝訓練/推論的移動模型；
  要避碰忠實影片再開，每筆需數分鐘）。
- `record_schedule_video(...)`：把跑完的 `env.trace` 渲染成**單一三面板影片**——
  上：機台甘特圖（整體 FJSSP 排程，長條標籤 `J{jid}(目前工序/總工序)`，
  紅色游標隨時間移動）；中：AMR Gantt（搬運/等待活動）；
  下：場域路線圖（AMR 即時位置、路徑、各機台加工中工序與剩餘時間、出貨口 T），
  底部附機台/AMR 狀態文字。影格數由 `video_max_frames` 封頂（時間軸等距抽樣）。
  **影片格式**：系統 PATH 有 ffmpeg → `.mp4`；否則用 `imageio-ffmpeg` 套件
  附帶的 ffmpeg（已裝，`pip install imageio-ffmpeg`）→ `.mp4`；
  兩者都沒有 → 自動退回 `.gif`（Pillow）。

`main.py` 對應參數：`test_data_dir`（預設 `data/test_data`，不存在則跳過）、
`test_output_dir`、`test_folder_collision_avoidance`（預設 False）、
`test_folder_max_scenarios`、`save_test_videos`、`test_video_max_scenarios`
（只為前 N 筆錄影，None＝全部）、`test_video_fps` / `test_video_max_frames` /
`test_video_dpi`。

### 4.11 `inference/` — 交付推論套件（契約 §5、§6）

- `scheduler.py`：
  - `load_model(path)`：**只憑 checkpoint** 重建模型——讀 `arch_config` 建
    `QNetwork`、載入 `state_dict`、用 `env_spec` 快照建內部模擬器、
    用 `feature_config.selection_bias` 還原動作評分權重。
  - `Scheduler.predict(scene)`：契約 §3 scene → §4 plan。
    `_scene_to_episode()` 把絕對時間轉相對、AMR 現況轉 `init_state`；
    然後在內部模擬器上自迴歸 rollout（與訓練同一套動作空間/評分），
    從 `env.trace` 讀出 `assignment`（job→AMR）與 `order`
    （全域 [job, pickup/unload] 序，依模擬事件時間排序），
    回傳前用 `validate_plan()` 再驗一次 §4 全部硬約束。
    保證**確定性**（eval、噪音關閉、純 argmax）、**無狀態**、單次 < 1 秒。
  - 契約模式差異：關避碰（整合方會用自己的 A*/避碰引擎重演 plan）、
    **關送 T**（契約 plan 每個 job 只有 pickup/unload）、初始庫存不抵扣
    （`consume_initial_inventory=False`）；同一材料來自多個 dock 時批次取料
    無法保真，自動降級為每趟只取一份（`max_add=1`）。
- `checkpoint_io.py`：`export_contract_checkpoint()` 輸出自描述 .pth——
  `format_version / io_schema_version / state_dict / arch_config /
  feature_config（含 state 佈局說明與 selection_bias）/ env_spec / metrics`。
  任何常數都不寫死在推論程式，一律從 checkpoint 讀。

### 4.12 `scripts/` — 資料產生與驗收

| 腳本 | 用途 |
|---|---|
| `Generate_training_data.py` | **FJSSP 訓練資料產生器（不可修改）**：6 機台；每 instance 10/15/20/25/30 個 job；每 job 4–8 道工序（順序固定）；每工序 1–6 台可行機台（同工序各機台加工時間相同，10–99）；每 job 隨機材料 A/B/C。ref: https://github.com/SchedulingLab/fjsp-instances |
| `random_job_gen.py` | 動態派工批次（legacy）：指數分佈到達間隔，每批隨機數量的單工序 job（type/station/dock），輸出 JSONL |
| `live_job_feeder.py` | 每 2 秒往 `data/live_jobs.jsonl` 追加一批隨機 job，搭配 live stream 模式 |
| `validate_contract.py` | 契約 §9 驗收：load_model → 冷啟動與中途狀態兩種 scene 各測 predict → 驗 §4 約束 / 確定性 / <1s 延遲，並輸出 `docs/examples/` 範例 scene/plan |

### 4.13 `viz/` — 視覺化

- `viz_matplotlib.py`：
  - **機台甘特圖** `draw/plot_machine_schedule`：經典 FJSSP 排程呈現——
    每台機台一條泳道，長條＝該工序的加工區間（材料色）。標籤格式
    **`J{jid}({目前工序}/{總工序數})`**（如 `J10(3/6)`），路線圖的加工中機台
    與 Plotly hover 也採同一格式（`format_trace_job_label`）；
    出貨任務顯示為 `J{jid}->T`。
  - AMR Gantt `draw/plot_amr_schedule`：transport/wait 分段＋料倉取料/庫存標註；
    加工段以**半透明**顯示（AMR 交件後已釋放，僅表示其送達的工序仍在機台上加工）。
  - 派工佇列圖、輸入（到達）佇列圖、帶時間滑桿的互動排程視窗；
    `plot_*` 版本另存 PNG 至 `results/`。
- `viz_plotly.py`：Plotly 互動 Gantt（縮放、hover 細節、時間窗捲動）。
- `viz_route_map.py`：場域路線圖——從 `trace` 重建任意時刻各 AMR 的位置與
  已走路徑快照；機台加工狀態獨立顯示（`_machine_states_at_time`：色塊高亮＋
  `J{jid}(k/N) left Xs` 剩餘時間，即使 AMR 已離開），支援時間滑桿回放/自動播放。

### 4.14 `data/`、`checkpoints/`、`docs/`、`notebooks/`、`results/`

- `data/test_data/test_dataset.jsonl`：**測試資料集——一行一筆測資**（FJSSP
  instance）；`do_test` 時自動逐筆評估（見 4.10）。資料夾內其他 `.jsonl/.json`
  也會被一併掃描。
- `data/fjssp_training_dataset.jsonl`：訓練資料（main.py 自動產生/讀取）。
- `data/data_README.md`＋`data/sample_abz5.json`：FJSSP 資料集格式說明與
  原始格式範例；`data/Generate_training_data.py` 為產生器原始對照檔
  （訓練實際 import 的是 `scripts/` 那份）。
- `data/dispatch_batches.jsonl`、`data/train_data/`：legacy 派工批次資料。
- `data/test_scenario_one_time.jsonl`：demo 測試情境；`data/live_jobs.jsonl`：live 注入檔。
- `checkpoints/ddqn_policy.pt`（續訓）/ `my_scheduler_v1.pth`（交付）/
  `contract_smoke.pth`（`--init-random` 煙霧測試專用）——**均為舊語義權重，待重訓**。
- `docs/Phase3_Model_IO_Contract.md`：I/O 契約全文；`docs/PARAMETER_GUIDE.md`：
  參數逐一解說（部分內容早於本次語義改版）；`docs/examples/`：範例 scene/plan。
- `notebooks/`：搬移成套件前的舊 notebook，import 路徑已過時，僅供參考。
- `results/test_runs/`：測試輸出——`summary.csv`（scenario / jobs / makespan /
  finish_sum / objective / eval_seconds）＋每筆測資的三面板影片。

## 5. 環境需求與工作流程

**執行環境**：conda 環境 `pytoch`（Python 3.10、PyTorch 2.9+cu126）。
必要套件：torch、numpy、matplotlib、Pillow；選用：plotly（互動 Gantt）、
`imageio-ffmpeg`（mp4 影片輸出，已安裝；沒有時自動退回 GIF）。
**務必先 `conda activate pytoch` 再執行**（直接呼叫 python.exe 會因 PATH 缺
MKL DLL 而閃退）。

```bash
conda activate pytoch

# 1. 訓練（自動產生訓練資料；結束時同時輸出兩種 checkpoint）
python main.py

# 2. 測試：把測資放進 data/test_data/（如 test_dataset.jsonl，一行一筆），
#    跑 main.py（do_test=True）即自動逐筆評估：
#    -> 印出每筆的模型計算時間（eval_seconds）與 makespan
#    -> 寫入 results/test_runs/summary.csv
#    -> 前 N 筆各輸出一支三面板影片（機台甘特+AMR甘特+路線圖，.mp4）
#    只要數字不要影片：save_test_videos=False

# 3. 契約驗收：load_model -> predict -> 檢查 §4 約束 / 確定性 / <1s 延遲
python scripts/validate_contract.py                  # 用 checkpoints/my_scheduler_v1.pth
python scripts/validate_contract.py --init-random    # 未訓練權重的管線煙霧測試

# （選用）live stream 展示：終端 A 跑 feeder，終端 B 的 main.py 開 show_live_stream
python scripts/live_job_feeder.py
```

整合方拿到 `my_scheduler_v1.pth` 後：

```python
from inference import load_model
scheduler = load_model("my_scheduler_v1.pth")
plan = scheduler.predict(scene)   # scene: 契約 §3；plan: 契約 §4
```

## 6. 與契約的對應

| 契約項目 | 本 repo 實作 |
|---|---|
| §2 環境事實 | `configs/env_spec.json`（env 由此讀參數，改版換檔即可） |
| §3 scene 輸入 | `inference/scheduler.py: Scheduler._scene_to_episode`（絕對時間→相對時間） |
| §4 plan 輸出 | 自迴歸 rollout 逐步派工，結構性滿足約束；`validate_plan` 再驗一次 |
| §5 推論介面 | `inference.load_model()` / `Scheduler.predict()` |
| §6 自描述權重檔 | `inference/checkpoint_io.py: export_contract_checkpoint` |
| §8 訓練注意事項 | `env.reset(scenario, init_state=...)` 支援中途狀態訓練；dock 取料耗時=duration、dock 互斥已入模擬器 |

> 契約模式下 env 的送 T 行為會關閉（plan 每個 job 恰為 pickup+unload 兩操作）。

## 7. 重要語義（FJSSP 交件 + 出貨 + 批次取料）

- **交件即釋放（機台獨立加工）**：AMR 送達選定機台後，若機台仍在加工前一件
  則原地等待（站點互斥）；機台空出即交件，AMR 立刻釋放接新任務，
  機台自行加工到完工才釋放該 job 的下一道工序。
- **出貨任務**：最後一道工序完工即釋放送 T 任務（取件耗時 0、無加工），
  job 送達 T 才完成；makespan ＝ 最後一個 job 到達 T 的時間。
  出貨口無互斥、不佔機台；契約推論端（`inference/scheduler.py`）自動關閉此行為。
- **取料工序（op0 / 單工序 job）**：動作 = 選 task + 選「這趟在料倉取幾份該材料」：
  - 車上庫存 0 → 必取 1~3 份（容量上限每種 3）；
  - 車上庫存 >0 → 可 `add=0` 直接用存貨送站（**不進料倉**），或順路補到滿（proactive）。
  - 取 N 份耗時 = N × 材料 duration（A/B/C = 5/10/15），期間佔用料倉（互斥）。
- **搬運工序（FJSSP op>0）**：到前一站取在製品（耗時 0），單一動作、不批次。
- **動作評分** `Score(i) = Q(i) + cover_bonus + load_bonus + wait_bonus`
  （features.py `select_action_index`；權重隨 checkpoint 的 `selection_bias` 交付，
  推論端 `predict()` 使用與訓練相同的權重）。
- **契約 plan 的批次對應**：每份材料入庫時記下料倉造訪的子區間（N 份 = N 段
  duration），job 消耗哪份（FIFO）其 `pickup` 就歸屬那段 → 同車連續 pickup 在
  `order` 中相鄰，整合方重演時自然形成批次取料；§4 約束仍結構性成立。
- **推論端保守規則**：scene 給的初始庫存不拿來抵扣（`consume_initial_inventory=False`），
  因為整合方會重演每個 job 自己的 pickup；訓練端則可完整使用庫存。
  另外若 scene 中同一材料來自多個 dock，批次取料無法保真，`predict()` 會自動
  降級為「每次進倉只取一份」（材料↔料倉一對一的場域不受影響）。
- **目標函數**：reward 對齊契約評估目標 `makespan + 0.001×Σ(各AMR完工時間)`
  （權重見 `env.objective_load_balance_weight`）。
- dock 等待與取料以「路徑停留步」寫進 transport path，避碰預約自然涵蓋 dock 佔用。

> **效能報告注意**：訓練與推論用的是快速估計（BFS 最短路、無避碰），
> 對方系統會用 A*+避碰重演 plan——因此 makespan 數字**必須以對方模擬器重演
> 的結果為準**，本 repo 內部數字只能當相對比較用。

## 8. 主要超參數速查（詳見 `docs/PARAMETER_GUIDE.md`）

| 類別 | 參數（`main.py`） |
|---|---|
| 資料 | `use_fjssp_dataset` / `fjssp_num_instances` / `fjssp_regenerate`；legacy：`multi_streams`、`num_streams`、`gen_*` |
| DDQN | `lr=1e-3`、`num_episodes=200`、`batch_size=128`、`gamma=0.99`、`grad_clip_norm=10` |
| Rainbow | `use_rainbow=True`、`rainbow_num_atoms=51`、`rainbow_v_min=-10000`、`rainbow_n_step=3`、`per_alpha=0.5`、`per_beta 0.4→1.0`、`use_noisy_exploration=True`、`target_sync_steps=2000` |
| 批次取料 | `allow_proactive_replenish`、`proactive_*_bias_weight`（cover 2.5 / load 1.8 / wait 1.5） |
| 避碰 | `train_collision_avoidance=False`、`test_collision_avoidance=True`（demo 回合） |
| 測試資料夾 | `test_data_dir`、`test_folder_collision_avoidance=False`、`save_test_videos`、`test_video_max_scenarios=3`、`test_video_fps/max_frames/dpi` |
| 視覺化 | `show_train_schedule`、`show_train_route_map`（長訓關閉）、`show_route_map`、`show_plotly` |

> `rainbow_v_min` 需覆蓋 n-step 目標範圍：|Q| ≈ (平均步成本) × 1/(1−γ)。
> 30-job FJSSP（makespan 數千、200+ 派工、γ=0.99）約 −3000，
> −10000 留有餘裕；instance 變大或 γ 提高時需重新估算。
