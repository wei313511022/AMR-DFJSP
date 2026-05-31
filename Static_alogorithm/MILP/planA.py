import gurobipy as gp
from gurobipy import GRB
import random
import numpy as np
import time

# --- 函數：計算兩點間最短路徑時間 (曼哈頓距離) ---
GRID_SIZE = 10 
def calculate_distance(node1, node2):
    """計算 10x10 網格上的曼哈頓距離 (最短步數)。"""
    r1, c1 = (node1 - 1) // GRID_SIZE, (node1 - 1) % GRID_SIZE
    r2, c2 = (node2 - 1) // GRID_SIZE, (node2 - 1) % GRID_SIZE
    return abs(r1 - r2) + abs(c1 - c2)

# --- 預計算距離矩陣 (用於簡化約束) ---
P_NODES = {1, 6, 10} # Pickup Nodes: (0,0), (0,5), (0,9)
D_NODES = {93, 96, 98} # Delivery Nodes: (9,2), (9,5), (9,7)
ALL_STATIONS = list(P_NODES | D_NODES)

DISTANCES = {}
for n1 in ALL_STATIONS:
    for n2 in ALL_STATIONS:
        DISTANCES[(n1, n2)] = calculate_distance(n1, n2)

# --- 站點和執行時間的固定映射 (Type 決定 D_node) ---
TASK_TYPE_MAPPING = {
    1: {'E_l': 7,  'g_l': 93, 'type': 1},  # Type 1: uses D1(93)
    2: {'E_l': 11, 'g_l': 96, 'type': 2}, # Type 2: uses D2(96)
    3: {'E_l': 15, 'g_l': 98, 'type': 3}  # Type 3: uses D3(98)
}

# ====================================================
# --- 腳本開始 ---
# ====================================================
if __name__ == "__main__":
    try:
        # --- 1. 定義集合與參數 ---
        
        M_SET = range(1, 4)           # 3 輛 AGV 
        L_SET = range(1, 11)           # 7 個任務 (1到7)
        N_SET = range(1, 101)         # 100 個節點 (10x10 網格)
        
        # AGV 初始位置
        S_m = {1: 1, 2: 6, 3: 10} 
        
        # 任務資料 (Task Data) - 包含固定的 D_node 和 E_l
        random.seed(9384) # 保持隨機種子
        TASK_DATA = {}
        TASK_TYPES_SET = {1, 2, 3}
        for l in L_SET:
            task_type = random.choice(list(TASK_TYPES_SET))
            TASK_DATA[l] = TASK_TYPE_MAPPING[task_type].copy() # 複製Type屬性
        
        # 新增虛擬節點：任務 0 (起始任務), 任務 8 (結束任務)
        L_PRIME = range(0, 12) 
        M_BIG = 1000 # 巨大的 M

        # --- 2. 建立 Gurobi 模型 ---
        model = gp.Model("VRP_Optimize_Pickup")
        
        model.setParam('TimeLimit', 3600) 
        model.setParam('MIPGap', 0.0)    
        
        # --- 3. 加入變數 (Variables) ---

        Y = model.addVars(L_SET, M_SET, vtype=GRB.BINARY, name="Y_assign") 
        W = model.addVars(L_PRIME, L_PRIME, M_SET, vtype=GRB.BINARY, name="W_sequence") 
        A_P = model.addVars(L_SET, P_NODES, vtype=GRB.BINARY, name="A_Task_To_Pickup") # 僅保留 Pickup 選擇變數
        # B_D (Delivery 選擇變數) 已移除
        
        T_Pick = model.addVars(L_SET, vtype=GRB.CONTINUOUS, lb=0, name="T_Pick_Arrival") 
        T_Del = model.addVars(L_SET, vtype=GRB.CONTINUOUS, lb=0, name="T_Del_Arrival")  
        T_End = model.addVars(L_SET, vtype=GRB.CONTINUOUS, lb=0, name="T_Task_End")    

        # --- 4. 設定目標函數 (最小化所有任務的執行結束時間總和) ---
        model.setObjective(gp.quicksum(T_End[l] for l in L_SET), GRB.MINIMIZE)

        # --- 5. 加入約束條件 (Constraints) ---

        # A. 站點選擇約束 
        # (C1) 每個任務必須分配給恰好一個 Pickup Station
        model.addConstrs((gp.quicksum(A_P[l, p] for p in P_NODES) == 1 for l in L_SET), name="C1_OnePickupStation")
        # (C2) Delivery 站點是固定的，無需約束 (B_D 已移除)

        # B. 序列和 AGV 分配約束 (C2, C3, C4, C5)
        model.addConstrs((gp.quicksum(Y[l, m] for m in M_SET) == 1 for l in L_SET), name="C2_AssignOneVehicle") 
        for l in L_SET:
            model.addConstr(gp.quicksum(W[l_prev, l, m] for l_prev in L_PRIME if l_prev != l for m in M_SET) == 1, name=f"C3_Predecessor_{l}")
            model.addConstr(gp.quicksum(W[l, l_next, m] for l_next in L_PRIME if l_next != l for m in M_SET) == 1, name=f"C3_Successor_{l}")
            model.addConstrs((W[l_prev, l, m] <= Y[l, m] for l_prev in L_PRIME if l_prev != l for m in M_SET), name=f"C3_Y_Link_1_{l}")
            model.addConstrs((W[l, l_next, m] <= Y[l, m] for l_next in L_PRIME if l_next != l for m in M_SET), name=f"C3_Y_Link_2_{l}")

        # C4: 虛擬起始節點 (0) 必須與 AGV 綁定
        for m in M_SET:
            model.addConstr(gp.quicksum(W[0, l, m] for l in L_SET) == 1, name=f"C4_Start_{m}")
            model.addConstr(gp.quicksum(W[l, 11, m] for l in L_SET) == 1, name=f"C4_End_{m}") 
            
        # C5: 防止 AGV 在序列中返回虛擬起始點 (0)
        model.addConstrs((W[l, 0, m] == 0 for l in L_SET for m in M_SET), name="C5_NoReturnToStart") 

        # C. 時間與距離約束 

        # C6: 任務執行完成時間: T_End = T_Del + E_l
        model.addConstrs((T_End[l] == T_Del[l] + TASK_DATA[l]['E_l'] for l in L_SET), name="C6_ExecutionTime")

        # C7: 運輸時間約束: T_Del - T_Pick >= d(u_l, g_l)
        for l in L_SET:
            g_l = TASK_DATA[l]['g_l'] # 固定 Delivery 節點
            for p in P_NODES:
                dist_pg = calculate_distance(p, g_l)
                # T_Del[l] >= T_Pick[l] + d(p, g_l) - M * (1 - A_P[l,p])
                model.addConstr(T_Del[l] >= T_Pick[l] + dist_pg - M_BIG * (1 - A_P[l, p]), name=f"C7_Trans_{l}_{p}")


        # C8: 序列時間約束: T_Pick[l] >= T_End[l_prev] + d(g_l_prev, u_l)
        for l in L_SET:
            # C8.1: 任務到任務的時間銜接
            for l_prev in L_SET: 
                if l != l_prev:
                    g_l_prev = TASK_DATA[l_prev]['g_l'] # 前驅任務的固定 Delivery 節點
                    for p in P_NODES: # 任務 l 的 Pickup 節點
                        dist_dp = calculate_distance(g_l_prev, p)

                        # T_Pick[l] >= T_End[l_prev] + dist_dp - M * (2 - W[l_prev, l, m] - A_P[l, p])
                        model.addConstrs((T_Pick[l] >= T_End[l_prev] + dist_dp - M_BIG * (2 - W[l_prev, l, m] - A_P[l, p])
                                          for m in M_SET), name=f"C8_T_T_{l_prev}_{l}_{p}")

            # C8.2: 虛擬起始節點 (0) 的時間約束 (AGV 從 S_m 開始)
            for m in M_SET:
                S_node = S_m[m] 
                for p in P_NODES:
                    dist_sp = calculate_distance(S_node, p)
                    # T_Pick[l] >= dist_sp - M * (2 - W[0, l, m] - A_P[l, p])
                    model.addConstr(T_Pick[l] >= dist_sp - M_BIG * (2 - W[0, l, m] - A_P[l, p]), name=f"C8_StartToTask_{l}_{m}_{p}")


        # --- 6. 求解模型 (Solve Model) ---
        print("\n開始求解 VRP 序列模型 (100節點, 7 Task, Optimize Pickup)...")
        model.update()
        model.optimize()

        # --- 7. 輸出結果 ---
        
        print("\n==========================================================")
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            
            if model.SolCount > 0:
                time.sleep(0.2) 
                model.setParam(GRB.Param.SolutionNumber, 0)
                
                # 批量提取變數值
                W_vals = model.getAttr('X', W)
                A_P_vals = model.getAttr('X', A_P)
                T_Pick_vals = model.getAttr('X', T_Pick)
                T_Del_vals = model.getAttr('X', T_Del)
                
            else:
                 print(f"❌ 求解失敗。Gurobi 狀態碼: {model.status}")

            print(f"✅ 求解成功，總完成時間最小化成本: {model.objVal:.2f}")
            
            assignment = {m: [] for m in M_SET}
            task_time_details = {}
            task_station_map = {}
            
            # 提取序列和時間
            for m in M_SET:
                current_l = 0
                sequence = []
                for _ in range(len(L_SET) + 2): 
                    if current_l == 11: break 
                    
                    found_next = False
                    for l_next in L_PRIME:
                        if W_vals.get((current_l, l_next, m), 0) >= 0.999: 
                            if l_next != 11:
                                sequence.append(l_next)
                                
                                if l_next in L_SET:
                                    try:
                                        # 提取站點信息（現在只有 Pickup 是變數）
                                        p_node = next(p for p in P_NODES if A_P_vals.get((l_next, p), 0) >= 0.999)
                                        d_node = TASK_DATA[l_next]['g_l'] # Delivery 是固定的
                                        
                                        T_trans = T_Del_vals.get(l_next, 0) - T_Pick_vals.get(l_next, 0)
                                        T_exec = TASK_DATA[l_next]['E_l']
                                        
                                        task_time_details[l_next] = {'T_trans': T_trans, 'T_exec': T_exec}
                                        task_station_map[l_next] = {'P': p_node, 'D': d_node}
                                    
                                    except StopIteration:
                                        task_time_details[l_next] = {'T_trans': 'Error', 'T_exec': TASK_DATA[l_next]['E_l']}
                                        task_station_map[l_next] = {'P': 'Error', 'D': 'Error'}
                            
                            current_l = l_next
                            found_next = True
                            break
                    if not found_next and current_l != 11: break

                assignment[m] = sequence

            
            print("\n📌 任務分配、序列與站點優化結果:")
            for m in M_SET:
                print(f"--- AGV {m} 序列: {' -> '.join(map(str, assignment[m]))} ---")
            
            print("\n📌 每個任務的運輸時間及執行時間:")
            for l in L_SET:
                 if l in task_time_details:
                     details = task_time_details[l]
                     stations = task_station_map.get(l, {'P': 'N/A', 'D': 'N/A'}) 
                     
                     print(f"  - 任務 {l} (Type={TASK_DATA[l]['type']}, P={stations['P']}, D={stations['D']}): 運輸時間={details['T_trans']:.2f}, 執行時間={details['T_exec']:.2f}, 總成本={details['T_trans'] + details['T_exec']:.2f}")

        else:
            print(f"❌ 求解失敗。Gurobi 狀態碼: {model.status}")
            
    except gp.GurobiError as e:
        print(f"\n❌ Gurobi 錯誤代碼 {e.errno}: {e}")
        print("請檢查 Gurobi 是否已安裝並取得有效授權。")
    except Exception as e:
        print(f"\n❌ 發生一般錯誤: {e}")