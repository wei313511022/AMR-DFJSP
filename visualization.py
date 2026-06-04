#!/usr/bin/env python3
"""
Interactive Visualization module for AMR Scheduler using Plotly.
"""
import os
import pandas as pd
import plotly.graph_objects as go

def save_dpr_schedule_interactive(schedule, output_path="schedule.html", rule_name="Scheduler"):
    """
    接收 JSON array，並生成包含互動式縮放、平移與懸浮提示的 HTML 網頁。
    """
    # 1. 防呆與路徑處理
    if not schedule:
        print("警告：傳入的排程為空。")
        return
        
    out_dir = os.path.dirname(str(output_path))
    if out_dir: 
        os.makedirs(out_dir, exist_ok=True)

    # 2. 將 List of Dicts 轉為 Pandas DataFrame 方便分群處理
    df = pd.DataFrame(schedule)
    
    # 確保必要欄位存在 (依據你傳入的 jsonl 格式)
    for col in ['job', 'op_index', 'machine', 'amr', 'start', 'end']:
        if col not in df.columns:
            df[col] = 0

    # 3. 資料前處理
    # 計算加工時間 (區塊寬度)
    df['processing_time'] = df['end'] - df['start']
    # 為了讓圖例 (Legend) 按任務分色，將 job 轉為字串標籤
    df['Job_Label'] = df['job'].astype(str)
    # Y 軸標籤
    df['AMR_Label'] = 'AMR ' + df['amr'].astype(str)

    # 4. 建立圖表
    fig = go.Figure()

    # 將資料依照 Job 進行分群繪製，這樣 Plotly 會自動為同一個 Job 分配相同顏色
    for job_label, group in df.groupby('Job_Label'):
        fig.add_trace(go.Bar(
            name=job_label,
            y=group['AMR_Label'],
            x=group['processing_time'], # 區塊的長度
            base=group['start'],        # 區塊的起點
            orientation='h',
            # 區塊內部顯示的文字：J_任務-工序
            text=group.apply(lambda row: f"J_{row['job']}-{row['op_index']}", axis=1),
            textposition='inside',
            insidetextanchor='middle',
            textfont=dict(color='white', size=10),
            # 懸浮視窗 (Hover) 顯示的詳細資訊
            hovertemplate=(
                "<b>%{y}</b><br>" +
                "Job: %{data.name}<br>" +
                "Operation: %{customdata[0]}<br>" +
                "Machine: %{customdata[1]}<br>" +
                "Start Time: %{base}<br>" +
                "End Time: %{customdata[2]}<br>" +
                "Processing Time: %{customdata[3]}<extra></extra>" # extra 取消旁邊多餘的 trace 名稱
            ),
            # 將其他資訊打包放進 customdata 供 hovertemplate 讀取
            customdata=group[['op_index', 'machine', 'end', 'processing_time']].values
        ))

    # 5. 版面與互動優化
    makespan = df['end'].max()
    amr_count = df['amr'].nunique()
    dynamic_width = max(1200, makespan * 0.5)
    
    fig.update_layout(
        title=f"<b>{rule_name}</b> | Makespan: {makespan:.1f}",
        barmode='overlay', # 確保區塊堆疊在同一水平線上
        yaxis=dict(
            autorange="reversed", # 反轉 Y 軸，讓 AMR 1 在最上面
            categoryorder='category ascending' 
        ),
        xaxis=dict(
            title="Time",
            zeroline=False,
            gridcolor='#eeeeee'
        ),
        hovermode="closest",
        template="plotly_white",
        width=dynamic_width + 50,
        height=max(400, amr_count * 120), # 依據 AMR 數量動態決定高度
        margin=dict(l=80, r=20, t=60, b=40)
    )

    # 6. 輸出為獨立的 HTML 檔案
    fig.write_html(str(output_path))