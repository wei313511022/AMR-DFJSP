import json
import pathlib
import webbrowser
from typing import Dict, List, Optional, Union

from viz.viz_matplotlib import (
    build_arrivals,
    build_robot_labels,
    format_job_label,
    format_trace_job_label,
    split_transport_intervals,
)


def show_interactive_schedule_plotly(
    trace: List[dict],
    input_source: Union[dict, List[dict]],
    makespan: Optional[float],
    inventories: Optional[List[Dict[str, int]]] = None,
    window: float = 80.0,
    step: float = 5.0,
    max_steps: int = 300,
    html_path: str = "interactive_schedule.html",
    title_info: Optional[str] = None,
):
    """Plotly viewer with HTML controls (time slider + window input)."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly not installed. Run: pip install plotly")
        return

    if not trace:
        print("No trace to plot.")
        return

    total_proc = sum(float(item.get("proc_time", 0.0)) for item in trace)
    arrivals, _t0 = build_arrivals(input_source)
    max_arrival = max((float(item.get("arrival_t", 0.0)) for item in arrivals), default=0.0)
    max_t = max(makespan or 0.0, total_proc, max_arrival + 1.0, 1.0)
    window = min(max(window, 1.0), max_t)

    type_to_color = {"A": "#2B6CB0", "B": "#DD6B20", "C": "#2F855A"}
    robot_labels = build_robot_labels(inventories)

    buckets: Dict[int, List[dict]] = {}
    for item in arrivals:
        sec = int(float(item.get("arrival_t", 0.0)))
        buckets.setdefault(sec, []).append(item)

    max_slots = max((len(v) for v in buckets.values()), default=1)
    slot_labels = [f"slot {i}" for i in range(1, max_slots + 1)]

    arrival_width = 2.0
    in_x, in_base, in_text, in_color, in_y, in_arrival = [], [], [], [], [], []
    for sec in sorted(buckets.keys()):
        for slot_idx, item in enumerate(buckets[sec], start=1):
            jtype = str(item.get("type", "")).upper()
            jid = item.get("jid", "")
            label = format_job_label(jid, item.get("station"))
            in_x.append(arrival_width)
            in_base.append(float(sec))
            in_text.append(label)
            in_color.append(type_to_color.get(jtype, "#999999"))
            in_y.append(f"slot {slot_idx}")
            in_arrival.append(float(item.get("arrival_t", 0.0)))

    disp_x, disp_base, disp_text, disp_color, disp_y = [], [], [], [], []
    cursor = 0.0
    for item in trace:
        proc = float(item.get("proc_time", 0.0))
        jtype = item.get("type")
        label = format_trace_job_label(item)
        disp_x.append(proc)
        disp_base.append(cursor)
        disp_text.append(label)
        disp_color.append(type_to_color.get(jtype, "#999999"))
        disp_y.append("Dispatching Queue")
        cursor += proc

    proc_by_type = {
        "A": {"x": [], "base": [], "y": [], "text": []},
        "B": {"x": [], "base": [], "y": [], "text": []},
        "C": {"x": [], "base": [], "y": [], "text": []},
        "OTHER": {"x": [], "base": [], "y": [], "text": []},
    }
    to_material_x, to_material_base, to_material_y = [], [], []
    to_station_x, to_station_base, to_station_y = [], [], []
    wait_x, wait_base, wait_y = [], [], []

    for item in trace:
        rid = item["robot"]
        lane = robot_labels[rid] if rid < len(robot_labels) else f"AMR{rid+1}"
        jtype = item.get("type")
        label = format_trace_job_label(item)

        for seg in item.get("segments", []):
            s = seg["start"]
            e = seg["end"]
            dur = e - s
            if dur <= 1e-9:
                continue
            if seg["kind"] == "transport":
                split_map = split_transport_intervals(item, seg)
                for base, x in split_map["pickup_move"]:
                    to_material_x.append(x)
                    to_material_base.append(base)
                    to_material_y.append(lane)
                for base, x in split_map["station_move"]:
                    to_station_x.append(x)
                    to_station_base.append(base)
                    to_station_y.append(lane)
                for base, x in split_map["pickup_wait"] + split_map["station_wait"]:
                    wait_x.append(x)
                    wait_base.append(base)
                    wait_y.append(lane)
            elif seg["kind"] == "wait":
                wait_x.append(dur)
                wait_base.append(s)
                wait_y.append(lane)
            else:
                bucket = proc_by_type.get(jtype, proc_by_type["OTHER"])
                bucket["x"].append(dur)
                bucket["base"].append(s)
                bucket["y"].append(lane)
                bucket["text"].append(label)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.42, 0.2, 0.38],
    )
    title_text = title_info if title_info else "Interactive Schedule"

    fig.add_trace(
        go.Bar(
            x=in_x,
            base=in_base,
            y=in_y,
            orientation="h",
            marker_color=in_color,
            text=in_text,
            textposition="inside",
            insidetextanchor="middle",
            customdata=in_arrival,
            name="Input Queue",
            hovertemplate="%{text}<br>arrival=%{customdata:.1f}s<br>sec=%{base:.0f}s<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=disp_x,
            base=disp_base,
            y=disp_y,
            orientation="h",
            marker_color=disp_color,
            text=disp_text,
            textposition="inside",
            insidetextanchor="middle",
            name="Dispatching Queue",
            hovertemplate="%{text}<br>start=%{base:.1f}s<br>dur=%{x:.1f}s<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=to_material_x,
            base=to_material_base,
            y=to_material_y,
            orientation="h",
            marker=dict(color="#E6F0FF", line=dict(color="#4C72B0", width=1), pattern_shape="\\"),
            name="To Material",
            hovertemplate="To Material<br>start=%{base:.1f}s<br>dur=%{x:.1f}s<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=to_station_x,
            base=to_station_base,
            y=to_station_y,
            orientation="h",
            marker=dict(color="#E2E8F0", line=dict(color="#8A8A8A", width=1), pattern_shape="/"),
            name="To Station",
            hovertemplate="To Station<br>start=%{base:.1f}s<br>dur=%{x:.1f}s<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=wait_x,
            base=wait_base,
            y=wait_y,
            orientation="h",
            marker=dict(color="#CBD5E0", pattern_shape="."),
            name="Waiting",
            hovertemplate="Waiting<br>start=%{base:.1f}s<br>dur=%{x:.1f}s<extra></extra>",
        ),
        row=3,
        col=1,
    )

    for tkey in ["A", "B", "C", "OTHER"]:
        bucket = proc_by_type[tkey]
        if not bucket["x"]:
            continue
        name = f"Type {tkey}" if tkey in ["A", "B", "C"] else "Type Other"
        color = type_to_color.get(tkey, "#999999")
        fig.add_trace(
            go.Bar(
                x=bucket["x"],
                base=bucket["base"],
                y=bucket["y"],
                orientation="h",
                marker_color=color,
                text=bucket["text"],
                textposition="inside",
                insidetextanchor="middle",
                name=name,
                hovertemplate="%{text}<br>start=%{base:.1f}s<br>dur=%{x:.1f}s<extra></extra>",
            ),
            row=3,
            col=1,
        )

    fig.update_layout(
        height=1250,
        title=dict(text=title_text, x=0.01, y=0.995, xanchor="left", yanchor="top"),
        barmode="overlay",
        template="plotly_white",
        hovermode="x unified",
        font=dict(family="Segoe UI, Arial", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.2, xanchor="left", x=0.01),
        margin=dict(l=70, r=40, t=130, b=90),
    )

    fig.update_yaxes(
        categoryorder="array",
        categoryarray=list(reversed(slot_labels)),
        row=1,
        col=1,
        tickfont=dict(size=11),
        showticklabels=False,
        title_text="Input Queue",
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=["Dispatching Queue"],
        row=2,
        col=1,
        tickfont=dict(size=12),
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=robot_labels,
        row=3,
        col=1,
        tickfont=dict(size=12),
    )
    fig.update_xaxes(title_text="Time (s)", row=3, col=1, tickfont=dict(size=11))
    fig.update_xaxes(rangeslider_visible=True, row=3, col=1)
    fig.update_traces(textfont=dict(size=14), selector=dict(type="bar"))

    step = max(step, max_t / max(1, max_steps))

    fig_dict = fig.to_dict()
    fig_json = json.dumps(fig_dict)

    html_template = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
  <style>
    body {
      font-family: "Segoe UI", "Noto Sans", Arial, sans-serif;
      margin: 0;
      padding: 0;
      color: #1A202C;
      background: #FFFFFF;
    }
    .controls {
      display: flex;
      align-items: center;
      gap: 14px;
      padding: 12px 20px;
      border-bottom: 1px solid #E2E8F0;
      background: #F8FAFC;
    }
    .controls label {
      font-size: 13px;
      letter-spacing: 0.2px;
    }
    .controls input[type=number] {
      width: 90px;
      padding: 6px 8px;
      border: 1px solid #CBD5E0;
      border-radius: 6px;
      font-size: 13px;
    }
    .controls input[type=range] {
      flex: 1;
      height: 6px;
      accent-color: #2B6CB0;
    }
    .controls button {
      padding: 6px 12px;
      border: 1px solid #CBD5E0;
      border-radius: 6px;
      background: #FFFFFF;
      font-size: 13px;
      cursor: pointer;
    }
    #plot { width: 100%; height: 980px; }
  </style>
</head>
<body>
  <div class="controls">
    <button id="playBtn">Play</button>
    <button id="pauseBtn">Pause</button>
    <label>Center</label>
    <input id="centerSlider" type="range" min="0" max="__MAX_T__" step="__STEP__" value="0" />
    <label>Window</label>
    <input id="windowInput" type="number" min="1" max="__MAX_T__" step="1" value="__WINDOW__" />
    <span id="centerLabel">t=0.0s</span>
  </div>
  <div id="plot"></div>
  <script>
    const fig = __FIG_JSON__;
    const plot = document.getElementById('plot');
    const centerSlider = document.getElementById('centerSlider');
    const windowInput = document.getElementById('windowInput');
    const centerLabel = document.getElementById('centerLabel');
    const playBtn = document.getElementById('playBtn');
    const pauseBtn = document.getElementById('pauseBtn');
    let timer = null;

    function clamp(val, min, max) {
      return Math.max(min, Math.min(max, val));
    }

    function updateWindow(center, windowSize) {
      const half = windowSize / 2.0;
      const left = clamp(center - half, 0.0, __MAX_T__);
      const right = clamp(center + half, 0.0, __MAX_T__);
      const line = {type: 'line', x0: center, x1: center, y0: 0, y1: 1, xref: 'x', yref: 'paper', line: {color: '#E53E3E', width: 1}};
      Plotly.relayout(plot, {
        'xaxis.range': [left, right],
        'xaxis2.range': [left, right],
        'xaxis3.range': [left, right],
        'shapes': [line]
      });
      centerLabel.textContent = `t=${center.toFixed(1)}s`;
    }

    Plotly.newPlot(plot, fig.data, fig.layout, {responsive: true}).then(() => {
      updateWindow(0.0, parseFloat(windowInput.value));
    });

    centerSlider.addEventListener('input', () => {
      const center = parseFloat(centerSlider.value);
      const windowSize = parseFloat(windowInput.value);
      updateWindow(center, windowSize);
    });

    windowInput.addEventListener('change', () => {
      const center = parseFloat(centerSlider.value);
      const windowSize = parseFloat(windowInput.value);
      updateWindow(center, windowSize);
    });

    playBtn.addEventListener('click', () => {
      if (timer) return;
      timer = setInterval(() => {
        let next = parseFloat(centerSlider.value) + __STEP__;
        if (next > __MAX_T__) next = 0.0;
        centerSlider.value = next.toFixed(2);
        updateWindow(next, parseFloat(windowInput.value));
      }, 150);
    });

    pauseBtn.addEventListener('click', () => {
      if (timer) {
        clearInterval(timer);
        timer = null;
      }
    });
  </script>
</body>
</html>
"""

    html = html_template.replace("__FIG_JSON__", fig_json)
    html = html.replace("__MAX_T__", f"{max_t:.2f}")
    html = html.replace("__STEP__", f"{step:.2f}")
    html = html.replace("__WINDOW__", f"{window:.2f}")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    webbrowser.open(pathlib.Path(html_path).absolute().as_uri())
