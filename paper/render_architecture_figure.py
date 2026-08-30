"""Render a compact, editable overview of the policy architecture.

Outputs:
  fig_architecture.pdf     vector artwork included by LaTeX
  fig_architecture.drawio  native diagrams.net XML for manual editing/import
"""

from math import atan2, cos, sin
from pathlib import Path
import xml.etree.ElementTree as ET

from reportlab.lib.colors import HexColor
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


PAPER_DIR = Path(__file__).resolve().parent
PDF_OUTPUT = PAPER_DIR / "fig_architecture.pdf"
DRAWIO_OUTPUT = PAPER_DIR / "fig_architecture.drawio"

PAGE_W, PAGE_H = 515.0, 230.0

COLORS = {
    "text": "#202020",
    "muted": "#5F5E5A",
    "panel_fill": "#FAFAF8",
    "panel_edge": "#C8C7C2",
    "data_fill": "#F1F1EF",
    "data_edge": "#77766F",
    "surrogate_fill": "#FFF1CC",
    "surrogate_edge": "#A86200",
    "state_fill": "#F0EAFE",
    "state_edge": "#654EA3",
    "learn_fill": "#E6F0FA",
    "learn_edge": "#185FA5",
    "action_fill": "#E8F3EA",
    "action_edge": "#3A7752",
    "executor_fill": "#FCE9E5",
    "executor_edge": "#A63C2F",
    "train_fill": "#F1EAF7",
    "train_edge": "#69438E",
    "arrow": "#45443F",
    "training_arrow": "#7A4E97",
}


def wrap_lines(text: str, font: str, size: float, max_width: float) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if stringWidth(candidate, font, size) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_box(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str,
    fill: str,
    edge: str,
    *,
    title_size: float = 7.1,
    body_size: float = 5.8,
) -> None:
    c.setFillColor(HexColor(fill))
    c.setStrokeColor(HexColor(edge))
    c.setLineWidth(0.85)
    c.roundRect(x, y, w, h, 5, fill=1, stroke=1)

    title_lines = wrap_lines(title, "Helvetica-Bold", title_size, w - 10)
    body_lines = wrap_lines(subtitle, "Helvetica", body_size, w - 10)
    title_leading = title_size + 1.0
    body_leading = body_size + 0.8
    total = len(title_lines) * title_leading + len(body_lines) * body_leading
    if body_lines:
        total += 1.0
    cursor = y + (h + total) / 2 - title_leading + 0.5

    c.setFillColor(HexColor(COLORS["text"]))
    c.setFont("Helvetica-Bold", title_size)
    for line in title_lines:
        c.drawCentredString(x + w / 2, cursor, line)
        cursor -= title_leading
    c.setFillColor(HexColor(COLORS["muted"]))
    c.setFont("Helvetica", body_size)
    cursor -= 1.0
    for line in body_lines:
        c.drawCentredString(x + w / 2, cursor, line)
        cursor -= body_leading


def draw_panel(c: canvas.Canvas, x: float, y: float, w: float, h: float, title: str) -> None:
    c.setFillColor(HexColor(COLORS["panel_fill"]))
    c.setStrokeColor(HexColor(COLORS["panel_edge"]))
    c.setLineWidth(0.65)
    c.roundRect(x, y, w, h, 7, fill=1, stroke=1)
    c.setFillColor(HexColor(COLORS["text"]))
    c.setFont("Helvetica-Bold", 7.1)
    c.drawString(x + 7, y + h - 11, title)


def draw_decision(
    c: canvas.Canvas,
    cx: float,
    cy: float,
    w: float,
    h: float,
    lines: tuple[str, ...],
) -> None:
    """Draw a compact decision diamond with centred text."""
    c.setFillColor(HexColor(COLORS["data_fill"]))
    c.setStrokeColor(HexColor(COLORS["data_edge"]))
    c.setLineWidth(0.85)
    path = c.beginPath()
    path.moveTo(cx, cy + h / 2)
    path.lineTo(cx + w / 2, cy)
    path.lineTo(cx, cy - h / 2)
    path.lineTo(cx - w / 2, cy)
    path.close()
    c.drawPath(path, fill=1, stroke=1)

    c.setFillColor(HexColor(COLORS["text"]))
    c.setFont("Helvetica-Bold", 5.8)
    leading = 6.6
    cursor = cy + (len(lines) - 1) * leading / 2 - 2.0
    for line in lines:
        c.drawCentredString(cx, cursor, line)
        cursor -= leading


def draw_arrow(
    c: canvas.Canvas,
    points: list[tuple[float, float]],
    *,
    color: str | None = None,
    dashed: bool = False,
    width: float = 0.85,
) -> None:
    color = color or COLORS["arrow"]
    c.setStrokeColor(HexColor(color))
    c.setFillColor(HexColor(color))
    c.setLineWidth(width)
    if dashed:
        c.setDash(3.2, 2.2)
    path = c.beginPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    c.drawPath(path, fill=0, stroke=1)
    c.setDash()

    x0, y0 = points[-2]
    x1, y1 = points[-1]
    angle = atan2(y1 - y0, x1 - x0)
    head, spread = 4.2, 2.2
    left = (x1 - head * cos(angle) + spread * sin(angle),
            y1 - head * sin(angle) - spread * cos(angle))
    right = (x1 - head * cos(angle) - spread * sin(angle),
             y1 - head * sin(angle) + spread * cos(angle))
    arrowhead = c.beginPath()
    arrowhead.moveTo(x1, y1)
    arrowhead.lineTo(*left)
    arrowhead.lineTo(*right)
    arrowhead.close()
    c.drawPath(arrowhead, fill=1, stroke=0)


def draw_label(c: canvas.Canvas, x: float, y: float, text: str) -> None:
    c.setFillColor(HexColor(COLORS["muted"]))
    c.setFont("Helvetica-Oblique", 5.5)
    c.drawCentredString(x, y, text)


def render_pdf(path: Path) -> None:
    c = canvas.Canvas(str(path), pagesize=(PAGE_W, PAGE_H))
    c.setTitle("Compact execution-grounded policy architecture")

    draw_panel(c, 7, 78, 501, 145, "A. Schedule construction")
    draw_box(c, 18, 157, 60, 36, "Schedule prefix", "committed events",
             COLORS["data_fill"], COLORS["data_edge"])
    draw_box(c, 94, 157, 70, 36, "Calibrated surrogate", "project clocks",
             COLORS["surrogate_fill"], COLORS["surrogate_edge"])
    draw_box(c, 181, 149, 79, 52, "Projected state", "AMR | station | job | action",
             COLORS["state_fill"], COLORS["state_edge"])
    draw_box(c, 278, 141, 101, 68, "Heterogeneous encoder",
             "robot-station attention | job fusion | commitment GIN",
             COLORS["learn_fill"], COLORS["learn_edge"])
    draw_box(c, 400, 151, 92, 48, "Joint actor + mask",
             "score feasible (operation, AMR, job)",
             COLORS["action_fill"], COLORS["action_edge"])

    draw_arrow(c, [(78, 175), (94, 175)])
    draw_arrow(c, [(164, 175), (181, 175)])
    draw_arrow(c, [(260, 175), (278, 175)])
    draw_arrow(c, [(379, 175), (400, 175)])

    draw_decision(c, 446, 112, 74, 42, ("All 2n actions", "committed?"))
    draw_arrow(c, [(446, 151), (446, 133)])
    draw_arrow(c, [(409, 112), (48, 112), (48, 157)])
    draw_label(c, 292, 116, "No: next decision")

    draw_panel(c, 7, 5, 501, 65, "B. Complete-schedule evaluation and PPO training")
    draw_box(c, 206, 16, 72, 28, "Fixed executor", "routing + service",
             COLORS["executor_fill"], COLORS["executor_edge"], title_size=6.6, body_size=5.1)
    draw_box(c, 300, 16, 72, 28, "Terminal return", "negative score",
             COLORS["executor_fill"], COLORS["executor_edge"], title_size=6.6, body_size=5.1)
    draw_box(c, 394, 16, 72, 28, "PPO update", "policy + value",
             COLORS["train_fill"], COLORS["train_edge"], title_size=6.6, body_size=5.1)
    draw_box(c, 110, 16, 75, 28, "Critic baseline", "mean-pooled states",
             COLORS["learn_fill"], COLORS["learn_edge"], title_size=6.4, body_size=5.1)

    draw_arrow(c, [(446, 91), (446, 74), (242, 74), (242, 44)])
    draw_label(c, 460, 83, "Yes")
    draw_arrow(c, [(278, 30), (300, 30)])
    draw_arrow(c, [(372, 30), (394, 30)], color=COLORS["training_arrow"], dashed=True)
    draw_arrow(c, [(185, 30), (194, 30), (194, 10), (430, 10), (430, 16)],
               color=COLORS["training_arrow"], dashed=True)
    c.showPage()
    c.save()


def html_label(title: str, subtitle: str) -> str:
    return f"<b>{title}</b><br><font color='{COLORS['muted']}'>{subtitle}</font>"


def render_drawio(path: Path) -> None:
    """Write native diagrams.net XML with independently editable objects."""
    mxfile = ET.Element("mxfile", host="app.diagrams.net", agent="Codex", version="24.7.17")
    diagram = ET.SubElement(mxfile, "diagram", id="architecture", name="Page-1")
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        dx="1200", dy="800", grid="1", gridSize="10", guides="1",
        tooltips="1", connect="1", arrows="1", fold="1", page="1",
        pageScale="1", pageWidth="1100", pageHeight="850", math="0", shadow="0",
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")

    def vertex(cell_id: str, value: str, x: float, y: float, w: float, h: float,
               fill: str, edge: str, *, panel: bool = False) -> None:
        if panel:
            style = (
                "rounded=1;whiteSpace=wrap;html=1;verticalAlign=top;align=left;"
                "spacingTop=8;spacingLeft=8;fontStyle=1;fontSize=15;"
                f"fillColor={fill};strokeColor={edge};arcSize=6;"
            )
        else:
            style = (
                "rounded=1;whiteSpace=wrap;html=1;align=center;verticalAlign=middle;"
                "fontFamily=Helvetica;fontSize=13;spacing=6;arcSize=12;"
                f"fillColor={fill};strokeColor={edge};strokeWidth=2;"
            )
        cell = ET.SubElement(root, "mxCell", id=cell_id, value=value, style=style,
                             vertex="1", parent="1")
        ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w),
                      height=str(h), **{"as": "geometry"})

    def edge(cell_id: str, source: str, target: str, *, dashed: bool = False,
             reverse: bool = False, points: list[tuple[float, float]] | None = None,
             ports: str = "") -> None:
        color = COLORS["training_arrow"] if dashed else COLORS["arrow"]
        style = (
            "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
            f"html=1;strokeWidth=2;strokeColor={color};endArrow=block;endFill=1;"
            f"dashed={1 if dashed else 0};{ports}"
        )
        cell = ET.SubElement(root, "mxCell", id=cell_id, style=style, edge="1",
                             parent="1", source=source, target=target)
        geometry = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
        if points:
            array = ET.SubElement(geometry, "Array", **{"as": "points"})
            for px, py in points:
                ET.SubElement(array, "mxPoint", x=str(px), y=str(py))

    vertex("panel_a", "A. Schedule construction", 14, 14, 1002, 300,
           COLORS["panel_fill"], COLORS["panel_edge"], panel=True)
    vertex("prefix", html_label("Schedule prefix", "committed events"), 36, 105, 120, 72,
           COLORS["data_fill"], COLORS["data_edge"])
    vertex("surrogate", html_label("Calibrated surrogate", "project clocks"), 188, 105, 140, 72,
           COLORS["surrogate_fill"], COLORS["surrogate_edge"])
    vertex("state", html_label("Projected state", "AMR | station | job | action"), 360, 89, 160, 104,
           COLORS["state_fill"], COLORS["state_edge"])
    vertex("encoder", html_label("Heterogeneous encoder", "robot-station attention<br>job fusion<br>commitment GIN"),
           556, 73, 202, 136, COLORS["learn_fill"], COLORS["learn_edge"])
    vertex("actor", html_label("Joint actor + mask", "score feasible<br>(operation, AMR, job)"),
           800, 95, 184, 92, COLORS["action_fill"], COLORS["action_edge"])

    vertex("complete_check", "<b>All 2n actions<br>committed?</b>", 840, 220, 104, 70,
           COLORS["data_fill"], COLORS["data_edge"])
    check_cell = root.find("./mxCell[@id='complete_check']")
    if check_cell is not None:
        check_cell.set(
            "style",
            "rhombus;whiteSpace=wrap;html=1;align=center;verticalAlign=middle;"
            "fontFamily=Helvetica;fontSize=12;fontStyle=1;spacing=4;"
            f"fillColor={COLORS['data_fill']};strokeColor={COLORS['data_edge']};strokeWidth=2;",
        )

    edge("e1", "prefix", "surrogate")
    edge("e2", "surrogate", "state")
    edge("e3", "state", "encoder")
    edge("e4", "encoder", "actor")
    edge("actor_to_check", "actor", "complete_check",
         ports="exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
    edge("feedback", "complete_check", "prefix", points=[(780, 300), (96, 300)],
         ports="exitX=0;exitY=0.5;entryX=0.5;entryY=1;")

    vertex("no_label", "<i>No: next decision</i>", 540, 278, 150, 22,
           COLORS["panel_fill"], COLORS["panel_fill"])
    no_cell = root.find("./mxCell[@id='no_label']")
    if no_cell is not None:
        no_cell.set(
            "style",
            "text;html=1;strokeColor=none;fillColor=none;align=center;"
            "verticalAlign=middle;fontSize=11;fontColor=#5F5E5A;",
        )

    vertex("panel_b", "B. Complete-schedule evaluation and PPO training", 14, 330, 1002, 130,
           COLORS["panel_fill"], COLORS["panel_edge"], panel=True)
    vertex("critic", html_label("Critic baseline", "mean-pooled states"), 210, 375, 150, 60,
           COLORS["learn_fill"], COLORS["learn_edge"])
    vertex("executor", html_label("Fixed executor", "routing + service"), 410, 375, 150, 60,
           COLORS["executor_fill"], COLORS["executor_edge"])
    vertex("terminal", html_label("Terminal return", "negative score"), 610, 375, 150, 60,
           COLORS["executor_fill"], COLORS["executor_edge"])
    vertex("ppo", html_label("PPO update", "policy + value"), 810, 375, 150, 60,
           COLORS["train_fill"], COLORS["train_edge"])

    edge("complete_edge", "complete_check", "executor", points=[(892, 322), (485, 322)],
         ports="exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
    vertex("yes_label", "<i>Yes</i>", 900, 300, 48, 22,
           COLORS["panel_fill"], COLORS["panel_fill"])
    yes_cell = root.find("./mxCell[@id='yes_label']")
    if yes_cell is not None:
        yes_cell.set(
            "style",
            "text;html=1;strokeColor=none;fillColor=none;align=center;"
            "verticalAlign=middle;fontSize=11;fontColor=#5F5E5A;",
        )

    edge("e6", "executor", "terminal")
    edge("e7", "terminal", "ppo", dashed=True)
    edge("e8", "critic", "ppo", dashed=True,
         points=[(385, 450), (885, 450)],
         ports="exitX=1;exitY=0.5;entryX=0.5;entryY=1;")

    tree = ET.ElementTree(mxfile)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    render_pdf(PDF_OUTPUT)
    render_drawio(DRAWIO_OUTPUT)


if __name__ == "__main__":
    main()
