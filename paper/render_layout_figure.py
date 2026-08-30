"""Render the paper layout directly from the experiment scenario."""

from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas


PAPER_DIR = Path(__file__).resolve().parent
STATIC_DIR = PAPER_DIR.parent / "AMR-DFJSP" / "Static_alogorithm"
if not STATIC_DIR.exists():
    raise FileNotFoundError(f"Experiment source not found: {STATIC_DIR}")
sys.path.insert(0, str(STATIC_DIR))

import scenario_v3 as scenario  # noqa: E402
import GA.GA as ga  # noqa: E402


PAGE_W, PAGE_H = 252.0, 235.0  # 3.5 in wide, single IEEE column
CELL = 9.6
GRID_X, GRID_Y = 29.5, 8.0
SCALE = 4.2  # approximately 300 dpi

COLORS = {
    "grid": "#D8D8D8",
    "in_fill": "#B5D4F4",
    "in_edge": "#185FA5",
    "out_fill": "#F5C4B3",
    "out_edge": "#993C1D",
    "bay_fill": "#CECBF6",
    "bay_edge": "#534AB7",
    "wait_fill": "#EDEDED",
    "wait_edge": "#BBBBBB",
    "flow": "#444441",
    "text": "#202020",
}


def geometry():
    scenario.apply_layout(num_amrs=16, depot_x=4)
    docks = tuple(ga.INBOUND_DOCK_LOCATIONS.values())
    stations = tuple(ga.STATIONS.values())
    bays = tuple(ga.AMR_STARTS.values())
    service = set(docks) | set(stations)
    waiting = {
        cell
        for station in docks + stations
        for cell in ga.dock_waiting_slots(station)
        if cell not in service
    }
    return docks, stations, bays, tuple(sorted(waiting))


def pdf_center(x: int, y: int):
    return GRID_X + (x + 0.5) * CELL, GRID_Y + (y - 0.5) * CELL


def draw_pdf(path: Path, docks, stations, bays, waiting) -> None:
    c = canvas.Canvas(str(path), pagesize=(PAGE_W, PAGE_H))

    for x in range(21):
        px = GRID_X + x * CELL
        c.setStrokeColor(HexColor(COLORS["grid"]))
        c.setLineWidth(0.25)
        c.line(px, GRID_Y, px, GRID_Y + 19 * CELL)
    for y in range(20):
        py = GRID_Y + y * CELL
        c.line(GRID_X, py, GRID_X + 20 * CELL, py)

    def square(cell, fill, edge=None, width=0.5):
        cx, cy = pdf_center(*cell)
        c.setFillColor(HexColor(fill))
        c.setStrokeColor(HexColor(edge or fill))
        c.setLineWidth(width)
        c.rect(cx - CELL / 2, cy - CELL / 2, CELL, CELL, fill=1, stroke=1)

    for cell in waiting:
        square(cell, COLORS["wait_fill"], COLORS["wait_fill"], 0.1)
    for cell in docks:
        square(cell, COLORS["in_fill"], COLORS["in_edge"], 0.9)
    for cell in stations:
        square(cell, COLORS["out_fill"], COLORS["out_edge"], 0.9)
    for cell in bays:
        square(cell, COLORS["bay_fill"], COLORS["bay_edge"], 0.9)

    # Dashed example flow bends into the free row above the middle queues.
    p0 = pdf_center(0, 10)
    p1 = pdf_center(4, 11)
    p2 = pdf_center(15, 11)
    p3 = pdf_center(19, 10)
    c.setStrokeColor(HexColor(COLORS["flow"]))
    c.setFillColor(HexColor(COLORS["flow"]))
    c.setLineWidth(0.9)
    c.setDash(4, 2)
    c.line(*p0, *p1)
    c.line(*p1, *p2)
    c.line(*p2, *p3)
    c.setDash()
    c.line(p3[0], p3[1], p3[0] - 5.5, p3[1] + 0.3)
    c.line(p3[0], p3[1], p3[0] - 3.8, p3[1] + 4.0)
    c.setFont("Helvetica-Oblique", 6.8)
    c.drawCentredString((p1[0] + p2[0]) / 2, p1[1] + 5.5, "pickup  ->  delivery")

    # Side labels.
    c.saveState()
    c.setFillColor(HexColor(COLORS["in_edge"]))
    c.setFont("Helvetica-Bold", 6.8)
    c.translate(7.0, GRID_Y + 9.5 * CELL)
    c.rotate(90)
    c.drawCentredString(0, 0, "inbound stations")
    c.restoreState()
    c.saveState()
    c.setFillColor(HexColor(COLORS["out_edge"]))
    c.setFont("Helvetica-Bold", 6.8)
    c.translate(PAGE_W - 6.0, GRID_Y + 9.5 * CELL)
    c.rotate(-90)
    c.drawCentredString(0, 0, "outbound stations")
    c.restoreState()

    # One-line legend.
    legend = [
        ("inbound station", COLORS["in_fill"], COLORS["in_edge"]),
        ("outbound station", COLORS["out_fill"], COLORS["out_edge"]),
        ("charging bay", COLORS["bay_fill"], COLORS["bay_edge"]),
        ("waiting cells", COLORS["wait_fill"], COLORS["wait_edge"]),
    ]
    c.setFont("Helvetica", 5.8)
    widths = [8 + 3 + c.stringWidth(label, "Helvetica", 5.8) for label, _, _ in legend]
    total = sum(widths) + 8 * (len(legend) - 1)
    cursor = (PAGE_W - total) / 2
    legend_y = PAGE_H - 12.0
    for (label, fill, edge), width in zip(legend, widths):
        c.setFillColor(HexColor(fill))
        c.setStrokeColor(HexColor(edge))
        c.setLineWidth(0.7)
        c.rect(cursor, legend_y - 3.5, 7, 7, fill=1, stroke=1)
        c.setFillColor(HexColor(COLORS["text"]))
        c.drawString(cursor + 10, legend_y - 2, label)
        cursor += width + 8

    c.showPage()
    c.save()


def font(name: str, size: float):
    candidates = {
        "regular": ["arial.ttf", "calibri.ttf"],
        "bold": ["arialbd.ttf", "calibrib.ttf"],
        "italic": ["ariali.ttf", "calibrii.ttf"],
    }[name]
    for filename in candidates:
        path = Path("C:/Windows/Fonts") / filename
        if path.exists():
            return ImageFont.truetype(str(path), round(size * SCALE))
    return ImageFont.load_default()


def draw_png(path: Path, docks, stations, bays, waiting) -> None:
    width, height = round(PAGE_W * SCALE), round(PAGE_H * SCALE)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    def pt(value):
        return round(value * SCALE)

    def image_center(x: int, y: int):
        return pt(GRID_X + (x + 0.5) * CELL), pt(PAGE_H - (GRID_Y + (y - 0.5) * CELL))

    for x in range(21):
        px = pt(GRID_X + x * CELL)
        draw.line((px, pt(PAGE_H - GRID_Y), px, pt(PAGE_H - GRID_Y - 19 * CELL)), fill=COLORS["grid"], width=1)
    for y in range(20):
        py = pt(PAGE_H - GRID_Y - y * CELL)
        draw.line((pt(GRID_X), py, pt(GRID_X + 20 * CELL), py), fill=COLORS["grid"], width=1)

    def square(cell, fill, edge=None, line=1):
        cx, cy = image_center(*cell)
        half = pt(CELL / 2)
        draw.rectangle((cx - half, cy - half, cx + half, cy + half), fill=fill, outline=edge or fill, width=pt(line))

    for cell in waiting:
        square(cell, COLORS["wait_fill"], COLORS["wait_fill"], 0.25)
    for cell in docks:
        square(cell, COLORS["in_fill"], COLORS["in_edge"], 0.9)
    for cell in stations:
        square(cell, COLORS["out_fill"], COLORS["out_edge"], 0.9)
    for cell in bays:
        square(cell, COLORS["bay_fill"], COLORS["bay_edge"], 0.9)

    p0 = image_center(0, 10)
    p1 = image_center(4, 11)
    p2 = image_center(15, 11)
    p3 = image_center(19, 10)

    def dashed_segment(a, b):
        import math

        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        dash, gap = pt(4), pt(2)
        cursor = 0.0
        while cursor < length:
            end = min(cursor + dash, length)
            draw.line(
                (a[0] + ux * cursor, a[1] + uy * cursor,
                 a[0] + ux * end, a[1] + uy * end),
                fill=COLORS["flow"], width=pt(0.9),
            )
            cursor += dash + gap

    dashed_segment(p0, p1)
    dashed_segment(p1, p2)
    dashed_segment(p2, p3)
    draw.polygon(
        [p3, (p3[0] - pt(5.5), p3[1] - pt(0.3)),
         (p3[0] - pt(3.8), p3[1] - pt(4.0))],
        fill=COLORS["flow"],
    )
    draw.text(((p1[0] + p2[0]) / 2, p1[1] - pt(7.5)), "pickup  ->  delivery", font=font("italic", 6.8), fill=COLORS["flow"], anchor="ms")

    def vertical_label(text, x, color, clockwise):
        f = font("bold", 6.8)
        box = f.getbbox(text)
        layer = Image.new("RGBA", (box[2] - box[0] + 12, box[3] - box[1] + 12), (255, 255, 255, 0))
        ImageDraw.Draw(layer).text((6, 6), text, font=f, fill=color, anchor="la")
        layer = layer.rotate(-90 if clockwise else 90, expand=True)
        image.paste(layer, (round(x - layer.width / 2), round(pt(PAGE_H - GRID_Y - 9.5 * CELL) - layer.height / 2)), layer)

    vertical_label("inbound stations", pt(7.0), COLORS["in_edge"], False)
    vertical_label("outbound stations", pt(PAGE_W - 6.0), COLORS["out_edge"], True)

    legend = [
        ("inbound station", COLORS["in_fill"], COLORS["in_edge"]),
        ("outbound station", COLORS["out_fill"], COLORS["out_edge"]),
        ("charging bay", COLORS["bay_fill"], COLORS["bay_edge"]),
        ("waiting cells", COLORS["wait_fill"], COLORS["wait_edge"]),
    ]
    legend_font = font("regular", 5.8)
    widths = [pt(11) + draw.textlength(label, font=legend_font) for label, _, _ in legend]
    total = sum(widths) + pt(8) * (len(legend) - 1)
    cursor = (width - total) / 2
    legend_y = pt(12)
    for (label, fill, edge), item_width in zip(legend, widths):
        draw.rectangle((cursor, legend_y - pt(3.5), cursor + pt(7), legend_y + pt(3.5)), fill=fill, outline=edge, width=pt(0.7))
        draw.text((cursor + pt(10), legend_y), label, font=legend_font, fill=COLORS["text"], anchor="lm")
        cursor += item_width + pt(8)

    image.save(path, dpi=(300, 300), optimize=True)


def main() -> None:
    items = geometry()
    draw_pdf(PAPER_DIR / "fig_layout.pdf", *items)
    draw_png(PAPER_DIR / "fig_layout.png", *items)


if __name__ == "__main__":
    main()
