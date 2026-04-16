#!/usr/bin/env python3
"""Generate Slide 3 visual: MRR/Recall@1 vs latency using PIL only."""

from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).resolve().parents[1] / "visuals" / "iu" / "s03_quality_efficiency_tradeoff.png"
OUT.parent.mkdir(parents=True, exist_ok=True)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates += [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/System/Library/Fonts/SFNS.ttf",
        ]
    candidates += [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def px_x(x: float, x0: int, x1: int, xmin: float, xmax: float) -> int:
    return int(x0 + (x - xmin) / (xmax - xmin) * (x1 - x0))


def px_y(y: float, y0: int, y1: int, ymin: float, ymax: float) -> int:
    return int(y1 - (y - ymin) / (ymax - ymin) * (y1 - y0))


def main() -> None:
    w, h = 1600, 900
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    title_font = load_font(44, bold=True)
    label_font = load_font(24, bold=True)
    tick_font = load_font(20, bold=False)
    note_font = load_font(18, bold=False)
    ann_font = load_font(19, bold=False)

    # Plot bounds
    x0, x1 = 140, 1510
    y0, y1 = 120, 710
    xmin, xmax = 0.12, 1.08
    ymin, ymax = 0.42, 0.78

    # Data
    points = [
        {"label": "Teacher (no FT)", "lat": 1.00, "mrr": 0.587, "r1": 0.458, "color": (83, 109, 254)},
        {"label": "Teacher (FT)", "lat": 1.00, "mrr": 0.731, "r1": 0.620, "color": (46, 125, 50)},
        {"label": "Student target (90%)", "lat": 22.0 / 125.0, "mrr": 0.658, "r1": 0.558, "color": (245, 124, 0)},
        {"label": "Student stretch (95%)", "lat": 33.0 / 125.0, "mrr": 0.694, "r1": 0.589, "color": (216, 27, 96)},
    ]

    # Grid and axes
    for y in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        py = px_y(y, y0, y1, ymin, ymax)
        draw.line((x0, py, x1, py), fill=(224, 224, 224), width=2)
        draw.text((68, py - 12), f"{y:.2f}", fill=(60, 60, 60), font=tick_font)

    for x in [0.2, 0.4, 0.6, 0.8, 1.0]:
        px = px_x(x, x0, x1, xmin, xmax)
        draw.line((px, y0, px, y1), fill=(240, 240, 240), width=2)
        draw.text((px - 15, y1 + 16), f"{x:.1f}", fill=(60, 60, 60), font=tick_font)

    draw.line((x0, y0, x0, y1), fill=(15, 15, 15), width=4)
    draw.line((x0, y1, x1, y1), fill=(15, 15, 15), width=4)

    # Data points
    for idx, p in enumerate(points):
        c = p["color"]
        px = px_x(p["lat"], x0, x1, xmin, xmax)
        py_mrr = px_y(p["mrr"], y0, y1, ymin, ymax)
        py_r1 = px_y(p["r1"], y0, y1, ymin, ymax)

        # vertical connector
        draw.line((px, py_mrr, px, py_r1), fill=c, width=3)

        # MRR circle
        r = 12
        draw.ellipse((px - r, py_mrr - r, px + r, py_mrr + r), fill=c, outline=(0, 0, 0), width=2)

        # Recall@1 square
        s = 11
        draw.rectangle((px - s, py_r1 - s, px + s, py_r1 + s), fill=c, outline=(0, 0, 0), width=2)

        # Annotation near MRR point
        ox = 18 if idx < 2 else 10
        oy = -12 if idx != 1 else -28
        draw.text((px + ox, py_mrr + oy), p["label"], fill=(25, 25, 25), font=ann_font)

    # Titles and labels
    title = "MRR and Recall@1 vs Latency (Teacher vs Student Targets)"
    draw.text((w // 2 - 520, 32), title, fill=(15, 15, 15), font=title_font)
    draw.text((w // 2 - 290, 760), "Estimated latency index (teacher=1.0, lower is faster)", fill=(20, 20, 20), font=label_font)
    draw.text((20, (y0 + y1) // 2 - 30), "Retrieval quality", fill=(20, 20, 20), font=label_font)

    # Legend
    lx, ly = 1140, 120
    draw.rectangle((lx - 20, ly - 20, lx + 360, ly + 110), fill=(250, 250, 250), outline=(170, 170, 170), width=2)
    draw.ellipse((lx, ly, lx + 24, ly + 24), fill=(90, 90, 90), outline=(0, 0, 0), width=2)
    draw.text((lx + 34, ly - 1), "MRR", fill=(20, 20, 20), font=tick_font)
    draw.rectangle((lx, ly + 46, lx + 24, ly + 70), fill=(90, 90, 90), outline=(0, 0, 0), width=2)
    draw.text((lx + 34, ly + 45), "Recall@1", fill=(20, 20, 20), font=tick_font)

    foot = "Teacher scores are measured. Student scores are 90%/95% retention targets; latency index is parameter-ratio proxy."
    draw.text((40, 845), foot, fill=(70, 70, 70), font=note_font)

    img.save(OUT, format="PNG")
    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    main()
