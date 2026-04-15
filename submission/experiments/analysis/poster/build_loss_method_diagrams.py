from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT

BG = "#f6f1e8"
PANEL = "#fffdf8"
INK = "#1f2830"
MUTED = "#5d6974"
LINE = "#d8d2c8"
SOFT = "#f0ebe1"
INACTIVE = "#ece7de"
INACTIVE_TEXT = "#8c918f"
BADGE_TEXT = "#14303a"
CALLOUT_BG = "#f7f3ec"

TITLE_FONT = "/Library/Fonts/Arial Unicode.ttf"
ALT_FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"

BADGES = [
    "Score KD",
    "Query Align",
    "Doc Align",
    "Hard Negatives",
    "Margin Guided",
]


@dataclass(frozen=True)
class CardSpec:
    slug: str
    title: str
    accent: str
    accent_soft: str
    formula: str
    active_badges: tuple[str, ...]
    signal: str
    takeaway: str
    note: str


CARDS = [
    CardSpec(
        slug="score_distill",
        title="1. ScoreDistill",
        accent="#c96d1b",
        accent_soft="#f6dfc7",
        formula="L = L_sup + dw·τ²·KL(softmax(S_s/τ) || softmax(S_t/τ))",
        active_badges=("Score KD",),
        signal="Student matches the teacher's full query-to-code similarity distribution.",
        takeaway="Improves ranking imitation, but it never directly forces query or document embeddings into teacher space.",
        note="Best read as score-level supervision only.",
    ),
    CardSpec(
        slug="embed_distill",
        title="2. EmbedDistill",
        accent="#198c83",
        accent_soft="#d7f1ee",
        formula="L = L_sup + dw·τ²·KL(...) + aw·(1/B)Σ||q_s - q_t||₂",
        active_badges=("Score KD", "Query Align"),
        signal="Adds explicit query embedding alignment on top of score matching.",
        takeaway="Student queries move toward teacher query space, but document embeddings still learn only indirectly through retrieval loss.",
        note="Good when query geometry matters more than document geometry.",
    ),
    CardSpec(
        slug="pairdistill",
        title="3. PairDistill / hard_neg_pair",
        accent="#cf4d44",
        accent_soft="#f8d8d5",
        formula="L = L_sup + dw·τ²·KL(...) + pw·BCE((s_s⁺-s_s⁻)/τ, σ((s_t⁺-s_t⁻)/τ))",
        active_badges=("Score KD", "Hard Negatives"),
        signal="Uses the teacher's top-k hardest negatives to train a binary preference: should the positive beat this negative?",
        takeaway="Sharpens local positive-vs-hard-negative separation, but does not reconstruct teacher document geometry.",
        note="Mechanistically different from embedding alignment methods.",
    ),
    CardSpec(
        slug="bimga",
        title="4. BiMGA (Ours)",
        accent="#2e8b57",
        accent_soft="#d8f0df",
        formula="L = L_sup + dw·τ²·KL(...) + aw·(1/B)Σσ(m_i/τ)(||q_s-q_t||₂ + ||d_s-d_t||₂)",
        active_badges=("Score KD", "Query Align", "Doc Align", "Margin Guided"),
        signal="Aligns both query and document embeddings, then weights each sample by teacher confidence.",
        takeaway="Directly addresses the symmetric retrieval gap because both sides of the retriever are pulled toward teacher space.",
        note="m_i = s_t(q_i,d_i⁺) - max_j≠i s_t(q_i,d_j⁻)",
    ),
]


class Fonts:
    def __init__(self) -> None:
        self.title = self._load(48)
        self.subtitle = self._load(24)
        self.card_title = self._load(30)
        self.card_formula = self._load(22)
        self.body = self._load(20)
        self.small = self._load(17)
        self.badge = self._load(18)
        self.footer = self._load(19)

    @staticmethod
    def _load(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        for path in (TITLE_FONT, ALT_FONT):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
        return ImageFont.load_default()


FONTS = Fonts()


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def wrap_text_px(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = word if not current else f"{current} {word}"
        width = draw.textbbox((0, 0), test, font=font)[2]
        if width <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str | None = None, width: int = 1, radius: int = 22) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_text_block(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: str, max_width: int, line_gap: int = 8) -> int:
    x, y = xy
    lines = wrap_text_px(draw, text, font, max_width)
    line_height = draw.textbbox((0, 0), "Ag", font=font)[3] + line_gap
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += line_height
    return y


def draw_badges(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, spec: CardSpec) -> int:
    gap_x = 10
    gap_y = 10
    cursor_x = x
    cursor_y = y
    max_y = y
    for label in BADGES:
        active = label in spec.active_badges
        font = FONTS.badge
        tw = draw.textbbox((0, 0), label, font=font)[2]
        badge_w = tw + 26
        badge_h = 36
        if cursor_x + badge_w > x + width:
            cursor_x = x
            cursor_y += badge_h + gap_y
        fill = spec.accent_soft if active else INACTIVE
        outline = spec.accent if active else LINE
        text_fill = BADGE_TEXT if active else INACTIVE_TEXT
        rounded(draw, (cursor_x, cursor_y, cursor_x + badge_w, cursor_y + badge_h), fill, outline, radius=18)
        draw.text((cursor_x + 13, cursor_y + 8), label, font=font, fill=text_fill)
        cursor_x += badge_w + gap_x
        max_y = max(max_y, cursor_y + badge_h)
    return max_y


def draw_flow(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, spec: CardSpec) -> None:
    left_w = int(w * 0.34)
    mid_w = int(w * 0.26)
    right_w = w - left_w - mid_w - 44
    box_h = h

    base_box = (x, y, x + left_w, y + box_h)
    signal_box = (x + left_w + 22, y, x + left_w + 22 + mid_w, y + box_h)
    result_box = (signal_box[2] + 22, y, signal_box[2] + 22 + right_w, y + box_h)

    rounded(draw, base_box, "#fffaf2", LINE, radius=20)
    rounded(draw, signal_box, spec.accent_soft, spec.accent, radius=20)
    rounded(draw, result_box, "#f8fafb", LINE, radius=20)

    draw.text((base_box[0] + 18, base_box[1] + 16), "Shared base", fill=MUTED, font=FONTS.small)
    draw.text((base_box[0] + 18, base_box[1] + 44), "L_sup", fill=INK, font=FONTS.card_title)
    draw_text_block(
        draw,
        (base_box[0] + 18, base_box[1] + 92),
        "One-hot supervised retrieval loss over the in-batch score matrix.",
        FONTS.body,
        INK,
        left_w - 36,
    )

    draw.text((signal_box[0] + 18, signal_box[1] + 16), "Extra teacher signal", fill=spec.accent, font=FONTS.small)
    draw_text_block(
        draw,
        (signal_box[0] + 18, signal_box[1] + 50),
        spec.signal,
        FONTS.body,
        INK,
        mid_w - 36,
    )

    draw.text((result_box[0] + 18, result_box[1] + 16), "Poster takeaway", fill=MUTED, font=FONTS.small)
    draw_text_block(
        draw,
        (result_box[0] + 18, result_box[1] + 50),
        spec.takeaway,
        FONTS.body,
        INK,
        right_w - 36,
    )

    cy = y + box_h // 2
    ax1 = base_box[2] + 6
    ax2 = signal_box[0] - 6
    ax3 = signal_box[2] + 6
    ax4 = result_box[0] - 6
    draw.line((ax1, cy, ax2, cy), fill=spec.accent, width=5)
    draw.polygon([(ax2, cy), (ax2 - 16, cy - 10), (ax2 - 16, cy + 10)], fill=spec.accent)
    draw.line((ax3, cy, ax4, cy), fill=spec.accent, width=5)
    draw.polygon([(ax4, cy), (ax4 - 16, cy - 10), (ax4 - 16, cy + 10)], fill=spec.accent)


def draw_card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], spec: CardSpec) -> None:
    x1, y1, x2, y2 = box
    rounded(draw, box, PANEL, outline=LINE, width=2, radius=28)
    rounded(draw, (x1, y1, x2, y1 + 56), spec.accent_soft, outline=None, radius=28)
    draw.text((x1 + 26, y1 + 16), spec.title, fill=INK, font=FONTS.card_title)

    formula_box = (x1 + 26, y1 + 84, x2 - 26, y1 + 150)
    rounded(draw, formula_box, SOFT, outline=LINE, radius=18)
    draw.text((formula_box[0] + 16, formula_box[1] + 20), spec.formula, fill=INK, font=FONTS.card_formula)

    badges_bottom = draw_badges(draw, x1 + 26, y1 + 172, (x2 - x1) - 52, spec)
    flow_top = badges_bottom + 20
    flow_h = 220
    draw_flow(draw, x1 + 26, flow_top, (x2 - x1) - 52, flow_h, spec)

    note_box = (x1 + 26, flow_top + flow_h + 20, x2 - 26, y2 - 24)
    rounded(draw, note_box, CALLOUT_BG, outline=LINE, radius=18)
    draw.text((note_box[0] + 16, note_box[1] + 14), "Interpretation", fill=spec.accent, font=FONTS.small)
    draw_text_block(draw, (note_box[0] + 16, note_box[1] + 44), spec.note, FONTS.body, INK, note_box[2] - note_box[0] - 32)


def build_png() -> Path:
    width, height = 2400, 1700
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)

    draw.ellipse((-180, -120, 460, 520), fill="#efe2d2")
    draw.ellipse((1840, -160, 2540, 540), fill="#dcefea")
    draw.ellipse((1880, 1240, 2520, 1880), fill="#e6efe2")

    draw.text((120, 72), "Knowledge Distillation Objectives Used in the TACO Experiments", fill=INK, font=FONTS.title)
    subtitle = "Each method starts from the same supervised retrieval objective. The structural difference is which extra teacher signal is injected into training."
    draw_text_block(draw, (120, 146), subtitle, FONTS.subtitle, MUTED, 2000)

    cards = [
        (120, 250, 1160, 920),
        (1240, 250, 2280, 920),
        (120, 960, 1160, 1630),
        (1240, 960, 2280, 1630),
    ]
    for box, spec in zip(cards, CARDS):
        draw_card(draw, box, spec)

    footer_y = 1644
    footer = "Visual summary: ScoreDistill learns ranking scores; EmbedDistill additionally aligns queries; PairDistill sharpens positive-vs-hard-negative preference; BiMGA is the only method here that aligns both query and document embeddings while weighting by teacher confidence."
    draw_text_block(draw, (120, footer_y), footer, FONTS.footer, MUTED, 2160, line_gap=6)

    out = OUT / "kd_loss_methods_poster.png"
    img.save(out)
    return out


def svg_text_lines(text: str, width_chars: int) -> list[str]:
    return textwrap.wrap(text, width=width_chars, break_long_words=False, break_on_hyphens=False) or [""]


def svg_round_rect(x: int, y: int, w: int, h: int, fill: str, stroke: str | None = None, stroke_width: int = 1, rx: int = 18) -> str:
    attrs = [f'x="{x}"', f'y="{y}"', f'width="{w}"', f'height="{h}"', f'rx="{rx}"', f'fill="{fill}"']
    if stroke:
        attrs.append(f'stroke="{stroke}"')
        attrs.append(f'stroke-width="{stroke_width}"')
    return f"<rect {' '.join(attrs)} />"


def svg_text(x: int, y: int, text: str, size: int, fill: str, weight: str = "400") -> str:
    return f'<text x="{x}" y="{y}" font-family="Arial Unicode MS, Arial, sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}">{escape(text)}</text>'


def svg_multiline(x: int, y: int, text: str, size: int, fill: str, width_chars: int, line_step: int, weight: str = "400") -> str:
    parts = [f'<text x="{x}" y="{y}" font-family="Arial Unicode MS, Arial, sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}">']
    for idx, line in enumerate(svg_text_lines(text, width_chars)):
        dy = 0 if idx == 0 else line_step
        parts.append(f'<tspan x="{x}" dy="{dy}">{escape(line)}</tspan>')
    parts.append("</text>")
    return "".join(parts)


def svg_badges(x: int, y: int, width: int, spec: CardSpec) -> tuple[str, int]:
    pieces: list[str] = []
    cursor_x = x
    cursor_y = y
    max_y = y
    for label in BADGES:
        active = label in spec.active_badges
        badge_w = max(104, len(label) * 11 + 28)
        badge_h = 36
        if cursor_x + badge_w > x + width:
            cursor_x = x
            cursor_y += badge_h + 10
        pieces.append(svg_round_rect(cursor_x, cursor_y, badge_w, badge_h, spec.accent_soft if active else INACTIVE, spec.accent if active else LINE, rx=18))
        pieces.append(svg_text(cursor_x + 14, cursor_y + 24, label, 18, BADGE_TEXT if active else INACTIVE_TEXT, weight="700"))
        cursor_x += badge_w + 10
        max_y = max(max_y, cursor_y + badge_h)
    return "".join(pieces), max_y


def svg_flow(x: int, y: int, w: int, h: int, spec: CardSpec) -> str:
    left_w = int(w * 0.34)
    mid_w = int(w * 0.26)
    right_w = w - left_w - mid_w - 44
    lx, mx, rx = x, x + left_w + 22, x + left_w + 22 + mid_w + 22
    pieces = [
        svg_round_rect(lx, y, left_w, h, "#fffaf2", LINE, rx=20),
        svg_round_rect(mx, y, mid_w, h, spec.accent_soft, spec.accent, rx=20),
        svg_round_rect(rx, y, right_w, h, "#f8fafb", LINE, rx=20),
        svg_text(lx + 18, y + 26, "Shared base", 17, MUTED, weight="700"),
        svg_text(lx + 18, y + 60, "L_sup", 30, INK, weight="700"),
        svg_multiline(lx + 18, y + 98, "One-hot supervised retrieval loss over the in-batch score matrix.", 20, INK, 24, 26),
        svg_text(mx + 18, y + 26, "Extra teacher signal", 17, spec.accent, weight="700"),
        svg_multiline(mx + 18, y + 58, spec.signal, 20, INK, 18, 26),
        svg_text(rx + 18, y + 26, "Poster takeaway", 17, MUTED, weight="700"),
        svg_multiline(rx + 18, y + 58, spec.takeaway, 20, INK, 27, 26),
    ]
    cy = y + h // 2
    pieces.append(f'<line x1="{lx + left_w + 6}" y1="{cy}" x2="{mx - 6}" y2="{cy}" stroke="{spec.accent}" stroke-width="5" marker-end="url(#arrow-{spec.slug})" />')
    pieces.append(f'<line x1="{mx + mid_w + 6}" y1="{cy}" x2="{rx - 6}" y2="{cy}" stroke="{spec.accent}" stroke-width="5" marker-end="url(#arrow-{spec.slug})" />')
    return "".join(pieces)


def svg_card(x1: int, y1: int, x2: int, y2: int, spec: CardSpec) -> str:
    pieces = [
        svg_round_rect(x1, y1, x2 - x1, y2 - y1, PANEL, LINE, stroke_width=2, rx=28),
        svg_round_rect(x1, y1, x2 - x1, 56, spec.accent_soft, rx=28),
        svg_text(x1 + 26, y1 + 37, spec.title, 30, INK, weight="700"),
        svg_round_rect(x1 + 26, y1 + 84, x2 - x1 - 52, 66, SOFT, LINE, rx=18),
        svg_text(x1 + 42, y1 + 126, spec.formula, 22, INK),
    ]
    badges_markup, badges_bottom = svg_badges(x1 + 26, y1 + 172, x2 - x1 - 52, spec)
    pieces.append(badges_markup)
    flow_top = badges_bottom + 20
    pieces.append(svg_flow(x1 + 26, flow_top, x2 - x1 - 52, 220, spec))
    note_y = flow_top + 210
    pieces.append(svg_round_rect(x1 + 26, note_y, x2 - x1 - 52, y2 - note_y - 24, CALLOUT_BG, LINE, rx=18))
    pieces.append(svg_text(x1 + 42, note_y + 32, "Interpretation", 17, spec.accent, weight="700"))
    pieces.append(svg_multiline(x1 + 42, note_y + 64, spec.note, 20, INK, 56, 26))
    return "".join(pieces)


def build_svg() -> Path:
    width, height = 2400, 1700
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<linearGradient id="bg-grad" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#fbf8f1"/><stop offset="100%" stop-color="#f6f1e8"/></linearGradient>',
    ]
    for spec in CARDS:
        pieces.append(
            f'<marker id="arrow-{spec.slug}" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="{spec.accent}" /></marker>'
        )
    pieces.append("</defs>")
    pieces.extend(
        [
            f'<rect width="{width}" height="{height}" fill="url(#bg-grad)" />',
            '<circle cx="130" cy="130" r="260" fill="#efe2d2" />',
            '<circle cx="2220" cy="120" r="250" fill="#dcefea" />',
            '<circle cx="2280" cy="1540" r="240" fill="#e6efe2" />',
            svg_text(120, 110, "Knowledge Distillation Objectives Used in the TACO Experiments", 48, INK, weight="700"),
            svg_multiline(120, 158, "Each method starts from the same supervised retrieval objective. The structural difference is which extra teacher signal is injected into training.", 24, MUTED, 112, 30),
        ]
    )

    cards = [
        (120, 250, 1160, 920),
        (1240, 250, 2280, 920),
        (120, 960, 1160, 1630),
        (1240, 960, 2280, 1630),
    ]
    for box, spec in zip(cards, CARDS):
        pieces.append(svg_card(*box, spec))

    footer = "Visual summary: ScoreDistill learns ranking scores; EmbedDistill additionally aligns queries; PairDistill sharpens positive-vs-hard-negative preference; BiMGA is the only method here that aligns both query and document embeddings while weighting by teacher confidence."
    pieces.append(svg_multiline(120, 1664, footer, 19, MUTED, 180, 24))
    pieces.append("</svg>")

    out = OUT / "kd_loss_methods_poster.svg"
    out.write_text("\n".join(pieces), encoding="utf-8")
    return out


def crop_card(source: Image.Image, box: tuple[int, int, int, int], out_name: str) -> Path:
    pad = 0
    crop = source.crop((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad))
    out = OUT / out_name
    crop.save(out)
    return out


def build_individual_pngs(master_png: Path) -> list[Path]:
    source = Image.open(master_png)
    boxes = {
        "score_distill_diagram.png": (120, 250, 1160, 920),
        "embed_distill_diagram.png": (1240, 250, 2280, 920),
        "pairdistill_diagram.png": (120, 960, 1160, 1630),
        "bimga_diagram.png": (1240, 960, 2280, 1630),
    }
    return [crop_card(source, box, name) for name, box in boxes.items()]


def build_individual_svgs(master_svg: Path) -> list[Path]:
    outputs: list[Path] = []
    card_w, card_h = 1040, 670
    positions = {
        "score_distill_diagram.svg": CARDS[0],
        "embed_distill_diagram.svg": CARDS[1],
        "pairdistill_diagram.svg": CARDS[2],
        "bimga_diagram.svg": CARDS[3],
    }
    for name, spec in positions.items():
        pieces = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{card_w}" height="{card_h}" viewBox="0 0 {card_w} {card_h}">',
            "<defs>",
            f'<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="{spec.accent}" /></marker>',
            "</defs>",
            f'<rect width="{card_w}" height="{card_h}" fill="{BG}" />',
            svg_card(0, 0, card_w, card_h, spec),
            "</svg>",
        ]
        out = OUT / name
        out.write_text("\n".join(pieces), encoding="utf-8")
        outputs.append(out)
    return outputs


if __name__ == "__main__":
    svg_path = build_svg()
    png_path = build_png()
    pngs = build_individual_pngs(png_path)
    svgs = build_individual_svgs(svg_path)
    print(svg_path)
    print(png_path)
    for path in pngs + svgs:
        print(path)
