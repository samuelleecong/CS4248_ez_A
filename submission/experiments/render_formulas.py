"""Render KD loss formulas as individual LaTeX PNGs via KaTeX + Playwright."""
import subprocess, json, sys, textwrap
from pathlib import Path

OUT = Path("formula_screenshots")
OUT.mkdir(exist_ok=True)

# Each entry: (filename, title, list of LaTeX lines)
FORMULAS = [
    (
        "supervised",
        "Supervised Baseline",
        [
            r"\mathbf{S} = \mathbf{S}_q \cdot \mathbf{S}_d^\top",
            r"\mathcal{L} = \mathrm{CE}\!\left(\mathrm{softmax}\!\left(\frac{\mathbf{S}}{\tau}\right),\; \mathbf{y}\right) = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(s_{ii}/\tau)}{\sum_{j}\exp(s_{ij}/\tau)}",
        ],
    ),
    (
        "score_distill",
        "Score Distillation",
        [
            r"\mathbf{S}_s = \mathbf{S}_q \cdot \mathbf{T}_d^\top \qquad \mathbf{S}_t = \mathbf{T}_q \cdot \mathbf{T}_d^\top",
            r"\mathcal{L}_{KL} = \mathrm{KL}\!\left(\mathrm{softmax}\!\left(\frac{\mathbf{S}_s}{\tau_d}\right) \;\Big\|\; \mathrm{softmax}\!\left(\frac{\mathbf{S}_t}{\tau_d}\right)\right) \cdot \tau_d^{\,2}",
            r"\mathcal{L} = \mathcal{L}_{CE} + \lambda_d \,\mathcal{L}_{KL}",
        ],
    ),
    (
        "embed_distill",
        "Embedding Distillation",
        [
            r"\mathcal{L}_{KL} = \mathrm{KL}\!\left(\mathrm{softmax}\!\left(\frac{\mathbf{S}_s}{\tau_d}\right) \;\Big\|\; \mathrm{softmax}\!\left(\frac{\mathbf{S}_t}{\tau_d}\right)\right) \cdot \tau_d^{\,2}",
            r"\mathcal{L}_{align} = \frac{1}{B}\sum_{i=1}^{B} \left\|\mathbf{S}_{q_i} - \mathbf{T}_{q_i}\right\|_2",
            r"\mathcal{L} = \mathcal{L}_{CE} + \lambda_d \,\mathcal{L}_{KL} + \lambda_a \,\mathcal{L}_{align}",
        ],
    ),
    (
        "hard_negative_pair_distill",
        "Hard-Negative Pairwise Distillation",
        [
            r"\mathcal{L}_{KL} = \mathrm{KL}\!\left(\mathrm{softmax}\!\left(\frac{\mathbf{S}_s}{\tau_d}\right) \;\Big\|\; \mathrm{softmax}\!\left(\frac{\mathbf{S}_t}{\tau_d}\right)\right) \cdot \tau_d^{\,2}",
            r"\mathcal{L}_{pair} = \frac{1}{|M|}\sum_{(i,j)\in M} \mathrm{BCE}\!\left(\sigma\!\left(\frac{s^s_{ii} - s^s_{ij}}{\tau_d}\right),\; \sigma\!\left(\frac{s^t_{ii} - s^t_{ij}}{\tau_d}\right)\right)",
            r"M = \{(i,\,j) : j \in \text{top-}k\text{ hardest negatives for query } i\}",
            r"\mathcal{L} = \mathcal{L}_{CE} + \lambda_d \,\mathcal{L}_{KL} + \lambda_p \,\mathcal{L}_{pair}",
        ],
    ),
    (
        "qed_align",
        "Query Embedding Alignment (QED)",
        [
            r"\mathcal{L}_{align} = \frac{1}{B}\sum_{i=1}^{B} \left\|\mathbf{S}_{q_i} - \mathbf{T}_{q_i}\right\|_2",
            r"\mathcal{L} = \mathcal{L}_{CE} + \lambda_a \,\mathcal{L}_{align}",
        ],
    ),
    (
        "margin_mse",
        "Margin MSE",
        [
            r"m^s_{ij} = s^s_{ii} - s^s_{ij} \qquad m^t_{ij} = s^t_{ii} - s^t_{ij}",
            r"\mathcal{L}_{mse} = \frac{1}{|M|}\sum_{\substack{i,j \\ i \neq j}} \left(m^s_{ij} - m^t_{ij}\right)^2",
            r"\mathcal{L} = \mathcal{L}_{CE} + \lambda_p \,\mathcal{L}_{mse}",
        ],
    ),
    (
        "bimga",
        "BiMGA -- Bidirectional Margin-Guided Alignment",
        [
            r"\mathcal{L}_{KL} = \mathrm{KL}\!\left(\mathrm{softmax}\!\left(\frac{\mathbf{S}_s}{\tau_d}\right) \;\Big\|\; \mathrm{softmax}\!\left(\frac{\mathbf{S}_t}{\tau_d}\right)\right) \cdot \tau_d^{\,2}",
            r"m_i = s^t_{ii} - \max_{j \neq i}\, s^t_{ij} \qquad w_i = \sigma\!\left(\frac{m_i}{\tau_d}\right)",
            r"\mathcal{L}_{align} = \frac{1}{B}\sum_{i=1}^{B} w_i \left(\left\|\mathbf{S}_{q_i} - \mathbf{T}_{q_i}\right\|_2 + \left\|\mathbf{S}_{d_i} - \mathbf{T}_{d_i}\right\|_2\right)",
            r"\mathcal{L} = \mathcal{L}_{CE} + \lambda_d \,\mathcal{L}_{KL} + \lambda_a \,\mathcal{L}_{align}",
        ],
    ),
    (
        "bimga_uniform",
        "BiMGA Uniform -- No Margin Weighting",
        [
            r"\mathcal{L}_{KL} = \mathrm{KL}\!\left(\mathrm{softmax}\!\left(\frac{\mathbf{S}_s}{\tau_d}\right) \;\Big\|\; \mathrm{softmax}\!\left(\frac{\mathbf{S}_t}{\tau_d}\right)\right) \cdot \tau_d^{\,2}",
            r"\mathcal{L}_{align} = \frac{1}{B}\sum_{i=1}^{B} \left(\left\|\mathbf{S}_{q_i} - \mathbf{T}_{q_i}\right\|_2 + \left\|\mathbf{S}_{d_i} - \mathbf{T}_{d_i}\right\|_2\right)",
            r"\mathcal{L} = \mathcal{L}_{CE} + \lambda_d \,\mathcal{L}_{KL} + \lambda_a \,\mathcal{L}_{align}",
        ],
    ),
    (
        "adam_lite",
        "ADAM-Lite -- Dark Example Distillation",
        [
            r"\mathcal{L}_{KL} = \mathrm{KL}\!\left(\mathrm{softmax}\!\left(\frac{\mathbf{S}_s}{\tau_d}\right) \;\Big\|\; \mathrm{softmax}\!\left(\frac{\mathbf{S}_t}{\tau_d}\right)\right) \cdot \tau_d^{\,2}",
            r"\tilde{\mathbf{d}}_k = \mathrm{normalize}\!\left(\alpha\,\mathbf{d}^{+} + (1-\alpha)\,\mathbf{d}^{-}_k\right)",
            r"c_i = \sigma\!\left(\frac{s^t_{i,+} - \overline{s^t_{i,\mathrm{dark}}}}{\tau_d}\right)",
            r"\mathcal{L}_{dark} = \frac{1}{B}\sum_{i=1}^{B} c_i \cdot \mathrm{KL}\!\left(\mathrm{softmax}\!\left(\frac{\tilde{\mathbf{S}}^s_i}{\tau_d}\right) \;\Big\|\; \mathrm{softmax}\!\left(\frac{\tilde{\mathbf{S}}^t_i}{\tau_d}\right)\right) \cdot \tau_d^{\,2}",
            r"\mathcal{L} = \mathcal{L}_{CE} + \lambda_d \left(\mathcal{L}_{KL} + \mathcal{L}_{dark}\right)",
        ],
    ),
]


def build_html(title: str, lines: list[str]) -> str:
    """Build a self-contained HTML page that renders LaTeX via KaTeX CDN."""
    formula_divs = "\n".join(
        f'        <div class="formula">$${tex}$$</div>' for tex in lines
    )
    return textwrap.dedent(f"""\
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <link rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
    <style>
        body {{
            font-family: "Computer Modern", "Latin Modern Math", "STIX Two Math", serif;
            background: white;
            margin: 0;
            padding: 24px 36px;
            width: 680px;
        }}
        h2 {{
            margin: 0 0 14px 0;
            font-size: 20px;
            text-align: center;
            color: #1a1a1a;
        }}
        .formula {{
            margin: 8px 0;
            font-size: 16px;
            text-align: center;
            line-height: 2.0;
        }}
        .katex {{ font-size: 1.1em; }}
    </style>
    </head>
    <body>
        <h2>{title}</h2>
{formula_divs}
    <script>
        renderMathInElement(document.body, {{
            delimiters: [
                {{left: "$$", right: "$$", display: true}},
            ],
            throwOnError: false,
        }});
    </script>
    </body>
    </html>
    """)


def render_all():
    """Write HTML files, then screenshot each with Playwright."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Installing playwright...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright"], stdout=subprocess.DEVNULL)
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"], stdout=subprocess.DEVNULL)
        from playwright.sync_api import sync_playwright

    html_dir = OUT / "html"
    html_dir.mkdir(exist_ok=True)

    print("Rendering formula screenshots...")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(device_scale_factor=3)

        for fname, title, lines in FORMULAS:
            html_path = html_dir / f"{fname}.html"
            html_path.write_text(build_html(title, lines), encoding="utf-8")

            page.goto(html_path.resolve().as_uri())
            page.wait_for_timeout(800)  # let KaTeX render

            body = page.locator("body")
            out_path = OUT / f"{fname}.png"
            body.screenshot(path=str(out_path))
            print(f"  {out_path}")

        browser.close()
    print("Done.")


if __name__ == "__main__":
    render_all()
