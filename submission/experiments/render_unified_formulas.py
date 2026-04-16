"""Render a single unified KD formula reference block as PNG via KaTeX + Playwright."""
import subprocess, sys, textwrap
from pathlib import Path

OUT = Path("formula_screenshots")
OUT.mkdir(exist_ok=True)


def build_unified_html() -> str:
    return textwrap.dedent(r"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <link rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: "Times New Roman", "Computer Modern", serif;
            background: white;
            margin: 0;
            padding: 16px 32px 18px 32px;
            width: 620px;
            color: #111;
            font-size: 12px;
            line-height: 1.4;
        }
        .top-title {
            text-align: center;
            font-size: 15px;
            font-weight: bold;
            margin: 0 0 8px 0;
        }
        .common {
            text-align: center;
            margin-bottom: 10px;
        }
        .common .defs {
            font-size: 11.5px;
            color: #333;
            margin-bottom: 4px;
        }
        .common .f {
            text-align: center;
            margin: 3px 0;
            line-height: 1.8;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        .cell {
            border: 1px solid #bbb;
            border-radius: 4px;
            padding: 8px 12px 10px 12px;
            overflow: hidden;
        }
        .cell h3 {
            text-align: center;
            font-size: 12.5px;
            font-weight: bold;
            margin: 0 0 4px 0;
        }
        .cell .f {
            text-align: center;
            margin: 3px 0;
            line-height: 1.8;
            font-size: 11.5px;
        }
        .cell .desc {
            margin: 4px 0 0 0;
            font-size: 10.5px;
            color: #333;
            text-align: justify;
        }
        .cell .note {
            margin: 1px 0 0 0;
            font-size: 9.5px;
            color: #555;
            text-align: center;
        }
        .katex { font-size: 1.0em; }
    </style>
    </head>
    <body>

    <div class="top-title">Knowledge Distillation Loss Functions</div>

    <div class="common">
        <div class="defs">
            $\mathbf{S}_q, \mathbf{S}_d$: student embeddings &nbsp;&bull;&nbsp;
            $\mathbf{T}_q, \mathbf{T}_d$: teacher embeddings &nbsp;&bull;&nbsp;
            $s^s_{ij}, s^t_{ij}$: similarity scores &nbsp;&bull;&nbsp;
            $B$: batch size &nbsp;&bull;&nbsp;
            $\tau_d$: distillation temperature &nbsp;&bull;&nbsp;
            $\sigma$: sigmoid
        </div>
        <div class="f">$$\mathcal{L}_{\text{supervised}} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(s_{ii}/\tau)}{\sum_{j}\exp(s_{ij}/\tau)} \qquad\qquad \mathbf{S}_s = \mathbf{S}_q \cdot \mathbf{T}_d^\top \qquad \mathbf{S}_t = \mathbf{T}_q \cdot \mathbf{T}_d^\top$$</div>
    </div>

    <div class="grid">

    <div class="cell">
        <h3>1. ScoreDistill (Pure Score KD)</h3>
        <div class="f">$$\mathcal{L} = \mathcal{L}_{\text{sup}} + \lambda_d \cdot \tau_d^{\,2} \;\mathrm{KL}\!\left(\mathrm{softmax}\!\left(\tfrac{\mathbf{S}_s}{\tau_d}\right) \;\Big\|\; \mathrm{softmax}\!\left(\tfrac{\mathbf{S}_t}{\tau_d}\right)\right)$$</div>
        <div class="desc">Transfers knowledge by training the student to match the teacher's output probability distribution (soft labels), capturing nuanced class relationships.</div>
    </div>

    <div class="cell">
        <h3>2. EmbedDistill (Score KD + Query Alignment)</h3>
        <div class="f">$$\mathcal{L} = \mathcal{L}_{\text{sup}} + \lambda_d \cdot \tau_d^{\,2} \;\mathrm{KL}\!\Big(\cdots\Big) + \lambda_a \cdot \frac{1}{B}\sum_{i=1}^{B}\left\|\mathbf{S}_{q_i} - \mathbf{T}_{q_i}\right\|_2$$</div>
        <div class="desc">Aligns the internal feature representations of the student with those of the teacher, encouraging similar understanding at intermediate layers.</div>
    </div>

    <div class="cell">
        <h3>3. Hard-Negative PairDistill (Pairwise Preference)</h3>
        <div class="f">$$\mathcal{L} = \mathcal{L}_{\text{sup}} + \lambda_d \cdot \tau_d^{\,2} \;\mathrm{KL}\!\Big(\cdots\Big) + \lambda_p \cdot \mathcal{L}_{\text{pair}}$$</div>
        <div class="f">$$\mathcal{L}_{\text{pair}} = \frac{1}{|M|}\!\sum_{(i,j)\in M}\! \mathrm{BCE}\!\left(\sigma\!\left(\tfrac{s^s_{ii} - s^s_{ij}}{\tau_d}\right), \sigma\!\left(\tfrac{s^t_{ii} - s^t_{ij}}{\tau_d}\right)\right)$$</div>
        <div class="note">$M = \{(i,j) : j \in \text{top-}k\text{ hardest negatives}\}$</div>
        <div class="desc">Preserves relational knowledge by teaching the student to mimic pairwise relationships between data samples learned by the teacher.</div>
    </div>

    <div class="cell">
        <h3>4. BiMGA (Bidirectional Margin-Guided Alignment)</h3>
        <div class="f">$$\mathcal{L} = \mathcal{L}_{\text{sup}} + \lambda_d \cdot \tau_d^{\,2} \;\mathrm{KL}\!\Big(\cdots\Big) + \lambda_a \cdot \mathcal{L}_{\text{align}}$$</div>
        <div class="f">$$\mathcal{L}_{\text{align}} = \frac{1}{B}\sum_{i=1}^{B} w_i \!\left(\left\|\mathbf{S}_{q_i}\! -\! \mathbf{T}_{q_i}\right\|_2 +\! \left\|\mathbf{S}_{d_i}\! -\! \mathbf{T}_{d_i}\right\|_2\right)$$</div>
        <div class="f">$$m_i = s^t_{ii} - \max_{j \neq i}\, s^t_{ij} \qquad w_i = \sigma\!\left(\frac{m_i}{\tau_d}\right)$$</div>
        <div class="desc">Improves student learning by aligning both fine and coarse-grained representations bidirectionally, weighted by teacher confidence margins.</div>
    </div>

    </div>

    <script>
        renderMathInElement(document.body, {
            delimiters: [
                {left: "$$", right: "$$", display: true},
                {left: "$", right: "$", display: false},
            ],
            throwOnError: false,
        });
    </script>
    </body>
    </html>
    """).strip()


def render():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Installing playwright...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright"], stdout=subprocess.DEVNULL)
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"], stdout=subprocess.DEVNULL)
        from playwright.sync_api import sync_playwright

    html_dir = OUT / "html"
    html_dir.mkdir(exist_ok=True)

    html_path = html_dir / "unified_formulas.html"
    html_path.write_text(build_unified_html(), encoding="utf-8")

    print("Rendering unified formula reference...")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(device_scale_factor=3)

        page.goto(html_path.resolve().as_uri())
        page.wait_for_timeout(1500)

        body = page.locator("body")
        out_path = OUT / "unified_kd_formulas.png"
        body.screenshot(path=str(out_path))
        print(f"  Saved to {out_path}")

        browser.close()
    print("Done.")


if __name__ == "__main__":
    render()
