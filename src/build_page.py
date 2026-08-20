"""Render BLOG.md into a standalone GitHub Pages site at docs/index.html."""

from __future__ import annotations

import pathlib
import re

import mistune

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

CSS = """
:root {
  --bg: #ffffff; --fg: #1a1d21; --muted: #5c6570; --rule: #e3e6ea;
  --accent: #2f6fb0; --code-bg: #f5f7f9; --table-head: #f0f3f6;
  --quote-bg: #f7f9fb; --shadow: rgba(0,0,0,.07);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14171a; --fg: #e6e9ec; --muted: #9aa4ae; --rule: #2b3138;
    --accent: #6aa9e0; --code-bg: #1c2126; --table-head: #1e242a;
    --quote-bg: #1a1f24; --shadow: rgba(0,0,0,.4);
  }
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 17px/1.7 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
}
.wrap { max-width: 820px; margin: 0 auto; padding: 56px 22px 96px; }
h1 { font-size: 2.35rem; line-height: 1.18; letter-spacing: -.022em; margin: 0 0 .3em; }
h2 { font-size: 1.5rem; letter-spacing: -.014em; margin: 2.6em 0 .7em;
     padding-top: 1.1em; border-top: 1px solid var(--rule); }
h3 { font-size: 1.16rem; margin: 2em 0 .6em; }
p, li { color: var(--fg); }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
em.sub { display: block; color: var(--muted); font-size: 1.06rem; margin: 0 0 1.6em; }
hr { border: 0; border-top: 1px solid var(--rule); margin: 2.4em 0; }
blockquote {
  margin: 1.5em 0; padding: .9em 1.2em; background: var(--quote-bg);
  border-left: 3px solid var(--accent); border-radius: 0 6px 6px 0; color: var(--fg);
}
blockquote p { margin: .4em 0; }
blockquote.pull {
  border-left: 0; background: none; text-align: center;
  margin: 2.2em auto; padding: 1.1em 0; max-width: 34ch;
  border-top: 2px solid var(--accent); border-bottom: 2px solid var(--accent);
}
blockquote.pull p {
  font-size: 1.42rem; line-height: 1.35; font-weight: 600;
  letter-spacing: -.014em; margin: 0; color: var(--fg);
}
blockquote.pull strong { font-weight: 600; }
@media (max-width: 620px) { blockquote.pull p { font-size: 1.18rem; } }
code {
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
  font-size: .88em; background: var(--code-bg); padding: .16em .38em; border-radius: 4px;
}
pre {
  background: var(--code-bg); padding: 1em 1.15em; border-radius: 8px;
  overflow-x: auto; border: 1px solid var(--rule); line-height: 1.5;
}
pre code { background: none; padding: 0; font-size: .855em; }
.tablewrap { overflow-x: auto; margin: 1.5em 0; }
table { border-collapse: collapse; width: 100%; font-size: .93rem; }
th, td { padding: .55em .7em; border-bottom: 1px solid var(--rule); text-align: left;
         white-space: nowrap; }
th { background: var(--table-head); font-weight: 600; }
td:not(:first-child), th:not(:first-child) { text-align: right; }
tbody tr:hover { background: var(--quote-bg); }
figure { margin: 2em 0; }
img { max-width: 100%; height: auto; display: block; border-radius: 8px;
      box-shadow: 0 1px 10px var(--shadow); }
sup { font-size: .75em; }
.byline { font-size: 1rem; font-weight: 600; color: var(--fg); margin: 0 0 .35em; }
.meta { color: var(--muted); font-size: .92rem; margin-bottom: 2.4em; }
.footer { margin-top: 4em; padding-top: 1.4em; border-top: 1px solid var(--rule);
          color: var(--muted); font-size: .9rem; }
@media (max-width: 620px) {
  .wrap { padding: 34px 16px 70px; }
  h1 { font-size: 1.8rem; }
  body { font-size: 16px; }
}
"""


def build() -> pathlib.Path:
    md = (ROOT / "BLOG.md").read_text()

    # Pull the H1 and the italic subtitle out for the header block.
    lines = md.splitlines()
    title = lines[0].lstrip("# ").strip()
    body_md = "\n".join(lines[1:])

    html = mistune.html(body_md)

    # Subtitle: first <em> paragraph becomes the standfirst.
    html = re.sub(r"<p><em>(.*?)</em></p>", r'<em class="sub">\1</em>', html, count=1,
                  flags=re.S)
    # Tables need a scroll container on narrow screens.
    html = html.replace("<table>", '<div class="tablewrap"><table>')
    html = html.replace("</table>", "</table></div>")
    # Wrap images in figures.
    html = re.sub(r"<p>(<img[^>]*>)</p>", r"<figure>\1</figure>", html)
    # A short blockquote whose entire content is bold is a pull quote. Longer
    # bold quotes (e.g. the research question) keep the ordinary treatment, since
    # the centred pull-quote box only reads well for a single punchy line.
    # Renders as an ordinary bold quote wherever markdown is read plainly.
    def _pull(m):
        inner = m.group(1)
        text = re.sub(r"<[^>]+>", "", inner)
        if len(text) > 110:
            return m.group(0)
        return f'<blockquote class="pull"><p>{inner}</p></blockquote>'

    html = re.sub(
        r"<blockquote>\s*<p><strong>(.*?)</strong></p>\s*</blockquote>",
        _pull, html, flags=re.S)

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="We replaced a calibrated peer-sycophancy prior with a plain critical-reasoning prompt. Sycophancy fell by up to 59%. Accuracy did not follow.">
<meta property="og:title" content="{title}">
<meta property="og:description" content="We replaced a calibrated peer-sycophancy prior with a plain critical-reasoning prompt. Sycophancy fell by up to 59%. Accuracy did not follow.">
<meta property="og:type" content="article">
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<h1>{title}</h1>
<div class="byline">Xiaoping Zhou</div>
<div class="meta">Intrigued by <a href="https://arxiv.org/abs/2604.02668">arXiv:2604.02668</a>
&middot; <a href="https://github.com/zxp567/sycophancy-vs-accuracy">code &amp; data</a></div>
{html}
<div class="footer">
All model responses and per-round discussion logs are cached in the repository, so
every table and figure here regenerates offline with no API calls.
</div>
</div>
</body>
</html>
"""
    DOCS.mkdir(exist_ok=True)
    (DOCS / "index.html").write_text(page)
    # Pages needs the figures alongside the HTML.
    figdir = DOCS / "figures"
    figdir.mkdir(exist_ok=True)
    for png in (ROOT / "figures").glob("*.png"):
        (figdir / png.name).write_bytes(png.read_bytes())
    (DOCS / ".nojekyll").write_text("")
    # The paper is not published from this repository.
    return DOCS / "index.html"


if __name__ == "__main__":
    out = build()
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")
