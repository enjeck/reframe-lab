"""Build the local web page and the AWS page from web/page.template.html.

Lines between @@LOCAL and @@END appear only in the local app (embedded in
src/reframelab/webapp.py). Lines between @@CLOUD and @@END appear only in
deploy/aws/api/index.html. Run from the repository root:

    python scripts/build_web_pages.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "web" / "page.template.html"
WEBAPP = ROOT / "src" / "reframelab" / "webapp.py"
CLOUD_PAGE = ROOT / "deploy" / "aws" / "api" / "index.html"


def render(text: str, cloud: bool) -> str:
    drop = "@@LOCAL" if cloud else "@@CLOUD"
    lines: list[str] = []
    block: str | None = None
    for line in text.split("\n"):
        tag = line.strip()
        if tag in ("@@LOCAL", "@@CLOUD"):
            if block is not None:
                raise ValueError("Nested @@ blocks are not supported")
            block = tag
            continue
        if tag == "@@END":
            if block is None:
                raise ValueError("@@END without an opening block")
            block = None
            continue
        if block != drop:
            lines.append(line)
    if block is not None:
        raise ValueError(f"Unclosed {block} block")
    page = "\n".join(lines).replace("@@CLOUDFLAG", "true" if cloud else "false")
    if "@@" in page:
        raise ValueError("Unknown @@ marker left in the page")
    return page


def main() -> None:
    text = TEMPLATE.read_text(encoding="utf-8")
    local = render(text, cloud=False).rstrip("\n")
    if '"""' in local:
        raise ValueError("The page may not contain triple quotes")
    source = WEBAPP.read_text(encoding="utf-8")
    start = source.index('_INDEX_HTML = r"""') + len('_INDEX_HTML = r"""')
    end = source.index('"""\n\n\ndef create_app', start)
    WEBAPP.write_text(source[:start] + local + source[end:], encoding="utf-8")
    CLOUD_PAGE.write_text(render(text, cloud=True), encoding="utf-8")
    print(f"Wrote {WEBAPP.relative_to(ROOT)} and {CLOUD_PAGE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
