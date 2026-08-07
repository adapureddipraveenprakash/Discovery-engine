"""Convert INTERVIEW_PREP.md into a polished PDF using reportlab + markdown."""
import re
from pathlib import Path

import markdown as md
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).parent.parent
SRC = ROOT / "INTERVIEW_PREP.md"
OUT = ROOT / "INTERVIEW_PREP.pdf"


# ---------- styles ----------
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "TitleBig", parent=styles["Title"], fontSize=24, leading=28,
    spaceAfter=14, textColor=colors.HexColor("#1a73e8"),
)
h1 = ParagraphStyle(
    "H1", parent=styles["Heading1"], fontSize=18, leading=22,
    spaceBefore=18, spaceAfter=10, textColor=colors.HexColor("#1a73e8"),
)
h2 = ParagraphStyle(
    "H2", parent=styles["Heading2"], fontSize=14, leading=18,
    spaceBefore=14, spaceAfter=8, textColor=colors.HexColor("#34495e"),
)
h3 = ParagraphStyle(
    "H3", parent=styles["Heading3"], fontSize=11.5, leading=15,
    spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#1a1a1a"),
    fontName="Helvetica-Bold",
)
body = ParagraphStyle(
    "Body", parent=styles["BodyText"], fontSize=10, leading=14,
    spaceAfter=6, alignment=TA_LEFT,
)
quote = ParagraphStyle(
    "Quote", parent=body, fontSize=10, leading=14,
    leftIndent=18, rightIndent=12, spaceBefore=4, spaceAfter=8,
    textColor=colors.HexColor("#2d3748"), fontName="Helvetica-Oblique",
    borderColor=colors.HexColor("#cbd5e0"), borderWidth=0,
    backColor=colors.HexColor("#f7fafc"),
    borderPadding=(6, 8, 6, 12),
)
code_block = ParagraphStyle(
    "Code", parent=body, fontName="Courier", fontSize=8.5, leading=11,
    leftIndent=14, rightIndent=14, spaceBefore=6, spaceAfter=8,
    backColor=colors.HexColor("#f1f3f4"), borderPadding=(6, 8, 6, 8),
    textColor=colors.HexColor("#202124"),
)
inline_bullet = ParagraphStyle(
    "Bullet", parent=body, leftIndent=18, bulletIndent=6,
)


# ---------- helpers ----------
def render_inline(text: str) -> str:
    """Convert markdown inline syntax to reportlab XML.

    Order matters — code spans are extracted first into placeholders so their
    contents (which may contain underscores or asterisks) aren't touched by
    later italic/bold regexes.
    """
    # 1. Pull out inline code into placeholders
    code_spans: list[str] = []

    def stash_code(m):
        code_spans.append(m.group(1))
        return f"\x00CODE{len(code_spans) - 1}\x00"

    text = re.sub(r"`([^`]+?)`", stash_code, text)

    # 2. Pull out links similarly
    link_spans: list[tuple[str, str]] = []

    def stash_link(m):
        link_spans.append((m.group(1), m.group(2)))
        return f"\x00LINK{len(link_spans) - 1}\x00"

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", stash_link, text)

    # 3. Escape XML on the remaining text
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # 4. Bold then italic on plain text
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^\*]+?)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!_)_([^_]+?)_(?!_)", r"<i>\1</i>", text)

    # 5. Restore code spans (escape their contents, then wrap in font tag)
    def code_xml(s: str) -> str:
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f'<font face="Courier" size="9" color="#c7254e" backColor="#f9f2f4">{s}</font>'

    text = re.sub(
        r"\x00CODE(\d+)\x00",
        lambda m: code_xml(code_spans[int(m.group(1))]),
        text,
    )

    # 6. Restore links — drop internal anchor links (which reportlab can't resolve)
    def link_xml(label: str, url: str) -> str:
        label = label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if url.startswith("#"):
            return f"<b>{label}</b>"
        return f'<link href="{url}" color="blue">{label}</link>'

    text = re.sub(
        r"\x00LINK(\d+)\x00",
        lambda m: link_xml(*link_spans[int(m.group(1))]),
        text,
    )
    return text


def parse_table(lines, i):
    """Parse a markdown table starting at lines[i]. Returns (Table, new_i)."""
    header = [c.strip() for c in lines[i].strip().strip("|").split("|")]
    i += 2  # skip separator row
    rows = [header]
    while i < len(lines) and lines[i].lstrip().startswith("|"):
        cells = [render_inline(c.strip()) for c in lines[i].strip().strip("|").split("|")]
        rows.append(cells)
        i += 1
    # wrap each cell in a Paragraph for word-wrap
    cell_style = ParagraphStyle(
        "TableCell", parent=body, fontSize=8.5, leading=11, spaceAfter=0
    )
    pdata = [[Paragraph(c, cell_style) for c in row] for row in rows]
    tbl = Table(pdata, repeatRows=1, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a73e8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return tbl, i


def parse_markdown(text):
    """Parse the markdown text and return a list of reportlab flowables."""
    flow = []
    lines = text.split("\n")
    i = 0
    in_code = False
    code_buf = []
    in_list = False
    list_items = []

    def flush_list():
        nonlocal list_items
        if list_items:
            for item in list_items:
                flow.append(Paragraph("• " + render_inline(item), inline_bullet))
            list_items = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Code block delimiter
        if stripped.startswith("```"):
            if in_code:
                # end of code block
                code_text = "<br/>".join(
                    c.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                     .replace(" ", "&nbsp;")
                    for c in code_buf
                )
                flow.append(Paragraph(code_text or "&nbsp;", code_block))
                code_buf = []
                in_code = False
            else:
                flush_list()
                in_code = True
            i += 1
            continue

        if in_code:
            code_buf.append(line)
            i += 1
            continue

        # Horizontal rule
        if stripped == "---":
            flush_list()
            flow.append(Spacer(1, 6))
            flow.append(Table(
                [[""]], colWidths=[6.5 * inch], rowHeights=[0.5],
                style=TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.6, colors.HexColor("#cbd5e0"))]),
            ))
            flow.append(Spacer(1, 6))
            i += 1
            continue

        # Headings
        if stripped.startswith("# "):
            flush_list()
            flow.append(Paragraph(render_inline(stripped[2:]), title_style))
            i += 1
            continue
        if stripped.startswith("## "):
            flush_list()
            flow.append(Paragraph(render_inline(stripped[3:]), h1))
            i += 1
            continue
        if stripped.startswith("### "):
            flush_list()
            flow.append(Paragraph(render_inline(stripped[4:]), h2))
            i += 1
            continue
        if stripped.startswith("#### "):
            flush_list()
            flow.append(Paragraph(render_inline(stripped[5:]), h3))
            i += 1
            continue

        # Tables
        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s\-:|]+\|\s*$", lines[i + 1]):
            flush_list()
            tbl, i = parse_table(lines, i)
            flow.append(KeepTogether([tbl, Spacer(1, 8)]))
            continue

        # Block quote
        if stripped.startswith("> "):
            flush_list()
            quote_lines = []
            while i < len(lines) and lines[i].lstrip().startswith(">"):
                ql = lines[i].lstrip()[1:].lstrip()
                quote_lines.append(ql)
                i += 1
            quote_text = "<br/>".join(render_inline(q) for q in quote_lines if q)
            flow.append(Paragraph(quote_text, quote))
            continue

        # Bullet list
        if stripped.startswith("- ") or stripped.startswith("* "):
            in_list = True
            list_items.append(stripped[2:])
            i += 1
            continue

        # Numbered list
        m = re.match(r"^\d+\.\s+(.+)$", stripped)
        if m:
            in_list = True
            list_items.append(m.group(1))
            i += 1
            continue

        # Empty line
        if not stripped:
            flush_list()
            flow.append(Spacer(1, 4))
            i += 1
            continue

        # Plain paragraph
        flush_list()
        flow.append(Paragraph(render_inline(stripped), body))
        i += 1

    flush_list()
    return flow


def build_pdf():
    text = SRC.read_text()

    doc = SimpleDocTemplate(
        str(OUT), pagesize=letter,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        title="Two-Tower Retrieval — Interview Prep",
        author="Nikhil Gautam",
    )

    flow = parse_markdown(text)
    doc.build(flow)
    print(f"Wrote {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    build_pdf()
