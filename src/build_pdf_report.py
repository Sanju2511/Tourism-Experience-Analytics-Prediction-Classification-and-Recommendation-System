from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_MD = PROJECT_ROOT / "reports" / "REPORT_DETAILED.md"
OUTPUT_PDF = PROJECT_ROOT / "reports" / "TOURISM_EXPERIENCE_ANALYTICS_FINAL_REPORT.pdf"


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _md_to_lines(md_text: str) -> list[str]:
    lines = []
    for raw in md_text.splitlines():
        line = raw.strip()
        if not line:
            lines.append("")
            continue
        if line.startswith("#"):
            line = line.lstrip("#").strip().upper()
        line = line.replace("**", "").replace("`", "")
        lines.extend(_wrap(line, width=95))
    return lines


def _wrap(text: str, width: int = 95) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    out = []
    cur = words[0]
    for w in words[1:]:
        if len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            out.append(cur)
            cur = w
    out.append(cur)
    return out


def _paginate(lines: list[str], max_lines: int = 46) -> list[list[str]]:
    pages = []
    i = 0
    while i < len(lines):
        pages.append(lines[i : i + max_lines])
        i += max_lines
    return pages


def build_pdf(text: str, out_path: Path) -> None:
    lines = _md_to_lines(text)
    pages = _paginate(lines)

    objects: list[bytes] = []

    # 1: Catalog, 2: Pages, 3: Font. Page and content objects follow.
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")

    page_ids = []
    content_ids = []

    next_obj_id = 4
    for _ in pages:
        page_ids.append(next_obj_id)
        content_ids.append(next_obj_id + 1)
        next_obj_id += 2

    kids = " ".join([f"{pid} 0 R" for pid in page_ids]).encode("ascii")
    objects.append(b"<< /Type /Pages /Count " + str(len(page_ids)).encode("ascii") + b" /Kids [ " + kids + b" ] >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for page_idx, page_lines in enumerate(pages, start=1):
        stream_lines = [
            "BT",
            "/F1 11 Tf",
            "52 800 Td",
            "14 TL",
        ]
        for line in page_lines:
            stream_lines.append(f"({_escape_pdf_text(line)}) Tj")
            stream_lines.append("T*")
        stream_lines.append(f"(Page {page_idx} of {len(pages)}) Tj")
        stream_lines.append("ET")

        content = "\n".join(stream_lines).encode("latin-1", errors="replace")
        content_obj = b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"\nendstream"

        page_obj = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] "
            + b"/Resources << /Font << /F1 3 0 R >> >> "
            + b"/Contents "
            + str(content_ids[page_idx - 1]).encode("ascii")
            + b" 0 R >>"
        )

        objects.append(page_obj)
        objects.append(content_obj)

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]

    for i, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{i} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_pos = len(pdf)
    pdf.extend(f"xref\n0 {len(objects)+1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode("ascii"))

    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode("ascii")
    )

    out_path.write_bytes(pdf)
    print(f"PDF created: {out_path}")


def main() -> None:
    text = INPUT_MD.read_text(encoding="utf-8")
    build_pdf(text, OUTPUT_PDF)


if __name__ == "__main__":
    main()
