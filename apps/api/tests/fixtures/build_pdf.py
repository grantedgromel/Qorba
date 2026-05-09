"""Build synthetic PDFs at test time so we have something for the cascade
without depending on real (private) tearsheets. PyMuPDF is already a runtime
dep so we use it for both reading and writing here."""

from __future__ import annotations

import calendar
from io import BytesIO

import pymupdf

_MONTHS = list(calendar.month_abbr)[1:]  # Jan..Dec


def calendar_table_pdf(
    *,
    years: list[int],
    values_pct: list[list[float]],  # rows = years, cols = 12 months, in percent
) -> bytes:
    """Build a one-page PDF whose only content is a Year × month table that
    PyMuPDF.find_tables can pick up. Values are written as percent strings.
    """
    if len(years) != len(values_pct):
        raise ValueError("years and values_pct must align")
    for row in values_pct:
        if len(row) != 12:
            raise ValueError("each row must have 12 monthly values")

    doc = pymupdf.open()
    page = doc.new_page(width=792, height=612)  # US Letter landscape

    headers = ["Year", *_MONTHS]
    rows = [
        [str(y), *(f"{v:.2f}" for v in row)]
        for y, row in zip(years, values_pct, strict=True)
    ]

    cell_w = 56
    cell_h = 22
    x0, y0 = 36, 36

    # Draw header
    for i, h in enumerate(headers):
        rect = pymupdf.Rect(x0 + i * cell_w, y0, x0 + (i + 1) * cell_w, y0 + cell_h)
        page.draw_rect(rect, color=(0, 0, 0), width=0.5)
        page.insert_text((rect.x0 + 4, rect.y1 - 6), h, fontsize=10)

    # Draw data
    for ri, row in enumerate(rows):
        ry0 = y0 + (ri + 1) * cell_h
        for ci, cell in enumerate(row):
            rect = pymupdf.Rect(x0 + ci * cell_w, ry0, x0 + (ci + 1) * cell_w, ry0 + cell_h)
            page.draw_rect(rect, color=(0, 0, 0), width=0.5)
            page.insert_text((rect.x0 + 4, rect.y1 - 6), cell, fontsize=10)

    buf = BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()
