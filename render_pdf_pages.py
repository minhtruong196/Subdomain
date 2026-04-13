from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(".codex_deps").resolve()))

import fitz


PDF_PATH = Path(
    "A_New_Technique_for_the_Subdomain_Method_in_Predicting_Electromagnetic_Performance_of_Surface-Mounted_Permanent_Magnet_Motors_With_Shaped_Magnets_and_a_Quasi-Regular_Polygon_Rotor_Core.pdf"
)
OUTPUT_DIR = Path("outputs/rendered_pages")


def render_page(doc: fitz.Document, page_index: int, zoom: float = 4.0) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    page = doc[page_index]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    path = OUTPUT_DIR / f"page_{page_index + 1:02d}_zoom{zoom:g}.png"
    pix.save(path)
    return path


def main() -> None:
    doc = fitz.open(PDF_PATH)
    # PDF pages are zero-based. These pages contain Table I / Eq. (17)-(21) / Appendix.
    for page_index in [2, 5, 11, 12]:
        path = render_page(doc, page_index)
        print(path.resolve())


if __name__ == "__main__":
    main()
