#!/usr/bin/env python3
"""
Docling Document Parser
========================
Converts documents (PDF, DOCX, PPTX, XLSX, HTML, images, etc.) to Markdown.

Features:
  - Table extraction (exported as Markdown tables + optional CSV)
  - Picture annotation using SmolVLM (local, no API key needed)
  - OCR for scanned PDFs and images (EasyOCR or Tesseract)
  - Chunking-friendly Markdown output
  - Live status spinner with download/processing detection

Supported input formats:
  PDF, DOCX, PPTX, XLSX, HTML, Markdown, AsciiDoc,
  PNG, JPEG, TIFF, BMP, WEBP, GIF

Installation:
  pip install docling
  pip install docling[easyocr]   # for EasyOCR (better accuracy)
  pip install hf_xet             # for faster model downloads

Usage:
  python docling_parser.py input.pdf
  python docling_parser.py input.pdf --output-dir ./output
  python docling_parser.py input.pdf --no-picture-annotation
  python docling_parser.py input.pdf --ocr tesseract
  python docling_parser.py *.pdf --export-tables-csv
  python docling_parser.py input.pdf --tableformer-mode accurate
"""

import argparse
import gc
import itertools
import json
import os
import sys
import threading
import time
from pathlib import Path

# ===========================================================================
# ✏️  USER CONFIGURATION
# Edit these switches, save the file, then run: python docling_parser.py file.pdf
# ===========================================================================

# ---------------------------------------------------------------------------
# 🔧 MASTER SWITCH
# ---------------------------------------------------------------------------
# True  = extract selectable text ONLY, all AI disabled (fastest, offline safe)
# False = use the individual switches below
TEXT_ONLY = False

# ---------------------------------------------------------------------------
# 🤖 AI FEATURES  (only active when TEXT_ONLY = False)
# ---------------------------------------------------------------------------

# OCR — reads text from scanned/image-based pages
ENABLE_OCR      = True    # False = skip OCR entirely (much faster for digital PDFs)
FORCE_FULL_OCR  = False   # True  = OCR every page even if text is already selectable

# OCR engine: "auto" | "easyocr" | "tesseract" | "rapidocr" | "mac"
# "auto"      = picks best available engine automatically
# "easyocr"   = best accuracy           (pip install "docling[easyocr]")
# "tesseract" = lightweight alternative
# "mac"       = macOS native Vision OCR (macOS only, no install needed)
OCR_ENGINE = "auto"

# Table extraction — uses TableFormer AI to detect and reconstruct tables
ENABLE_TABLE_EXTRACTION = True    # False = skip table detection (faster)
TABLE_CELL_MATCHING     = True    # False = use predicted cells instead of PDF cells / try False if columns merge incorrectly
                                  
TABLEFORMER_MODE = "accurate"     # "accurate" = best quality, slower  |  "fast" = quicker, less precise

# SmolVLM model size for picture annotation, use to describe the images in the document
# "256M" → smallest, least RAM (default)
# "500M" → better quality, ~3GB RAM
# "2B"   → best quality, ~8GB RAM (needs good hardware)
# Each size downloads once and stays cached — switching is free after that.
SMOLVLM_MODEL = "256M"

ENABLE_PICTURE_ANNOTATION    = True    # False = skip image descriptions (saves RAM)
ENABLE_PICTURE_CLASSIFICATION = True  # True  = classify image type (photo/chart/diagram)
PICTURE_PROMPT = "Describe the image in three concise sentences. Be accurate and specific."

# Code and formula enrichment (experimental — needs extra models, slower)
ENABLE_CODE_ENRICHMENT    = False  # True = detect and format code blocks
ENABLE_FORMULA_ENRICHMENT = False  # True = detect and convert math formulas to LaTeX

# ---------------------------------------------------------------------------
# 📄 PDF-SPECIFIC OPTIONS
# ---------------------------------------------------------------------------

# PDF backend — controls how the raw PDF bytes are read
# "pypdfium2" = default, fast, good for standard PDFs
# "docling"   = Docling's own parser, better layout on complex/academic PDFs
PDF_BACKEND = "pypdfium2"

# True  = always use the PDF's raw embedded text (faster, ignores VLM)
# False = let Docling decide (default, better quality)
FORCE_BACKEND_TEXT = False

# Generate page/picture images — needed for OCR and picture annotation
# Set both False only if doing pure text extraction (same effect as TEXT_ONLY)
GENERATE_PAGE_IMAGES    = True
GENERATE_PICTURE_IMAGES = True

# ---------------------------------------------------------------------------
# 📁 OFFICE / WEB FORMAT OPTIONS  (DOCX, PPTX, HTML, etc.)
# ---------------------------------------------------------------------------
# These formats use a simpler pipeline. Most AI features do not apply.
# Table extraction still works for DOCX and PPTX.
OFFICE_ENABLE_TABLE_EXTRACTION = True

# ---------------------------------------------------------------------------
# 📤 OUTPUT OPTIONS
# ---------------------------------------------------------------------------

# Also export each detected table as a separate CSV file
EXPORT_TABLES_CSV = False

# Maximum pages to process per document (None = no limit)
MAX_PAGES = None

# Maximum file size in bytes (None = no limit). Example: 20971520 = 20 MB
MAX_FILE_SIZE = None

# Timeout in seconds before aborting a slow document. None = no timeout (may hang)
# Recommended: 120 for normal use, None only for TEXT_ONLY mode
CONVERSION_TIMEOUT = 120

# ---------------------------------------------------------------------------
# ⚡ PERFORMANCE & MEMORY
# ---------------------------------------------------------------------------

# CPU threads PyTorch can use (None = all cores — fastest but may freeze your PC)
# Rule of thumb: half your core count. E.g. 4-core CPU -> set 2
CPU_THREADS = 2

# Hardware accelerator: "auto" | "cpu" | "cuda" (NVIDIA GPU) | "mps" (Apple Silicon)
ACCELERATOR = "auto"

# Image resolution scale — directly controls RAM usage
# 2.0 = high quality (default)  |  1.0 = half RAM  |  0.5 = minimum RAM
# Lower this FIRST if you get std::bad_alloc crashes
IMAGES_SCALE = 1.0

# Pages held in memory at once. Lower = less RAM.
# Recommended: 4 for 8GB RAM | 8 for 16GB RAM | 16 for 32GB RAM
PAGE_BATCH_SIZE = 4

# Custom model cache directory (None = default: ~/.cache/huggingface)
# Change if your C: drive is full. Example: "D:/ModelCache"
MODEL_CACHE_DIR = None








# ===========================================================================

# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx", ".doc",
    ".pptx", ".ppt",
    ".xlsx", ".xls",
    ".html", ".htm",
    ".md", ".markdown",
    ".adoc", ".asciidoc",
    ".png", ".jpg", ".jpeg",
    ".tiff", ".tif", ".bmp",
    ".webp", ".gif",
}

# ---------------------------------------------------------------------------
# Model cache paths (to detect if already downloaded)
# ---------------------------------------------------------------------------
HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
DOCLING_MODEL_CACHE = HF_CACHE / "hub"


# ---------------------------------------------------------------------------
# Spinner
# ---------------------------------------------------------------------------
class Spinner:
    """Animated terminal spinner that runs in a background thread."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = ""):
        self.message = message
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._current_line_len = 0

    def _spin(self):
        for frame in itertools.cycle(self.FRAMES):
            if self._stop_event.is_set():
                break
            line = f"\r  {frame}  {self.message}"
            sys.stdout.write(line)
            sys.stdout.flush()
            self._current_line_len = len(line)
            time.sleep(0.08)

    def update(self, message: str):
        self.message = message

    def start(self):
        self._thread.start()
        return self

    def stop(self, final_msg: str = "", status: str = "OK"):
        self._stop_event.set()
        self._thread.join()
        sys.stdout.write(f"\r{' ' * (self._current_line_len + 2)}\r")
        sys.stdout.flush()
        if final_msg:
            icon = "✔" if status == "OK" else "✖" if status == "ERROR" else "⚠"
            print(f"  {icon}  {final_msg}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


# ---------------------------------------------------------------------------
# Model download detection
# ---------------------------------------------------------------------------
def models_are_cached() -> bool:
    """Return True if Docling models appear to already be downloaded."""
    if not DOCLING_MODEL_CACHE.exists():
        return False
    for item in DOCLING_MODEL_CACHE.iterdir():
        if "docling" in item.name.lower():
            return True
    return False


def print_download_notice():
    width = 58
    print()
    print("  ┌" + "─" * width + "┐")
    print("  │  📥  FIRST-TIME SETUP — DOWNLOADING AI MODELS          │")
    print("  │                                                          │")
    print("  │  Docling needs to download its AI models.               │")
    print("  │  This only happens ONCE and may take 5–20 minutes       │")
    print("  │  depending on your internet speed (~1–3 GB total).      │")
    print("  │                                                          │")
    print("  │  Models will be saved to:                               │")
    cache_str = str(HF_CACHE)
    # Wrap long path across two lines if needed
    if len(cache_str) <= width - 4:
        print(f"  │  {cache_str:<{width-2}}│")
    else:
        print(f"  │  {cache_str[:width-4]+'...':^{width-2}}│")
    print("  │                                                          │")
    print("  │  ⚡ Tip: pip install hf_xet  for faster downloads       │")
    print("  └" + "─" * width + "┘")
    print()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert documents to chunking-ready Markdown using Docling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="FILE",
        help="Input file(s) to convert.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        metavar="DIR",
        help="Output directory (overrides saving next to the input file).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
def _resolve_ocr_options(ocr_engine: str, force_full_ocr: bool):
    if ocr_engine == "easyocr":
        from docling.datamodel.pipeline_options import EasyOcrOptions
        return EasyOcrOptions(force_full_page_ocr=force_full_ocr)

    elif ocr_engine == "tesseract":
        try:
            from docling.datamodel.pipeline_options import TesseractOcrOptions
            return TesseractOcrOptions(force_full_page_ocr=force_full_ocr)
        except ImportError:
            from docling.datamodel.pipeline_options import TesseractCliOcrOptions
            return TesseractCliOcrOptions(force_full_page_ocr=force_full_ocr)

    elif ocr_engine == "rapidocr":
        from docling.datamodel.pipeline_options import RapidOcrOptions
        return RapidOcrOptions(force_full_page_ocr=force_full_ocr)

    else:  # auto
        try:
            from docling.datamodel.pipeline_options import AutoOcrOptions
            return AutoOcrOptions()
        except ImportError:
            try:
                from docling.datamodel.pipeline_options import EasyOcrOptions
                return EasyOcrOptions(force_full_page_ocr=force_full_ocr)
            except ImportError:
                return None


# ---------------------------------------------------------------------------
# Converter builder
# ---------------------------------------------------------------------------
def build_converter():
    """Build a DocumentConverter with per-format pipeline options from config."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        PaginatedPipelineOptions,
        TableFormerMode,
        TableStructureOptions,
    )
    from docling.document_converter import (
        DocumentConverter,
        PdfFormatOption,
        WordFormatOption,
        PowerpointFormatOption,
        HTMLFormatOption,
        ImageFormatOption,
    )
    from docling.pipeline.simple_pipeline import SimplePipeline
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
    
    # ── Accelerator options ───────────────────────────────────────────────────
    try:
        from docling.datamodel.pipeline_options import AcceleratorOptions, AcceleratorDevice
        accel_device = {
            "auto": AcceleratorDevice.AUTO,
            "cpu":  AcceleratorDevice.CPU,
            "cuda": AcceleratorDevice.CUDA,
            "mps":  AcceleratorDevice.MPS,
        }.get(ACCELERATOR, AcceleratorDevice.AUTO)
        accel_opts = AcceleratorOptions(
            num_threads=CPU_THREADS or 4,
            device=accel_device,
        )
    except Exception:
        accel_opts = None

    # ── PDF backend ───────────────────────────────────────────────────────────
    pdf_backend = None  # None = let Docling pick its default
    if PDF_BACKEND == "docling":
        try:
            from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
            pdf_backend = DoclingParseV2DocumentBackend
        except ImportError:
            try:
                from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
                pdf_backend = DoclingParseDocumentBackend
            except ImportError:
                print("  ⚠  Docling backend not found — using default.")
    else:  # pypdfium2      
            try:
                from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
                pdf_backend = PyPdfiumDocumentBackend
            except ImportError:
                print("  ⚠  pypdfium2 backend not found — using default.")

    # ── Model cache ───────────────────────────────────────────────────────────
    artifacts_path = MODEL_CACHE_DIR  # None = use Docling default

    # =========================================================================
    # TEXT_ONLY — master override, no AI whatsoever
    # =========================================================================
    if TEXT_ONLY:
        pdf_opts = PdfPipelineOptions()
        pdf_opts.do_ocr                  = False
        pdf_opts.do_table_structure      = False
        pdf_opts.generate_page_images    = False
        pdf_opts.generate_picture_images = False
        pdf_opts.do_picture_description  = False
        if accel_opts:
            pdf_opts.accelerator_options = accel_opts
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_opts,
                    backend=pdf_backend,
                ),
            }
        )

    # =========================================================================
    # PDF pipeline options
    # =========================================================================
    pdf_opts = PdfPipelineOptions()

    # -- OCR ------------------------------------------------------------------
    pdf_opts.do_ocr = ENABLE_OCR
    if ENABLE_OCR:
        pdf_opts.ocr_options = _resolve_ocr_options(OCR_ENGINE, FORCE_FULL_OCR)

    # -- Tables ---------------------------------------------------------------
    pdf_opts.do_table_structure = ENABLE_TABLE_EXTRACTION
    if ENABLE_TABLE_EXTRACTION:
        pdf_opts.table_structure_options = TableStructureOptions(
            do_cell_matching=TABLE_CELL_MATCHING,
            mode=(
                TableFormerMode.ACCURATE
                if TABLEFORMER_MODE == "accurate"
                else TableFormerMode.FAST
            ),
        )

    # -- Images ---------------------------------------------------------------
    pdf_opts.generate_page_images    = GENERATE_PAGE_IMAGES
    pdf_opts.generate_picture_images = GENERATE_PICTURE_IMAGES
    pdf_opts.images_scale            = IMAGES_SCALE

    # -- Picture annotation (SmolVLM) -----------------------------------------
    pdf_opts.do_picture_description = False
    if ENABLE_PICTURE_ANNOTATION:
        try:
            from docling.datamodel.pipeline_options import (
                smolvlm_picture_description,
                PictureDescriptionVlmOptions,
            )
            _model_map = {
                "256M": "HuggingFaceTB/SmolVLM-256M-Instruct",
                "500M": "HuggingFaceTB/SmolVLM-500M-Instruct",
                "2B":   "HuggingFaceTB/SmolVLM-Instruct",
            }
            _repo_id = _model_map.get(SMOLVLM_MODEL, "HuggingFaceTB/SmolVLM-256M-Instruct")
            pic_opts = PictureDescriptionVlmOptions(
                repo_id=_repo_id,
                prompt=PICTURE_PROMPT,
            )
            pdf_opts.picture_description_options = pic_opts
        except ImportError:
            print("  ⚠  SmolVLM not available. Install: pip install docling[vlm]")

    # -- Picture classification -----------------------------------------------
    try:
        pdf_opts.do_picture_classification = ENABLE_PICTURE_CLASSIFICATION
    except AttributeError:
        pass  # older Docling versions don't have this field

    # -- Code / formula enrichment --------------------------------------------
    try:
        pdf_opts.do_code_enrichment    = ENABLE_CODE_ENRICHMENT
        pdf_opts.do_formula_enrichment = ENABLE_FORMULA_ENRICHMENT
    except AttributeError:
        pass  # not available in all versions

    # -- Text layer override --------------------------------------------------
    try:
        pdf_opts.force_backend_text = FORCE_BACKEND_TEXT
    except AttributeError:
        pass

    # -- Accelerator ----------------------------------------------------------
    if accel_opts:
        try:
            pdf_opts.accelerator_options = accel_opts
        except AttributeError:
            pass

    # -- Artifacts path -------------------------------------------------------
    if artifacts_path:
        try:
            pdf_opts.artifacts_path = artifacts_path
        except AttributeError:
            pass

    # =========================================================================
    # Office / paginated format pipeline options (DOCX, PPTX, HTML)
    # =========================================================================
    try:
        office_opts = PaginatedPipelineOptions()
        office_opts.do_table_structure   = OFFICE_ENABLE_TABLE_EXTRACTION
        office_opts.generate_page_images = GENERATE_PAGE_IMAGES
        office_opts.images_scale         = IMAGES_SCALE
        if accel_opts:
            office_opts.accelerator_options = accel_opts
        has_paginated = True
    except Exception:
        has_paginated = False

    # =========================================================================
    # Image format pipeline options (PNG, JPG, TIFF, etc.)
    # =========================================================================
    try:
        img_opts = PdfPipelineOptions()  # images use same pipeline as PDF
        img_opts.do_ocr                  = True   # always OCR images (no text layer)
        img_opts.ocr_options             = _resolve_ocr_options(OCR_ENGINE, True)
        img_opts.generate_page_images    = True
        img_opts.generate_picture_images = GENERATE_PICTURE_IMAGES
        img_opts.images_scale            = IMAGES_SCALE
        img_opts.do_table_structure      = ENABLE_TABLE_EXTRACTION
        img_opts.do_picture_description  = pdf_opts.do_picture_description
        if ENABLE_PICTURE_ANNOTATION and hasattr(img_opts, "picture_description_options"):
            img_opts.picture_description_options = pdf_opts.picture_description_options
        if accel_opts:
            try: img_opts.accelerator_options = accel_opts
            except AttributeError: pass
    except Exception as e:
        img_opts = None
        print(f"  ⚠  Could not configure image pipeline options: {e}")

    # =========================================================================
    # Build format_options dict
    # =========================================================================
    format_options = {
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pdf_opts,
            backend=pdf_backend,
        ),
    }

    # DOCX
    try:
        if has_paginated:
            format_options[InputFormat.DOCX] = WordFormatOption(
                pipeline_options=office_opts,
            )
        else:
            format_options[InputFormat.DOCX] = WordFormatOption(
                pipeline_cls=SimplePipeline,
            )
    except Exception:
        pass

    # PPTX
    try:
        if has_paginated:
            format_options[InputFormat.PPTX] = PowerpointFormatOption(
                pipeline_options=office_opts,
            )
        else:
            format_options[InputFormat.PPTX] = PowerpointFormatOption(
                pipeline_cls=SimplePipeline,
            )
    except Exception:
        pass

    # HTML
    try:
        if has_paginated:
            format_options[InputFormat.HTML] = HTMLFormatOption(
                pipeline_options=office_opts,
            )
        else:
            format_options[InputFormat.HTML] = HTMLFormatOption(
                pipeline_cls=SimplePipeline,
            )
    except Exception:
        pass

    # Images
    try:
        if img_opts:
            format_options[InputFormat.IMAGE] = ImageFormatOption(
                pipeline_options=img_opts,
            )
    except Exception:
        pass

    return DocumentConverter(format_options=format_options)


# ---------------------------------------------------------------------------
def export_tables_to_csv(result, output_dir: Path, stem: str) -> int:
    from docling_core.types.doc import TableItem

    table_count = 0
    for item, _level in result.document.iterate_items():
        if isinstance(item, TableItem):
            table_count += 1
            try:
                df = item.export_to_dataframe()
                csv_path = output_dir / f"{stem}_table_{table_count}.csv"
                df.to_csv(csv_path, index=False)
                print(f"  ✔  Table {table_count} → {csv_path.name}")
            except Exception as exc:
                print(f"  ⚠  Could not export table {table_count}: {exc}")

    return table_count


# ---------------------------------------------------------------------------
# Picture annotation summary
# ---------------------------------------------------------------------------
def print_picture_annotations(result):
    from docling_core.types.doc import PictureItem

    pictures = [
        item
        for item, _level in result.document.iterate_items()
        if isinstance(item, PictureItem)
    ]
    if not pictures:
        return

    print(f"\n  📷  {len(pictures)} picture(s) annotated:")
    for i, pic in enumerate(pictures, 1):
        for ann in getattr(pic, "annotations", []):
            desc = getattr(ann, "text", None)
            if desc:
                preview = desc[:120] + ("…" if len(desc) > 120 else "")
                print(f"    [{i}] {preview}")


# ---------------------------------------------------------------------------
# Full content JSON builder
# ---------------------------------------------------------------------------
def _build_content_json(result, original_filename: str) -> dict:
    """
    Build a JSON dict with fully resolved text content — not $ref pointers.
    Walks every item in the document and extracts actual text, tables,
    and picture annotations so the JSON is human-readable and chunker-ready.
    """
    from docling_core.types.doc import (
        TextItem, TableItem, PictureItem, SectionHeaderItem
    )

    doc = result.document
    elements = []

    for item, level in doc.iterate_items():
        # ── Section headings ───────────────────────────────────────────────
        if isinstance(item, SectionHeaderItem):
            elements.append({
                "type":    "heading",
                "level":   level,
                "text":    item.text.strip(),
                "page":    _get_page(item),
            })

        # ── Regular text / paragraphs ──────────────────────────────────────
        elif isinstance(item, TextItem):
            text = item.text.strip()
            if not text:
                continue
            elements.append({
                "type":  "text",
                "label": str(item.label) if hasattr(item, "label") else "paragraph",
                "text":  text,
                "page":  _get_page(item),
            })

        # ── Tables ────────────────────────────────────────────────────────
        elif isinstance(item, TableItem):
            try:
                df = item.export_to_dataframe()
                rows = [df.columns.tolist()] + df.values.tolist()
                table_data = [[str(cell) for cell in row] for row in rows]
            except Exception:
                table_data = []
            # Also export as markdown for readability
            try:
                table_md = item.export_to_markdown()
            except Exception:
                table_md = ""
            elements.append({
                "type":     "table",
                "page":     _get_page(item),
                "markdown": table_md,
                "data":     table_data,
            })

        # ── Pictures / figures ────────────────────────────────────────────
        elif isinstance(item, PictureItem):
            annotations = []
            for ann in getattr(item, "annotations", []):
                desc = getattr(ann, "text", None)
                if desc:
                    annotations.append(desc.strip())
            elements.append({
                "type":        "picture",
                "page":        _get_page(item),
                "annotations": annotations,
            })

    # ── Page count ────────────────────────────────────────────────────────────
    num_pages = getattr(doc, "num_pages", None)

    return {
        "filename":   original_filename,
        "num_pages":  num_pages,
        "elements":   elements,
    }


def _get_page(item) -> int | None:
    """Safely extract page number from a document item."""
    try:
        prov = item.prov
        if prov:
            return prov[0].page_no
    except (AttributeError, IndexError, TypeError):
        pass
    return None


# ---------------------------------------------------------------------------
# Single file conversion
# ---------------------------------------------------------------------------
def convert_file(converter, input_path: Path, output_dir_override=None) -> bool:
    ext = input_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        print(f"  ⚠  Skipping unsupported file type: {input_path.name}")
        return False

    output_dir = Path(output_dir_override) if output_dir_override else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    md_path = output_dir /  f"{stem}.md"
    meta_path = output_dir / f"{stem}_meta.json"

    file_size_mb = input_path.stat().st_size / (1024 * 1024)

    print(f"  {'─'*56}")
    print(f"  📄  {input_path.name}  ({file_size_mb:.2f} MB)")
    print(f"  {'─'*56}")
    print(f"  Output folder : {output_dir}")
    print()

    # ── Conversion (runs in a thread so spinner can animate) ─────────────────
    t0 = time.time()
    result = None
    error = None

    def do_convert():
        nonlocal result, error
        try:
            kwargs = {}
            if MAX_PAGES:
                kwargs["max_num_pages"] = MAX_PAGES
            if MAX_FILE_SIZE:
                kwargs["max_file_size"] = MAX_FILE_SIZE
            kwargs["raises_on_error"] = False  # don't crash on bad pages, skip them
            result = converter.convert(str(input_path), **kwargs)
        except Exception as exc:
            error = exc

    # Rotating status messages shown while converting
    stage_messages = [
        "Analysing document layout…",
        "Running OCR on scanned pages…",
        "Extracting and structuring tables…",
        "Annotating pictures with AI…",
        "Building Markdown output…",
        "Still working — large documents take time…",
    ]
    stage_cycle = itertools.cycle(stage_messages)

    spinner = Spinner(next(stage_cycle))
    spinner.start()

    convert_thread = threading.Thread(target=do_convert, daemon=True)
    convert_thread.start()

    last_update = time.time()
    stage_interval = 5.0  # seconds between message rotations

    while convert_thread.is_alive():
        if time.time() - last_update >= stage_interval:
            spinner.update(next(stage_cycle))
            last_update = time.time()
        time.sleep(0.1)

    convert_thread.join()
    elapsed = time.time() - t0

    if error:
        spinner.stop(f"Conversion failed: {error}", status="ERROR")
        return False

    spinner.stop(f"Document parsed successfully in {elapsed:.1f}s", status="OK")
    gc.collect()  # release page buffers immediately after parsing

    # ── Save Markdown ─────────────────────────────────────────────────────────
    with Spinner("Saving Markdown…"):
        markdown = result.document.export_to_markdown()
        md_path.write_text(markdown, encoding="utf-8")
    print(f"  ✔  Markdown  → {md_path.name}")

    # ── Save content JSON (full resolved text, tables, picture annotations) ──
    try:
        with Spinner("Saving content JSON…"):
            content_json = _build_content_json(result, input_path.name)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(content_json, f, indent=2, ensure_ascii=False, default=str)
        print(f"  ✔  Content JSON → {meta_path.name}")
    except Exception as exc:
        print(f"  ⚠  Could not export content JSON: {exc}")

    # ── Export tables to CSV ──────────────────────────────────────────────────
    if EXPORT_TABLES_CSV:
        with Spinner("Exporting tables to CSV…"):
            time.sleep(0.3)
        n = export_tables_to_csv(result, output_dir, stem)
        if n == 0:
            print("  ℹ  No tables detected in this document.")

    # ── Summary ───────────────────────────────────────────────────────────────
    pages = getattr(result.document, "num_pages", None) or "?"
    print(f"\n  ✅  Finished!  Pages: {pages}  |  Total time: {elapsed:.1f}s")

    if ENABLE_PICTURE_ANNOTATION:
        print_picture_annotations(result)

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def parse_with_docling(file_bytes: bytes, file_name: str, mime_type: str) -> str:
    import tempfile

    # ── Apply CPU/memory settings from config ─────────────────────────────────
    if CPU_THREADS is not None:
        try:
            import torch
            torch.set_num_threads(CPU_THREADS)
        except ImportError:
            pass
        os.environ["OMP_NUM_THREADS"]  = str(CPU_THREADS)
        os.environ["MKL_NUM_THREADS"]  = str(CPU_THREADS)
        os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_THREADS)

    # Write file_bytes to a temp file preserving the original extension
    suffix = Path(file_name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    # Resolve input files
    input_paths = [tmp_path]

    try:
        converter = build_converter()

        kwargs = {"raises_on_error": False}
        if MAX_PAGES:
            kwargs["max_num_pages"] = MAX_PAGES
        if MAX_FILE_SIZE:
            kwargs["max_file_size"] = MAX_FILE_SIZE

        result = converter.convert(str(tmp_path), **kwargs)
        gc.collect()

        return result.document.export_to_markdown()
    finally:
        tmp_path.unlink(missing_ok=True)

