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

Installation:
  pip install docling
  pip install docling[easyocr]   # for EasyOCR (better accuracy)
  pip install hf_xet             # for faster model downloads

"""


import asyncio
import gc
import json
import os
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

ENABLE_PICTURE_ANNOTATION    = False    # False = skip image descriptions (saves RAM)
ENABLE_PICTURE_CLASSIFICATION = False  # True  = classify image type (photo/chart/diagram)
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


# Custom model cache directory (None = default: ~/.cache/huggingface)
# Change if your C: drive is full. Example: "D:/ModelCache"
MODEL_CACHE_DIR = None


# ===========================================================================
# Supported file extensions
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
                df = item.export_to_dataframe(doc=doc)
                rows = [df.columns.tolist()] + df.values.tolist()
                table_data = [[str(cell) for cell in row] for row in rows]
            except Exception:
                table_data = []
            try:
                table_md = item.export_to_markdown(doc=doc)
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
    # num_pages may be a property, method, or plain int depending on version
    try:
        num_pages = doc.num_pages
        if callable(num_pages):
            num_pages = num_pages()
        num_pages = int(num_pages) if num_pages is not None else None
    except Exception:
        num_pages = None


    return {
        "filename":   original_filename,
        "num_pages":  num_pages,
        "elements":   elements,
    }

# ---------------------------------------------------------------------------
# Helper tools
# ---------------------------------------------------------------------------
def _get_page(item) -> int | None:
    """Safely extract page number from a document item."""
    try:
        prov = item.prov
        if prov:
            p = prov[0]
            # page_no is the attribute in newer docling_core
            for attr in ("page_no", "page", "page_number"):
                val = getattr(p, attr, None)
                if val is not None:
                    return int(val)
    except (AttributeError, IndexError, TypeError):
        pass
    return None

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
# Main
# ---------------------------------------------------------------------------
async def parse_with_docling(file_bytes: bytes, file_name: str, mime_type: str) -> str:
    import tempfile

    ext = Path(file_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        print(f"  ⚠  Skipping unsupported file type: {file_name}")
        return False
    
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
        # Build kwargs before entering the thread
        kwargs = {"raises_on_error": False}
        if MAX_PAGES:
            kwargs["max_num_pages"] = MAX_PAGES
        if MAX_FILE_SIZE:
            kwargs["max_file_size"] = MAX_FILE_SIZE

        # Run it in a thread pool so the event loop stays free
        # to serve chat requests while this file is being processed.
        def _blocking_convert():
            converter = build_converter()
            result = converter.convert(str(tmp_path), **kwargs)
            gc.collect()
            return result

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _blocking_convert)

        # Return structured JSON so pipelineDocument.py can use
        # page numbers, headings and element types in chunk metadata.
        # pipelineDocument.py detects this and calls chunk_elements()
        # instead of chunk_markdown() — falling back to markdown if needed.
        content_json = _build_content_json(result, file_name)

        if content_json["elements"]:
            return json.dumps(content_json, ensure_ascii=False)

        return result.document.export_to_markdown()
    finally:
        tmp_path.unlink(missing_ok=True)

