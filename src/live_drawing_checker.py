"""Live Drawing Checker: datum extraction for PDFs in %TEMP%\\Live Drawing Checker.

Takes no command-line arguments. Every PDF in that folder is rendered, sent to the
Azure endpoint configured in .env, and its results are written next to it as
<pdf name>.json. Built into a standalone exe with PyInstaller.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from extract_datum_azure import (
    DEFAULT_ENDPOINT,
    LEGAL_DATUM_LABELS,
    LOGGER,
    SCRIPT_VERSION,
    call_endpoint,
    load_env,
    log_timing_summary,
    pdf_to_images,
    resolve_token,
    setup_logging,
    step,
)

APP_FOLDER_NAME = "Live Drawing Checker"
PDF_DPI = 200
PDF_ROTATION = 90


def _app_dir() -> Path:
    """Folder holding the exe when frozen, otherwise the project root."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _load_endpoint_env() -> None:
    # load_env never overrides existing values, so a .env next to the exe wins over a bundled one.
    load_env(_app_dir() / ".env")
    bundle_dir = getattr(sys, "_MEIPASS", None)
    if bundle_dir:
        load_env(Path(bundle_dir) / ".env")


def _endpoint_settings() -> SimpleNamespace:
    return SimpleNamespace(
        endpoint=os.environ.get("AZURE_ENDPOINT_URL") or DEFAULT_ENDPOINT,
        model=os.environ.get("AZURE_ENDPOINT_MODEL"),
        token=resolve_token(),
        max_tokens=8192,
        temperature=0.0,
        top_p=0.95,
        top_k=20,
        timeout=300,
        retries=1,
    )


def _write_json_atomic(path: Path, data: dict) -> None:
    # Write then rename so the CATIA side never reads a half-written file.
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _already_processed(json_path: Path) -> bool:
    """PDF names are unique, so an existing non-error JSON means the PDF is done."""
    if not json_path.is_file():
        return False
    try:
        return json.loads(json_path.read_text(encoding="utf-8")).get("status") != "error"
    except (OSError, ValueError, AttributeError):
        return False


def process_pdf(pdf_path: Path, work_dir: Path, settings: SimpleNamespace) -> dict:
    with step("convert_pdf_to_images", f"{pdf_path.name} @ {PDF_DPI} dpi, rotate {PDF_ROTATION} cw") as marker:
        pages = pdf_to_images(pdf_path, work_dir / "pdf_pages", PDF_DPI, PDF_ROTATION)
        marker.detail = f"{len(pages)} page(s)"

    results: list[dict] = []
    errors: list[dict] = []
    for image_path, page_number in pages:
        try:
            with step("process_page", f"{pdf_path.name} page {page_number}"):
                response = call_endpoint(image_path, settings)
            results.extend({**d.model_dump(), "pdf_page": page_number} for d in response.results)
        except Exception as exc:
            LOGGER.error("%s page %d | %s", pdf_path.name, page_number, exc)
            errors.append({"pdf_page": page_number, "error": str(exc)})

    feature_classes: dict[str, list[str]] = {}
    for row in results:
        classes = feature_classes.setdefault(row["datum_label"], [])
        if row["datum_feature_class"] not in classes:
            classes.append(row["datum_feature_class"])

    if results:
        status = "datum_found"
    elif errors:
        status = "error"
    else:
        status = "no_datum_found"

    document = {
        "image_id": pdf_path.stem,
        "source_pdf": str(pdf_path),
        "page_count": len(pages),
        "legal_datum_labels": sorted(LEGAL_DATUM_LABELS),
        "datum_found": bool(results),
        "status": status,
        "datum_feature_classes": dict(sorted(feature_classes.items())),
        "results": results,
    }
    if errors:
        document["errors"] = errors
    return document


def main() -> int:
    run_start = time.perf_counter()
    # resolve() expands Windows 8.3 short names (e.g. SANJAY~1) returned by gettempdir().
    work_dir = (Path(tempfile.gettempdir()) / APP_FOLDER_NAME).resolve()
    if not work_dir.is_dir():
        print(f"Folder not found: {work_dir}", file=sys.stderr)
        return 2

    log_path = setup_logging(work_dir / "logs")
    LOGGER.info("Live Drawing Checker | extractor version: %s", SCRIPT_VERSION)
    LOGGER.info("working folder: %s", work_dir)

    try:
        with step("load_env_and_token", str(_app_dir())):
            _load_endpoint_env()
            settings = _endpoint_settings()
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 3

    pdfs = sorted(p for p in work_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    if not pdfs:
        LOGGER.error("no PDFs found in %s", work_dir)
        return 2

    failures = 0
    skipped = 0
    for index, pdf_path in enumerate(pdfs, start=1):
        json_path = work_dir / f"{pdf_path.stem}.json"
        if _already_processed(json_path):
            LOGGER.info("skipping PDF %d/%d (already processed): %s", index, len(pdfs), pdf_path.name)
            skipped += 1
            continue

        LOGGER.info("-" * 78)
        LOGGER.info("processing PDF %d/%d: %s", index, len(pdfs), pdf_path.name)
        try:
            document = process_pdf(pdf_path, work_dir, settings)
        except Exception as exc:
            LOGGER.error("%s | %s", pdf_path.name, exc)
            document = {
                "image_id": pdf_path.stem,
                "source_pdf": str(pdf_path),
                "datum_found": False,
                "status": "error",
                "datum_feature_classes": {},
                "results": [],
                "errors": [{"pdf_page": None, "error": str(exc)}],
            }
        if document["status"] == "error":
            failures += 1

        with step("write_document_json", str(json_path)):
            _write_json_atomic(json_path, document)

    LOGGER.info(
        "PDFs: %d total | %d processed | %d skipped | %d failed",
        len(pdfs), len(pdfs) - skipped, skipped, failures,
    )
    log_timing_summary(time.perf_counter() - run_start)
    LOGGER.info("timed log written to %s", log_path)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
