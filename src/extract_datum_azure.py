"""Identify datum labels attached to geometric elements in engineering drawing images.

Sends OpenAI-compatible chat completion requests to an Azure ML hosted Qwen3-VL
endpoint. No image preprocessing is applied; each image is read and sent as-is.
PDF inputs are rendered page by page to PNG (requires PyMuPDF) and then processed
like images.
Legal datum labels are A, B, and C. This script only describes each datum
label and the geometric element it is attached to. It does not judge whether
the datum usage is correct.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal, get_args

import requests
from PIL import Image
from pydantic import BaseModel, Field, ValidationError, field_validator

SCRIPT_VERSION = "1.3.0-datum-feature-class"

SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PDF_SUFFIX = ".pdf"
SUPPORTED = SUPPORTED_IMAGES | {PDF_SUFFIX}

DEFAULT_ENDPOINT = "https://qwen3-6-27b-ep.westeurope.inference.ml.azure.com/v1/chat/completions"

LEGAL_DATUM_LABELS = {"A", "B", "C"}

LOGGER = logging.getLogger("datum_extract")
STEP_TIMINGS: list[tuple[str, float, str]] = []


def setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"datum_run_{datetime.now():%Y%m%d_%H%M%S}.log"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-5s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))

    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(console)
    LOGGER.propagate = False
    return log_path


class _Step:
    """Mutable handle so a step can attach result detail before it is logged."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.start = time.perf_counter()
        self.elapsed = 0.0
        self.detail = ""


@contextmanager
def step(name: str, detail: str = ""):
    marker = _Step(name)
    LOGGER.info("START | %s%s", name, f" | {detail}" if detail else "")
    try:
        yield marker
    except Exception as exc:
        marker.elapsed = time.perf_counter() - marker.start
        STEP_TIMINGS.append((name, marker.elapsed, "failed"))
        LOGGER.error("FAIL  | %s | %.3fs | %s", name, marker.elapsed, exc)
        raise
    marker.elapsed = time.perf_counter() - marker.start
    STEP_TIMINGS.append((name, marker.elapsed, "ok"))
    LOGGER.info(
        "DONE  | %s | %.3fs%s", name, marker.elapsed,
        f" | {marker.detail}" if marker.detail else "",
    )


def log_timing_summary(total_elapsed: float) -> None:
    grouped: dict[str, list[float]] = {}
    for name, elapsed, _ in STEP_TIMINGS:
        entry = grouped.setdefault(name, [0.0, 0.0])
        entry[0] += 1
        entry[1] += elapsed

    LOGGER.info("=" * 78)
    LOGGER.info("TIMING SUMMARY (aggregated by step)")
    for name, (calls, total) in sorted(grouped.items(), key=lambda kv: kv[1][1], reverse=True):
        share = (total / total_elapsed * 100) if total_elapsed else 0.0
        LOGGER.info(
            "  %-34s calls=%-3d total=%8.3fs avg=%8.3fs %5.1f%%",
            name, int(calls), total, total / calls, share,
        )
    LOGGER.info("  TOTAL RUNTIME %.3fs", total_elapsed)
    LOGGER.info("=" * 78)


DatumFeatureClass = Literal[
    "planar", "cylindrical", "width", "spherical", "conical", "helical", "edge", "other",
]
DrawnShape = Literal[
    "line", "rectangle", "square", "circle", "arc", "triangle", "hexagon", "polygon",
    "ellipse", "slot", "other",
]
DATUM_FEATURE_CLASSES = set(get_args(DatumFeatureClass))
DRAWN_SHAPES = set(get_args(DrawnShape))

# Common model wordings mapped onto the closed class list.
_FEATURE_CLASS_ALIASES = {
    "plane": "planar", "planar_surface": "planar", "flat_surface": "planar",
    "cylinder": "cylindrical", "cylindrical_surface": "cylindrical", "axis": "cylindrical",
    "center_plane": "width", "sphere": "spherical", "cone": "conical",
    "thread": "helical", "threaded": "helical", "straight_edge": "edge",
}


def _snake(value) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


class DatumDetection(BaseModel):
    datum_label: str
    datum_feature_class: DatumFeatureClass
    drawn_shape: DrawnShape
    part_feature: str
    geometric_element: str
    explanation: str
    interpretation: str

    @field_validator("datum_label", mode="before")
    @classmethod
    def _normalize_label(cls, value) -> str:
        return str(value or "").strip().upper()

    @field_validator("datum_feature_class", mode="before")
    @classmethod
    def _normalize_feature_class(cls, value) -> str:
        text = _snake(value)
        text = _FEATURE_CLASS_ALIASES.get(text, text)
        if text not in DATUM_FEATURE_CLASSES:
            LOGGER.warning("unrecognized datum_feature_class %r mapped to 'other'", value)
            return "other"
        return text

    @field_validator("drawn_shape", mode="before")
    @classmethod
    def _normalize_drawn_shape(cls, value) -> str:
        text = _snake(value)
        return text if text in DRAWN_SHAPES else "other"

    @field_validator("part_feature", mode="before")
    @classmethod
    def _normalize_part_feature(cls, value) -> str:
        return _snake(value) or "unknown"


class DatumExtractionResponse(BaseModel):
    image_id: str | None = None
    results: list[DatumDetection] = Field(default_factory=list)


SYSTEM_PROMPT = """You read 2D engineering drawing images and describe datum feature symbols.
A datum feature symbol is a boxed letter, typically attached to a surface, edge, axis, or feature by a leader line or a triangular datum feature indicator.
The only legal datum labels in this system are A, B, and C, but you must not judge or decide whether a label in the image is correct, legal, or appropriately applied. Only describe what is visible.
For every datum feature symbol visible in the image, report:
- datum_label: the exact letter shown inside the datum symbol, uppercase, as visible in the image
- datum_feature_class: exactly one of planar, cylindrical, width, spherical, conical, helical, edge, other. This value is compared across different parts, so apply these rules consistently:
  - planar: the datum triangle sits on a straight outline or extension line representing a flat face (a flat face appears as a straight line in a 2D view) and is not aligned with a size dimension.
  - cylindrical: the datum belongs to an unthreaded round feature such as a hole, bore, shaft, or pin, is aligned with a diameter (Ø) dimension, or sits on a curved outline (arc or circle) representing a round surface.
  - width: the datum is aligned with a linear size dimension between two parallel lines, such as a slot, tab, or groove width.
  - spherical: the datum is on a spherical feature, for example with an SØ or SR callout.
  - conical: the datum is on a tapered feature such as a cone or countersink.
  - helical: the datum is on a threaded feature such as a screw, bolt, or tapped hole (thread callout like M6 or thread lines). Threaded features are helical even though they are round.
  - edge: the datum explicitly identifies a single straight edge rather than a face. Use rarely.
  - other: none of the above fits or it cannot be determined.
- drawn_shape: the basic 2D shape as drawn on the sheet, exactly one of line, rectangle, square, circle, arc, triangle, hexagon, polygon, ellipse, slot, other. If the datum points to an edge of a closed outline, name the shape of that outline. A curved line that is only part of a circle is "arc"; a full circle is "circle"; an isolated straight line is "line".
- part_feature: a short, generic, lowercase name of the physical feature, such as screw, hole, shaft, pin, flange, face, slot. No sizes, thread specs, counts, or part numbers.
- geometric_element: a concise description of the geometric element the datum symbol is attached to, based on its position and any nearby dimension or feature callouts
- explanation: what is visually connecting the datum symbol to that geometric element, such as a leader line, triangular flag, or direct attachment
- interpretation: a short, neutral restatement of what the datum symbol denotes for that element
Do not state whether the datum label is valid, legal, or correctly used. Only describe. Return JSON only, with no markdown fences and no commentary.
The JSON must have this exact shape:
{"image_id":"file name without extension","results":[{"datum_label":"A","datum_feature_class":"planar","drawn_shape":"rectangle","part_feature":"face","geometric_element":"short description","explanation":"short description","interpretation":"short description"}]}
If no datum feature symbol is visible, return {"image_id":"file name without extension","results":[]}.
"""

USER_PROMPT = """Look at this 2D engineering drawing image and find every datum feature symbol.
For each one, report datum_label, datum_feature_class, drawn_shape, part_feature, geometric_element, explanation, and interpretation as described.
Use only the allowed values for datum_feature_class and drawn_shape.
Do not decide whether the datum label is correct or legal. Only describe what is visible.
Return JSON only.
"""


def load_env(env_path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file without overriding real environment values."""
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_token() -> str:
    token = (os.environ.get("AZURE_ENDPOINT_TOKEN") or "").strip()
    if not token:
        raise RuntimeError(
            "Missing bearer token. Add AZURE_ENDPOINT_TOKEN=<your token> to the .env "
            "file in the project root, or set it as an environment variable."
        )
    return token


def image_data_url(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix == "jpg" else suffix
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"


def _extract_complete_json(text: str) -> str:
    """Return the first complete JSON object, ignoring surrounding model text."""
    text = (text or "").strip()
    start = text.find("{")
    if start < 0:
        raise RuntimeError("Model output contains no JSON object.")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    raise RuntimeError("Model returned incomplete JSON with unmatched braces.")


def _normalize_output(raw_output: str, image_id: str) -> DatumExtractionResponse:
    json_text = _extract_complete_json(raw_output)
    try:
        parsed = DatumExtractionResponse.model_validate_json(json_text)
    except ValidationError:
        data = json.loads(json_text)
        rows = data.get("results", []) if isinstance(data, dict) else []
        normalized = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            normalized.append({
                "datum_label": row.get("datum_label") or row.get("label"),
                "datum_feature_class": row.get("datum_feature_class") or row.get("feature_class"),
                "drawn_shape": row.get("drawn_shape") or row.get("geometry_type") or row.get("shape"),
                "part_feature": row.get("part_feature") or row.get("feature"),
                "geometric_element": str(row.get("geometric_element") or row.get("element") or ""),
                "explanation": str(row.get("explanation") or ""),
                "interpretation": str(row.get("interpretation") or ""),
            })
        parsed = DatumExtractionResponse.model_validate({
            "image_id": data.get("image_id") if isinstance(data, dict) else None,
            "results": normalized,
        })
    parsed.image_id = parsed.image_id or image_id
    return parsed


def _build_payload(image_path: Path, args) -> dict:
    with step("encode_image_base64", str(image_path)) as marker:
        data_url = image_data_url(image_path)
        marker.detail = f"{len(data_url) / 1024:.1f} KB data URL"
    return {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": False,
        "extra_body": {"top_k": args.top_k},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _post_chat(payload: dict, args) -> str:
    headers = {
        "Authorization": f"Bearer {args.token}",
        "Content-Type": "application/json",
    }
    with step("http_request_endpoint", f"max_tokens={payload.get('max_tokens')}") as marker:
        response = requests.post(args.endpoint, json=payload, headers=headers, timeout=args.timeout)
        marker.detail = f"HTTP {response.status_code} | {len(response.content) / 1024:.1f} KB response"
    if response.status_code == 401:
        raise RuntimeError("Endpoint rejected the bearer token (401). Check AZURE_ENDPOINT_TOKEN in .env.")
    response.raise_for_status()
    body = response.json()
    usage = body.get("usage") or {}
    if usage:
        LOGGER.info(
            "token usage | prompt=%s completion=%s total=%s",
            usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens"),
        )
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(f"Endpoint returned no choices: {json.dumps(body)[:500]}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    raw_output = (content or "").strip() or (message.get("reasoning_content") or "").strip()
    if not raw_output:
        raise RuntimeError(
            f"Endpoint returned empty content. finish_reason={choices[0].get('finish_reason')}"
        )
    return raw_output


def check_endpoint(args) -> None:
    probe = {
        "model": args.model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 8,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {
        "Authorization": f"Bearer {args.token}",
        "Content-Type": "application/json",
    }
    try:
        with step("endpoint_health_check", args.endpoint):
            response = requests.post(args.endpoint, json=probe, headers=headers, timeout=30)
    except requests.RequestException as exc:
        raise RuntimeError(f"Cannot reach endpoint {args.endpoint}. Details: {exc}") from exc
    if response.status_code == 401:
        raise RuntimeError("Endpoint rejected the bearer token (401). Check AZURE_ENDPOINT_TOKEN in .env.")
    if response.status_code >= 400:
        raise RuntimeError(
            f"Endpoint health check failed with HTTP {response.status_code}: {response.text[:300]}"
        )


def call_endpoint(image_path: Path, args) -> DatumExtractionResponse:
    with step("build_payload"):
        payload = _build_payload(image_path, args)
    last_error = None
    for attempt in range(args.retries + 1):
        try:
            with step("extraction_attempt", f"attempt {attempt + 1}/{args.retries + 1}"):
                raw_output = _post_chat(payload, args)
                with step("parse_model_json") as marker:
                    result = _normalize_output(raw_output, image_path.stem)
                    marker.detail = f"{len(result.results)} datum symbol(s) found"
            LOGGER.info("parsed endpoint output from choices[0].message.content")
            return result
        except (requests.RequestException, KeyError, json.JSONDecodeError,
                ValidationError, RuntimeError) as exc:
            last_error = exc
            if attempt < args.retries:
                LOGGER.warning("attempt %d failed: %s; retrying", attempt + 1, exc)
                with step("retry_backoff_sleep", f"{2 * (attempt + 1)}s"):
                    time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Endpoint extraction failed after {args.retries + 1} attempt(s): {last_error}")


def process_file(path: Path, args) -> dict:
    LOGGER.info("calling endpoint for %s", path)
    response = call_endpoint(path, args)
    results = [detection.model_dump() for detection in response.results]
    datum_feature_classes: dict[str, list[str]] = {}
    for detection in response.results:
        classes = datum_feature_classes.setdefault(detection.datum_label, [])
        if detection.datum_feature_class not in classes:
            classes.append(detection.datum_feature_class)
    datum_found = bool(results)
    return {
        "image_id": path.stem,
        "source_file": str(path),
        "legal_datum_labels": sorted(LEGAL_DATUM_LABELS),
        "datum_found": datum_found,
        "status": "datum_found" if datum_found else "no_datum_found",
        "datum_feature_classes": dict(sorted(datum_feature_classes.items())),
        "results": results,
    }


def input_files(input_paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for input_path in input_paths:
        if input_path.is_file():
            if input_path.suffix.lower() not in SUPPORTED:
                raise ValueError(f"Unsupported input type: {input_path}")
            files.append(input_path)
        elif input_path.is_dir():
            files.extend(sorted(p for p in input_path.rglob("*") if p.suffix.lower() in SUPPORTED))
        else:
            raise FileNotFoundError(f"Input not found: {input_path}")
    unique: dict[Path, Path] = {}
    for path in files:
        unique.setdefault(path.resolve(), path)
    return list(unique.values())


def pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int, rotation: int = 90) -> list[tuple[Path, int]]:
    """Render every PDF page to PNG, rotated clockwise by `rotation` degrees."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PDF input requires PyMuPDF. Install it with: python -m pip install pymupdf"
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    pages: list[tuple[Path, int]] = []
    with fitz.open(pdf_path) as doc:
        if doc.page_count == 0:
            raise RuntimeError(f"PDF has no pages: {pdf_path}")
        for number, page in enumerate(doc, start=1): # type: ignore
            # Single-page PDFs keep the PDF stem so image_id matches the uploaded file name.
            stem = pdf_path.stem if doc.page_count == 1 else f"{pdf_path.stem}_page{number}"
            image_path = out_dir / f"{stem}.png"
            # PDF /Rotate is clockwise; only the in-memory copy is changed, not the source file.
            page.set_rotation((page.rotation + rotation) % 360)
            page.get_pixmap(dpi=dpi).save(image_path)
            pages.append((image_path, number))
    return pages


def main() -> int:
    run_start = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Describe datum feature symbols and their attached geometric elements using a hosted Qwen3-VL endpoint."
    )
    parser.add_argument("input", type=Path, nargs="+", help="One or more image/PDF files or directories")
    parser.add_argument("--output", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"), help="Timed log directory")
    parser.add_argument("--endpoint", default=os.environ.get("AZURE_ENDPOINT_URL") or DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=os.environ.get("AZURE_ENDPOINT_MODEL"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--pdf-dpi", type=int, default=200, help="Render resolution for PDF pages")
    parser.add_argument("--pdf-rotation", type=int, default=90, choices=[0, 90, 180, 270],
                        help="Clockwise rotation applied to PDF pages before rendering")
    args = parser.parse_args()

    log_path = setup_logging(args.log_dir)
    LOGGER.info("Datum Extractor (Azure endpoint) version: %s", SCRIPT_VERSION)
    LOGGER.info("log file: %s", log_path)
    input_label = ", ".join(str(p) for p in args.input)
    LOGGER.info("input: %s | output: %s", input_label, args.output)

    with step("load_env_and_token", str(args.env_file)):
        load_env(args.env_file)
        args.endpoint = os.environ.get("AZURE_ENDPOINT_URL", args.endpoint)
        args.model = os.environ.get("AZURE_ENDPOINT_MODEL", args.model)
        args.token = resolve_token()

    args.output.mkdir(parents=True, exist_ok=True)
    if not args.skip_health_check:
        check_endpoint(args)

    with step("discover_input_files", input_label) as marker:
        files = input_files(args.input)
        marker.detail = f"{len(files)} file(s)"
    if not files:
        LOGGER.error("no supported images or PDFs found")
        return 2

    all_documents = []
    for index, path in enumerate(files, start=1):
        LOGGER.info("-" * 78)
        LOGGER.info("processing file %d/%d: %s", index, len(files), path)
        is_pdf = path.suffix.lower() == PDF_SUFFIX
        if is_pdf:
            try:
                with step("convert_pdf_to_images",
                          f"{path} @ {args.pdf_dpi} dpi, rotate {args.pdf_rotation} cw") as marker:
                    pages = pdf_to_images(path, args.output / "pdf_pages", args.pdf_dpi, args.pdf_rotation)
                    marker.detail = f"{len(pages)} page(s)"
            except Exception as exc:
                LOGGER.error("%s | %s", path, exc)
                all_documents.append({"source_file": str(path), "error": str(exc)})
                continue
        else:
            pages = [(path, None)]

        for image_path, page_number in pages:
            try:
                with step("process_file_total", str(image_path)) as file_marker:
                    document = process_file(image_path, args)
                    if is_pdf:
                        document["source_pdf"] = str(path)
                        document["pdf_page"] = page_number
                    all_documents.append(document)
                    json_path = args.output / f"{image_path.stem}_datum.json"
                    with step("write_document_json", str(json_path)):
                        json_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
                    file_marker.detail = f"{len(document['results'])} datum symbol(s)"
            except Exception as exc:
                LOGGER.error("%s | %s", image_path, exc)
                error_doc: dict[str, object] = {"source_file": str(image_path), "error": str(exc)}
                if is_pdf:
                    error_doc.update(source_pdf=str(path), pdf_page=page_number)
                all_documents.append(error_doc)

    summary = args.output / "all_datum_results.json"
    with step("write_summary_json", str(summary)):
        summary.write_text(json.dumps({
            "inputs": [str(p) for p in files],
            "document_count": len(all_documents),
            "documents": all_documents,
        }, indent=2), encoding="utf-8")

    log_timing_summary(time.perf_counter() - run_start)
    LOGGER.info("timed log written to %s", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
