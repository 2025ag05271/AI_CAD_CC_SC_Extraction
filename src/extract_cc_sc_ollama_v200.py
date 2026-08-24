"""Extract No.<digits> to CC/SC associations from engineering drawing images.

Uses an Ollama vision model through http://localhost:11434/api/chat.
Python 3.11, Windows-friendly, no cloud calls.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import requests
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

NO_RE = re.compile(r"(?i)\bNO\.?\s*(\d+)\b")
SCRIPT_VERSION = "2.5.0-evidence-priority"

SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class Detection(BaseModel):
    dimension_number: str
    classification: Literal["CC", "SC"] | None = None
    evidence_text: str | None = None
    interpretation: str
    confidence: float = Field(ge=0.0, le=1.0)
    needs_review: bool


class ExtractionResponse(BaseModel):
    document_convention: str | None = None
    results: list[Detection] = Field(default_factory=list)


class MinimalDetection(BaseModel):
    dimension_number: str
    classification: Literal["CC", "SC"] | None = None
    evidence_text: str | None = None


class MinimalResponse(BaseModel):
    results: list[MinimalDetection] = Field(default_factory=list)


RESPONSE_SCHEMA = ExtractionResponse.model_json_schema()

SYSTEM_PROMPT = """You extract CC/SC classifications from engineering document images.
Do not assume a fixed layout or template. Infer the document's own visual convention from headers, legends, rows, cells, symbols, annotations, grouping, and spatial relationships.
For every identifier matching No. followed by digits, independently return CC, SC, or null.
Only classify when visible document evidence supports the relationship. Do not copy a classification from a nearby row. A general CC/SC heading alone is not row-level evidence. Use null when evidence is absent, conflicting, or unclear.
Keep evidence_text and interpretation short. Return JSON only and exactly match the schema.
"""

USER_PROMPT = """/no_think
Infer the document convention, then extract every No.<digits> item.
For each item return:
- dimension_number
- classification: CC, SC, or null
- evidence_text: exact short visible evidence, maximum 20 words
- interpretation: concise reason the evidence applies to this identifier, maximum 25 words
- confidence from 0 to 1
- needs_review
Do not return bounding boxes. Do not omit unclassified No.<digits> items. Return JSON only.
"""


def check_server(base_url: str, model: str, timeout: int) -> None:
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {base_url}. Start Ollama, then retry. Details: {exc}"
        ) from exc

    names = {item.get("name") for item in response.json().get("models", [])}
    if model not in names and not any(name and name.split(":")[0] == model for name in names):
        raise RuntimeError(
            f"Model '{model}' is not installed. Run: ollama pull {model}\n"
            f"Installed models: {', '.join(sorted(names)) or 'none'}"
        )


def normalize_no(value: object) -> str | None:
    """Normalize common model renderings while preserving the No.<digits> invariant."""
    text = str(value or "").strip()
    match = NO_RE.search(text)
    if match:
        return f"No.{int(match.group(1))}"

    # Models sometimes return only the digits even when the image visibly says No.<digits>.
    # Accept only a standalone integer, not arbitrary numbers embedded in other text.
    bare = re.fullmatch(r"\s*(\d+)\s*", text)
    if bare:
        return f"No.{int(bare.group(1))}"
    return None


def preprocess(image: Image.Image, max_side: int = 2600) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    denoised = cv2.fastNlMeansDenoising(gray, None, 7, 7, 21)
    enhanced = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)

    h, w = enhanced.shape[:2]
    scale = min(max_side / max(h, w), 2.5)
    if abs(scale - 1.0) > 0.05:
        enhanced = cv2.resize(
            enhanced, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        )
    return Image.fromarray(enhanced)


def content_crop(image: Image.Image, padding: int = 30) -> tuple[Image.Image, tuple[int, int]]:
    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    mask = gray < 245
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return image, (0, 0)
    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(image.width, int(xs.max()) + padding)
    y2 = min(image.height, int(ys.max()) + padding)
    return image.crop((x1, y1, x2, y2)), (x1, y1)


def make_tiles(image: Image.Image, tile_size: int = 1500, overlap: int = 250):
    """Yield (tile, x_offset, y_offset, label), including a whole-image view."""
    yield image, 0, 0, "full"
    if image.width <= tile_size and image.height <= tile_size:
        return
    step = max(1, tile_size - overlap)
    xs = list(range(0, max(1, image.width - tile_size + 1), step))
    ys = list(range(0, max(1, image.height - tile_size + 1), step))
    last_x = max(0, image.width - tile_size)
    last_y = max(0, image.height - tile_size)
    if not xs or xs[-1] != last_x:
        xs.append(last_x)
    if not ys or ys[-1] != last_y:
        ys.append(last_y)
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            tile = image.crop((x, y, min(x + tile_size, image.width), min(y + tile_size, image.height)))
            yield tile, x, y, f"tile_{x}_{y}"


def image_b64(image: Image.Image) -> str:
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


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


def _normalize_compact_output(raw_output: str) -> ExtractionResponse:
    json_text = _extract_complete_json(raw_output)
    try:
        return ExtractionResponse.model_validate_json(json_text)
    except ValidationError:
        data = json.loads(json_text)
        rows = data.get("results", []) if isinstance(data, dict) else []
        normalized = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            dim = (row.get("dimension_number") or row.get("dim_no") or
                   row.get("dimension_no") or row.get("no") or "")
            raw_tag = (row.get("classification") or row.get("tag_type") or
                       row.get("sc_cc") or row.get("tag") or row.get("type"))
            tag_match = re.search(r"(?i)\b(CC|SC)\b", str(raw_tag or ""))
            classification = tag_match.group(1).upper() if tag_match else None
            evidence_text = (row.get("evidence_text") or row.get("tag_text") or
                             row.get("evidence") or (str(raw_tag) if raw_tag else None))
            if isinstance(evidence_text, (list, dict)):
                evidence_text = json.dumps(evidence_text, ensure_ascii=False)[:300]
            try:
                confidence = max(0.0, min(1.0, float(row.get("confidence", 0.70))))
            except (TypeError, ValueError):
                confidence = 0.0
            needs_review = row.get("needs_review")
            if needs_review is None:
                needs_review = classification is None and bool(evidence_text)
            normalized.append({
                "dimension_number": str(dim),
                "classification": classification,
                "evidence_text": str(evidence_text) if evidence_text is not None else None,
                "interpretation": str(row.get("interpretation") or row.get("association_basis") or "Document-inferred visual relationship"),
                "confidence": confidence,
                "needs_review": bool(needs_review),
            })
        return ExtractionResponse.model_validate({
            "document_convention": data.get("document_convention") if isinstance(data, dict) else None,
            "results": normalized,
        })



def _extract_explicit_labels(text: str | None) -> set[str]:
    """Extract explicit CC/SC values while ignoring generic paired labels.

    A phrase such as SC/CC names the classification category and is not a
    conflict. Bracketed values such as [SC] or [CC] are stronger item evidence.
    """
    value = str(text or "").strip()
    if not value:
        return set()

    bracketed = {
        match.upper()
        for match in re.findall(r"(?i)[\[\(]\s*(CC|SC)\s*[\]\)]", value)
    }
    if bracketed:
        return bracketed

    # Remove category/header expressions before looking for standalone labels.
    contextual = re.sub(r"(?i)\bSC\s*/\s*CC\b|\bCC\s*/\s*SC\b", " ", value)
    return {
        match.upper()
        for match in re.findall(r"(?i)\b(CC|SC)\b", contextual)
    }


def _reconcile_semantics(result: ExtractionResponse) -> ExtractionResponse:
    """Reconcile model fields using evidence priority, without layout rules.

    Priority:
    1. A valid structured classification from the model.
    2. An unambiguous explicit value in evidence_text.
    3. An unambiguous explicit value in interpretation after generic SC/CC
       category phrases are removed.
    """
    for item in result.results:
        evidence_labels = _extract_explicit_labels(item.evidence_text)
        interpretation_labels = _extract_explicit_labels(item.interpretation)

        if item.classification in {"CC", "SC"}:
            # Keep the structured value unless direct evidence explicitly
            # contradicts it.
            if evidence_labels and item.classification not in evidence_labels:
                item.classification = None
                item.needs_review = True
                item.interpretation = (
                    f"Structured classification conflicts with explicit evidence "
                    f"{sorted(evidence_labels)}. " + item.interpretation
                )
            continue

        if len(evidence_labels) == 1:
            item.classification = next(iter(evidence_labels))  # type: ignore[assignment]
            item.needs_review = False
        elif len(evidence_labels) > 1:
            item.classification = None
            item.needs_review = True
            item.interpretation = (
                f"Conflicting explicit labels in item evidence: "
                f"{sorted(evidence_labels)}. " + item.interpretation
            )
        elif len(interpretation_labels) == 1:
            item.classification = next(iter(interpretation_labels))  # type: ignore[assignment]
            item.needs_review = False
        elif len(interpretation_labels) > 1:
            item.classification = None
            item.needs_review = True
            item.interpretation = (
                f"Conflicting explicit labels in interpretation: "
                f"{sorted(interpretation_labels)}. " + item.interpretation
            )
        else:
            item.classification = None
            if item.confidence >= 0.90:
                item.needs_review = True

    return result


def _call_minimal_fallback(
    image: Image.Image,
    model: str,
    base_url: str,
    timeout: int,
) -> ExtractionResponse:
    """Second extraction pass used only when the primary pass returns zero rows."""
    minimal_prompt = """/no_think
Read the complete engineering table image. Find every identifier whose visible text starts with No. and is followed by digits.
For every such identifier, infer whether the same document visually classifies it as CC, SC, or neither. Do not assume a fixed layout. Use the document's headers, row structure, legends, symbols, and annotations.
Return JSON only in exactly this form:
{"results":[{"dimension_number":"No.123","classification":"SC","evidence_text":"short visible evidence"}]}
classification must be "CC", "SC", or null. The results array must contain every visible No.<digits> identifier and must not be empty when such identifiers are visible.
"""
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "format": MinimalResponse.model_json_schema(),
        "messages": [{"role": "user", "content": minimal_prompt, "images": [image_b64(image)]}],
        "options": {"temperature": 0, "seed": 117, "num_ctx": 4096, "num_predict": 2048},
        "keep_alive": "10m",
    }
    response = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    message = body.get("message", {})
    content = (message.get("content") or "").strip()
    thinking = (message.get("thinking") or body.get("thinking") or "").strip()
    raw_output = content or thinking
    if not raw_output:
        raise RuntimeError("Minimal fallback returned no content or thinking output.")
    minimal = MinimalResponse.model_validate_json(_extract_complete_json(raw_output))
    if not minimal.results:
        raise RuntimeError("Minimal fallback also returned an empty results array.")

    rows = []
    for item in minimal.results:
        evidence = item.evidence_text
        rows.append({
            "dimension_number": item.dimension_number,
            "classification": item.classification,
            "evidence_text": evidence,
            "interpretation": "Classification inferred from the document's visual convention.",
            "confidence": 0.80 if item.classification else 0.60,
            "needs_review": item.classification is None,
        })
    result = ExtractionResponse.model_validate({
        "document_convention": "Recovered through minimal extraction pass.",
        "results": rows,
    })
    return _reconcile_semantics(result)


def call_ollama(
    image: Image.Image,
    model: str,
    base_url: str,
    timeout: int,
    retries: int,
) -> ExtractionResponse:
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "format": RESPONSE_SCHEMA,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT, "images": [image_b64(image)]},
        ],
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": 4096,
            "num_predict": 3072,
        },
        "keep_alive": "10m",
    }
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
            response.raise_for_status()
            body = response.json()
            message = body.get("message", {})
            content = (message.get("content") or "").strip()
            thinking = (message.get("thinking") or body.get("thinking") or "").strip()
            raw_output = content or thinking
            output_field = "message.content" if content else "message.thinking"
            if not raw_output:
                raise RuntimeError(
                    "Ollama returned neither content nor thinking output. "
                    f"done_reason={body.get('done_reason')}, eval_count={body.get('eval_count')}"
                )
            result = _normalize_compact_output(raw_output)
            result = _reconcile_semantics(result)
            if not result.results:
                print(
                    "    Primary response was valid JSON but contained zero results; "
                    "running minimal recovery pass...",
                    file=sys.stderr,
                )
                result = _call_minimal_fallback(
                    image, model, base_url, timeout
                )
                print("    Parsed Ollama output from minimal recovery pass")
                return result
            print(f"    Parsed Ollama output from {output_field}")
            return result
        except (requests.RequestException, KeyError, json.JSONDecodeError,
                ValidationError, RuntimeError) as exc:
            last_error = exc
            if attempt < retries:
                print(f"    Attempt {attempt + 1} failed: {exc}; retrying...", file=sys.stderr)
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Ollama extraction failed after {retries + 1} attempt(s): {last_error}")

def map_bbox(box: list[int] | None, tile: Image.Image, xoff: int, yoff: int):
    if not box:
        return None
    x1, y1, x2, y2 = [max(0, min(1000, int(v))) for v in box]
    return [
        round(xoff + x1 * tile.width / 1000),
        round(yoff + y1 * tile.height / 1000),
        round(xoff + x2 * tile.width / 1000),
        round(yoff + y2 * tile.height / 1000),
    ]


def consolidate(candidates: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for item in candidates:
        normalized = normalize_no(item.get("dimension_number", ""))
        if not normalized:
            continue
        item["dimension_number"] = normalized
        grouped.setdefault(normalized, []).append(item)

    final = []
    for _, items in grouped.items():
        items.sort(key=lambda item: item.get("confidence", 0.0), reverse=True)
        best = items[0]
        labels = {item.get("classification") for item in items
                  if item.get("classification") in {"CC", "SC"}}
        if len(labels) > 1:
            best["classification"] = None
            best["needs_review"] = True
            best["interpretation"] = f"Conflicting classifications: {sorted(labels)}" # type: ignore
        best["evidence_count"] = len(items)
        final.append(best)

    def sort_key(item):
        match = NO_RE.search(item["dimension_number"])
        return int(match.group(1)) if match else 10**9
    return sorted(final, key=sort_key)


def annotate(image: Image.Image, detections: list[dict], output_path: Path) -> None:
    """Save the processed image. Version 2.1 omits model-generated boxes for reliability."""
    canvas = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), canvas)


def process_file(path: Path, args) -> dict:
    original = Image.open(path).convert("RGB")
    processed = preprocess(original, args.max_side)
    cropped, _ = content_crop(processed)

    print(f"  Calling Ollama for full ({cropped.width}x{cropped.height})")
    response = call_ollama(
        cropped, args.model, args.ollama_url, args.timeout, args.retries
    )
    candidates = []
    for detection in response.results:
        item = detection.model_dump()
        item["source_crop"] = "full"
        candidates.append(item)

    results = consolidate(candidates)
    print(
        f"    Model rows: {len(response.results)}; "
        f"rows after identifier normalization: {len(results)}"
    )

    # A model may return non-empty rows with malformed or bare identifiers.
    # If every row is rejected during normalization, recover rather than silently
    # writing an empty successful document.
    if response.results and not results:
        print(
            "    Primary rows were dropped during identifier normalization; "
            "running minimal recovery pass...",
            file=sys.stderr,
        )
        recovered = _call_minimal_fallback(
            cropped, args.model, args.ollama_url, args.timeout
        )
        recovered_candidates = []
        for detection in recovered.results:
            item = detection.model_dump()
            item["source_crop"] = "minimal_recovery"
            recovered_candidates.append(item)
        results = consolidate(recovered_candidates)
        response.document_convention = recovered.document_convention
        print(f"    Recovery rows after normalization: {len(results)}")

    if not results:
        raise RuntimeError(
            "No valid No.<digits> results remained after extraction and recovery. "
            "The model response was not task-valid."
        )

    return {
        "document_id": path.stem,
        "source_file": str(path),
        "model": args.model,
        "document_convention": response.document_convention,
        "results": results,
        "unresolved": [x for x in results if x.get("needs_review")],
    }, processed # type: ignore


def input_files(input_path: Path):
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED:
            raise ValueError(f"Unsupported image type: {input_path.suffix}")
        return [input_path]
    return sorted(p for p in input_path.rglob("*") if p.suffix.lower() in SUPPORTED)


def main() -> int:
    print(f"CC/SC Extractor version: {SCRIPT_VERSION}")
    parser = argparse.ArgumentParser(description="Extract No.<digits> to CC/SC mappings using local Ollama.")
    parser.add_argument("input", type=Path, help="Image file or directory")
    parser.add_argument("--output", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--model", default="qwen3-vl:2b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--tile-size", type=int, default=1500)
    parser.add_argument("--overlap", type=int, default=250)
    parser.add_argument("--max-side", type=int, default=2600)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    check_server(args.ollama_url.rstrip("/"), args.model, 10)
    args.ollama_url = args.ollama_url.rstrip("/")

    files = input_files(args.input)
    if not files:
        print("No supported images found.", file=sys.stderr)
        return 2

    all_documents = []
    for path in files:
        print(f"Processing: {path}")
        try:
            document, processed = process_file(path, args)
            all_documents.append(document)
            json_path = args.output / f"{path.stem}_cc_sc.json"
            annotated_path = args.output / f"{path.stem}_annotated.png"
            json_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
            annotate(processed, document["results"], annotated_path)
            print(f"  Saved: {json_path}")
            print(f"  Saved: {annotated_path}")
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            all_documents.append({"source_file": str(path), "error": str(exc)})

    summary = args.output / "all_results.json"
    summary.write_text(json.dumps({"documents": all_documents}, indent=2), encoding="utf-8")
    print(f"Summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())