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
SCRIPT_VERSION = "2.0.0-generic-evidence"

SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class Evidence(BaseModel):
    visible_text: str | None = None
    semantic_role: str
    relationship: str
    bbox: list[int] | None = Field(default=None, min_length=4, max_length=4)


class Detection(BaseModel):
    dimension_number: str
    classification: Literal["CC", "SC"] | None = None
    dimension_bbox: list[int] | None = Field(default=None, min_length=4, max_length=4)
    interpretation: str
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    needs_review: bool
    reason: str | None = None


class ExtractionResponse(BaseModel):
    document_structure_summary: str | None = None
    results: list[Detection] = Field(default_factory=list)


RESPONSE_SCHEMA = ExtractionResponse.model_json_schema()

SYSTEM_PROMPT = """You are an evidence-grounded visual information extraction system for engineering documents.

Your task is not to match a fixed template. First infer the document's own visual grammar: regions, tables, rows, columns, headers, legends, symbols, repeated conventions, merged cells, leader lines, and spatial groupings. Then independently evaluate every identifier matching No. followed by digits.

For each identifier, determine whether the document visually classifies it as CC, SC, or neither. A classification is valid only when supported by visible evidence and a defensible spatial or semantic relationship. The evidence may be expressed directly, indirectly through a header or legend, through a marker, through an embedded annotation, or through another document-specific convention. Do not assume any particular layout, marker, column position, or wording.

Core constraints:
- Infer meaning from the current document rather than from a memorized template.
- Evaluate each No.<digits> item independently.
- Do not transfer a classification from a nearby item without explicit visual evidence.
- A generic header mentioning CC/SC is context, not by itself an item classification.
- Absence of evidence means classification null, not a guess.
- Conflicting or unclear evidence means classification null and needs_review true.
- Preserve short visible evidence text and describe why each evidence item applies.
- Bounding boxes use normalized 0..1000 coordinates relative to the supplied image.
- Return JSON matching the schema only.
"""

USER_PROMPT = """Analyze the complete engineering document using an infer-then-extract approach.

Step 1: Briefly summarize the document structure and the visual convention used, if any, to express CC or SC classifications.
Step 2: Find every visible identifier matching No. followed by digits.
Step 3: For each identifier, independently decide CC, SC, or null from visible evidence.
Step 4: Return evidence items. Each evidence item must state:
- visible_text: exact short text or symbol when readable
- semantic_role: what the item represents in this document
- relationship: why it applies to this dimension identifier
- bbox: normalized coordinates when possible

Use interpretation to explain the association without relying on a fixed pattern name. Set needs_review true whenever evidence is incomplete, conflicting, or uncertain. Return valid JSON only.
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


def normalize_no(value: str) -> str | None:
    match = NO_RE.search(value or "")
    return f"No.{int(match.group(1))}" if match else None


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
            "num_predict": 2048,
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
                    "Ollama returned neither message.content nor message.thinking. "
                    f"done_reason={body.get('done_reason')}, eval_count={body.get('eval_count')}"
                )
            print(f"    Parsed Ollama output from {output_field}")
            try:
                return ExtractionResponse.model_validate_json(raw_output)
            except ValidationError:
                data = json.loads(raw_output)
                rows = data.get("results", []) if isinstance(data, dict) else []
                normalized = []
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    dim = (row.get("dimension_number") or row.get("dim_no") or
                           row.get("dimension_no") or row.get("no") or "")
                    raw_tag = (row.get("classification") or row.get("tag_type") or
                               row.get("sc_cc") or row.get("tag") or row.get("type"))
                    match = re.search(r"(?i)\b(CC|SC)\b", str(raw_tag or ""))
                    classification = match.group(1).upper() if match else None
                    raw_evidence = row.get("evidence", [])
                    if not isinstance(raw_evidence, list):
                        raw_evidence = []
                    evidence = []
                    for item in raw_evidence:
                        if isinstance(item, dict):
                            evidence.append({
                                "visible_text": item.get("visible_text") or item.get("text"),
                                "semantic_role": item.get("semantic_role") or item.get("role") or "visual evidence",
                                "relationship": item.get("relationship") or item.get("basis") or "associated by document structure",
                                "bbox": item.get("bbox"),
                            })
                    if not evidence and (row.get("tag_text") or raw_tag):
                        evidence.append({
                            "visible_text": row.get("tag_text") or str(raw_tag),
                            "semantic_role": "classification evidence",
                            "relationship": row.get("association_basis") or "associated by document structure",
                            "bbox": row.get("tag_bbox"),
                        })
                    try:
                        confidence = max(0.0, min(1.0, float(row.get("confidence", 0.70))))
                    except (TypeError, ValueError):
                        confidence = 0.0
                    needs_review = row.get("needs_review")
                    if needs_review is None:
                        needs_review = classification is None and bool(evidence)
                    normalized.append({
                        "dimension_number": str(dim),
                        "classification": classification,
                        "dimension_bbox": row.get("dimension_bbox"),
                        "interpretation": row.get("interpretation") or row.get("association_basis") or "Model-inferred document relationship",
                        "evidence": evidence,
                        "confidence": confidence,
                        "needs_review": bool(needs_review),
                        "reason": row.get("reason"),
                    })
                return ExtractionResponse.model_validate({
                    "document_structure_summary": data.get("document_structure_summary") if isinstance(data, dict) else None,
                    "results": normalized,
                })

           
        except (requests.RequestException, KeyError, json.JSONDecodeError, ValidationError, RuntimeError) as exc:
            last_error = exc
            if attempt < retries:
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
    for dimension_number, items in grouped.items():
        labels = {
            item.get("classification") for item in items
            if item.get("classification") in {"CC", "SC"}
        }
        items.sort(key=lambda item: item.get("confidence", 0.0), reverse=True)
        best = items[0]
        all_evidence = []
        seen = set()
        for item in items:
            for evidence in item.get("evidence", []):
                key = (
                    evidence.get("visible_text"), evidence.get("semantic_role"),
                    evidence.get("relationship"), str(evidence.get("bbox"))
                )
                if key not in seen:
                    seen.add(key)
                    all_evidence.append(evidence)
        best["evidence"] = all_evidence
        best["evidence_count"] = len(all_evidence)
        if len(labels) > 1:
            best["classification"] = None
            best["needs_review"] = True
            best["reason"] = f"Conflicting classifications across evidence: {sorted(labels)}" # type: ignore
        final.append(best)

    def sort_key(item):
        match = NO_RE.search(item["dimension_number"])
        return int(match.group(1)) if match else 10**9
    return sorted(final, key=sort_key)


def annotate(image: Image.Image, detections: list[dict], output_path: Path) -> None:
    canvas = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    for item in detections:
        color = (0, 0, 255) if item.get("needs_review") else (0, 160, 0)
        boxes = [item.get("dimension_bbox")]
        boxes.extend(e.get("bbox") for e in item.get("evidence", []))
        for box in boxes:
            if box and len(box) == 4:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        box = item.get("dimension_bbox")
        if box:
            label = f'{item["dimension_number"]} -> {item.get("classification") or "NONE/REVIEW"} ({item.get("confidence", 0):.2f})'
            cv2.putText(canvas, label, (int(box[0]), max(20, int(box[1]) - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.imwrite(str(output_path), canvas)


def process_file(path: Path, args) -> dict:
    original = Image.open(path).convert("RGB")
    processed = preprocess(original, args.max_side)
    cropped, crop_offset = content_crop(processed)

    candidates = []
    for tile, xoff, yoff, label in [(cropped, 0, 0, "full")]:
        print(f"  Calling Ollama for {label} ({tile.width}x{tile.height})")
        response = call_ollama(tile, args.model, args.ollama_url, args.timeout, args.retries)
        for detection in response.results:
            item = detection.model_dump()
            item["dimension_bbox"] = map_bbox(item["dimension_bbox"], tile, xoff + crop_offset[0], yoff + crop_offset[1])
            for evidence in item.get("evidence", []):
                evidence["bbox"] = map_bbox(
                    evidence.get("bbox"), tile,
                    xoff + crop_offset[0], yoff + crop_offset[1]
                )
            item["source_crop"] = label
            candidates.append(item)

    results = consolidate(candidates)
    return {
        "document_id": path.stem,
        "source_file": str(path),
        "model": args.model,
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