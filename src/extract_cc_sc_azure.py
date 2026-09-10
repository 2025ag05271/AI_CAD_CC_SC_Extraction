"""Extract No.<digits> to CC/SC associations using an Azure ML hosted Qwen3-VL endpoint.

Sends OpenAI-compatible chat completion requests to
https://<endpoint>/v1/chat/completions with a bearer token read from .env.
Images are inlined as base64 data URLs, so no public image hosting is needed.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
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
SCRIPT_VERSION = "3.0.0-azure-endpoint"

SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

DEFAULT_ENDPOINT = "https://qwen3-6-27b-ep.westeurope.inference.ml.azure.com/v1/chat/completions"


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


SYSTEM_PROMPT = """You extract CC/SC classifications from engineering document images.
Do not assume a fixed layout or template. Infer the document's own visual convention from headers, legends, rows, cells, symbols, annotations, grouping, and spatial relationships.
For every identifier matching No. followed by digits, independently return CC, SC, or null.
Only classify when visible document evidence supports the relationship. Do not copy a classification from a nearby row. A general CC/SC heading alone is not row-level evidence. Use null when evidence is absent, conflicting, or unclear.
Keep evidence_text and interpretation short. Return JSON only, with no markdown fences and no commentary.
The JSON must have this exact shape:
{"document_convention":"short description","results":[{"dimension_number":"No.123","classification":"SC","evidence_text":"short visible evidence","interpretation":"short reason","confidence":0.9,"needs_review":false}]}
"""

USER_PROMPT = """Infer the document convention, then extract every No.<digits> item from this image.
For each item return:
- dimension_number
- classification: "CC", "SC", or null
- evidence_text: exact short visible evidence, maximum 20 words
- interpretation: concise reason the evidence applies to this identifier, maximum 25 words
- confidence from 0 to 1
- needs_review
Do not return bounding boxes. Do not omit unclassified No.<digits> items. Return JSON only.
"""

MINIMAL_PROMPT = """Read the complete engineering table image. Find every identifier whose visible text starts with No. and is followed by digits.
For every such identifier, infer whether the same document visually classifies it as CC, SC, or neither. Do not assume a fixed layout. Use the document's headers, row structure, legends, symbols, and annotations.
Return JSON only in exactly this form:
{"results":[{"dimension_number":"No.123","classification":"SC","evidence_text":"short visible evidence"}]}
classification must be "CC", "SC", or null. The results array must contain every visible No.<digits> identifier and must not be empty when such identifiers are visible.
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


def normalize_no(value: object) -> str | None:
    """Normalize common model renderings while preserving the No.<digits> invariant."""
    text = str(value or "").strip()
    match = NO_RE.search(text)
    if match:
        return f"No.{int(match.group(1))}"

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


def image_data_url(image: Image.Image) -> str:
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


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
    """Extract explicit CC/SC values while ignoring generic paired labels such as SC/CC."""
    value = str(text or "").strip()
    if not value:
        return set()

    bracketed = {
        match.upper()
        for match in re.findall(r"(?i)[\[\(]\s*(CC|SC)\s*[\]\)]", value)
    }
    if bracketed:
        return bracketed

    contextual = re.sub(r"(?i)\bSC\s*/\s*CC\b|\bCC\s*/\s*SC\b", " ", value)
    return {
        match.upper()
        for match in re.findall(r"(?i)\b(CC|SC)\b", contextual)
    }


def _reconcile_semantics(result: ExtractionResponse) -> ExtractionResponse:
    """Reconcile model fields using evidence priority, without layout rules."""
    for item in result.results:
        evidence_labels = _extract_explicit_labels(item.evidence_text)
        interpretation_labels = _extract_explicit_labels(item.interpretation)

        if item.classification in {"CC", "SC"}:
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


def _build_payload(
    image: Image.Image,
    text_prompt: str,
    args,
    system_prompt: str | None,
) -> dict:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_data_url(image)}},
            {"type": "text", "text": text_prompt},
        ],
    })
    return {
        "model": args.model,
        "messages": messages,
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
    response = requests.post(args.endpoint, json=payload, headers=headers, timeout=args.timeout)
    if response.status_code == 401:
        raise RuntimeError("Endpoint rejected the bearer token (401). Check AZURE_ENDPOINT_TOKEN in .env.")
    response.raise_for_status()
    body = response.json()
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
        response = requests.post(args.endpoint, json=probe, headers=headers, timeout=30)
    except requests.RequestException as exc:
        raise RuntimeError(f"Cannot reach endpoint {args.endpoint}. Details: {exc}") from exc
    if response.status_code == 401:
        raise RuntimeError("Endpoint rejected the bearer token (401). Check AZURE_ENDPOINT_TOKEN in .env.")
    if response.status_code >= 400:
        raise RuntimeError(
            f"Endpoint health check failed with HTTP {response.status_code}: {response.text[:300]}"
        )


def _call_minimal_fallback(image: Image.Image, args) -> ExtractionResponse:
    """Second extraction pass used only when the primary pass returns zero usable rows."""
    payload = _build_payload(image, MINIMAL_PROMPT, args, system_prompt=None)
    raw_output = _post_chat(payload, args)
    minimal = MinimalResponse.model_validate_json(_extract_complete_json(raw_output))
    if not minimal.results:
        raise RuntimeError("Minimal fallback also returned an empty results array.")

    rows = []
    for item in minimal.results:
        rows.append({
            "dimension_number": item.dimension_number,
            "classification": item.classification,
            "evidence_text": item.evidence_text,
            "interpretation": "Classification inferred from the document's visual convention.",
            "confidence": 0.80 if item.classification else 0.60,
            "needs_review": item.classification is None,
        })
    result = ExtractionResponse.model_validate({
        "document_convention": "Recovered through minimal extraction pass.",
        "results": rows,
    })
    return _reconcile_semantics(result)


def call_endpoint(image: Image.Image, args) -> ExtractionResponse:
    payload = _build_payload(image, USER_PROMPT, args, SYSTEM_PROMPT)
    last_error = None
    for attempt in range(args.retries + 1):
        try:
            raw_output = _post_chat(payload, args)
            result = _reconcile_semantics(_normalize_compact_output(raw_output))
            if not result.results:
                print(
                    "    Primary response was valid JSON but contained zero results; "
                    "running minimal recovery pass...",
                    file=sys.stderr,
                )
                result = _call_minimal_fallback(image, args)
                print("    Parsed endpoint output from minimal recovery pass")
                return result
            print("    Parsed endpoint output from choices[0].message.content")
            return result
        except (requests.RequestException, KeyError, json.JSONDecodeError,
                ValidationError, RuntimeError) as exc:
            last_error = exc
            if attempt < args.retries:
                print(f"    Attempt {attempt + 1} failed: {exc}; retrying...", file=sys.stderr)
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Endpoint extraction failed after {args.retries + 1} attempt(s): {last_error}")


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


def annotate(image: Image.Image, output_path: Path) -> None:
    canvas = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), canvas)


def process_file(path: Path, args):
    original = Image.open(path).convert("RGB")
    processed = preprocess(original, args.max_side)
    cropped, _ = content_crop(processed)

    print(f"  Calling endpoint for full ({cropped.width}x{cropped.height})")
    response = call_endpoint(cropped, args)

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

    if response.results and not results:
        print(
            "    Primary rows were dropped during identifier normalization; "
            "running minimal recovery pass...",
            file=sys.stderr,
        )
        recovered = _call_minimal_fallback(cropped, args)
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

    document = {
        "document_id": path.stem,
        "source_file": str(path),
        "endpoint": args.endpoint,
        "model": args.model,
        "document_convention": response.document_convention,
        "results": results,
        "unresolved": [x for x in results if x.get("needs_review")],
    }
    return document, processed


def input_files(input_path: Path):
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED:
            raise ValueError(f"Unsupported image type: {input_path.suffix}")
        return [input_path]
    return sorted(p for p in input_path.rglob("*") if p.suffix.lower() in SUPPORTED)


def main() -> int:
    print(f"CC/SC Extractor (Azure endpoint) version: {SCRIPT_VERSION}")
    parser = argparse.ArgumentParser(
        description="Extract No.<digits> to CC/SC mappings using a hosted Qwen3-VL endpoint."
    )
    parser.add_argument("input", type=Path, help="Image file or directory")
    parser.add_argument("--output", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--endpoint", default=os.environ.get("AZURE_ENDPOINT_URL") or DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=os.environ.get("AZURE_ENDPOINT_MODEL"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-side", type=int, default=2600)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--skip-health-check", action="store_true")
    args = parser.parse_args()

    load_env(args.env_file)
    args.endpoint = os.environ.get("AZURE_ENDPOINT_URL", args.endpoint)
    args.model = os.environ.get("AZURE_ENDPOINT_MODEL", args.model)
    args.token = resolve_token()

    args.output.mkdir(parents=True, exist_ok=True)
    if not args.skip_health_check:
        check_endpoint(args)

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
            annotate(processed, annotated_path)
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
