"""Identify datum labels attached to geometric elements in engineering drawing images.

Sends OpenAI-compatible chat completion requests to an Azure ML hosted Qwen3-VL
endpoint. No image preprocessing is applied; each image is read and sent as-is.
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
from typing import Literal

import requests
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

SCRIPT_VERSION = "1.0.0-datum-description"

SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

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


class DatumDetection(BaseModel):
    datum_label: str
    geometric_element: str
    explanation: str
    interpretation: str


class DatumExtractionResponse(BaseModel):
    image_id: str | None = None
    results: list[DatumDetection] = Field(default_factory=list)


SYSTEM_PROMPT = """You read engineering drawing images and describe datum feature symbols.
A datum feature symbol is a boxed letter, typically attached to a surface, edge, axis, or feature by a leader line or a triangular datum feature indicator.
The only legal datum labels in this system are A, B, and C, but you must not judge or decide whether a label in the image is correct, legal, or appropriately applied. Only describe what is visible.
For every datum feature symbol visible in the image, report:
- datum_label: the exact letter shown inside the datum symbol, uppercase, as visible in the image
- geometric_element: a concise description of the geometric element the datum symbol is attached to, such as a specific surface, edge, hole axis, or feature, based on its position and any nearby dimension or feature callouts
- explanation: what is visually connecting the datum symbol to that geometric element, such as a leader line, triangular flag, or direct attachment
- interpretation: a short, neutral restatement of what the datum symbol denotes for that element
Do not state whether the datum label is valid, legal, or correctly used. Only describe. Return JSON only, with no markdown fences and no commentary.
The JSON must have this exact shape:
{"image_id":"file name without extension","results":[{"datum_label":"A","geometric_element":"short description","explanation":"short description","interpretation":"short description"}]}
"""

USER_PROMPT = """Look at this engineering drawing image and find every datum feature symbol.
For each one, report datum_label, geometric_element, explanation, and interpretation as described.
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
            label = str(row.get("datum_label") or row.get("label") or "").strip().upper()
            normalized.append({
                "datum_label": label,
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
    return {
        "image_id": path.stem,
        "source_file": str(path),
        "endpoint": args.endpoint,
        "model": args.model,
        "legal_datum_labels": sorted(LEGAL_DATUM_LABELS),
        "results": results,
    }


def input_files(input_path: Path):
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED:
            raise ValueError(f"Unsupported image type: {input_path.suffix}")
        return [input_path]
    return sorted(p for p in input_path.rglob("*") if p.suffix.lower() in SUPPORTED)


def main() -> int:
    run_start = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Describe datum feature symbols and their attached geometric elements using a hosted Qwen3-VL endpoint."
    )
    parser.add_argument("input", type=Path, help="Image file or directory")
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
    args = parser.parse_args()

    log_path = setup_logging(args.log_dir)
    LOGGER.info("Datum Extractor (Azure endpoint) version: %s", SCRIPT_VERSION)
    LOGGER.info("log file: %s", log_path)
    LOGGER.info("input: %s | output: %s", args.input, args.output)

    with step("load_env_and_token", str(args.env_file)):
        load_env(args.env_file)
        args.endpoint = os.environ.get("AZURE_ENDPOINT_URL", args.endpoint)
        args.model = os.environ.get("AZURE_ENDPOINT_MODEL", args.model)
        args.token = resolve_token()

    args.output.mkdir(parents=True, exist_ok=True)
    if not args.skip_health_check:
        check_endpoint(args)

    with step("discover_input_files", str(args.input)) as marker:
        files = input_files(args.input)
        marker.detail = f"{len(files)} file(s)"
    if not files:
        LOGGER.error("no supported images found")
        return 2

    all_documents = []
    for index, path in enumerate(files, start=1):
        LOGGER.info("-" * 78)
        LOGGER.info("processing file %d/%d: %s", index, len(files), path)
        try:
            with step("process_file_total", str(path)) as file_marker:
                document = process_file(path, args)
                all_documents.append(document)
                json_path = args.output / f"{path.stem}_datum.json"
                with step("write_document_json", str(json_path)):
                    json_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
                file_marker.detail = f"{len(document['results'])} datum symbol(s)"
        except Exception as exc:
            LOGGER.error("%s | %s", path, exc)
            all_documents.append({"source_file": str(path), "error": str(exc)})

    summary = args.output / "all_datum_results.json"
    with step("write_summary_json", str(summary)):
        summary.write_text(json.dumps({"documents": all_documents}, indent=2), encoding="utf-8")

    log_timing_summary(time.perf_counter() - run_start)
    LOGGER.info("timed log written to %s", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
