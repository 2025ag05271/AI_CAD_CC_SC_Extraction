# CC/SC Extraction from Engineering Drawings

This project extracts CC/SC classifications for dimensions such as `No.1503`, `No.1504`, etc., from engineering drawing images using a local Ollama vision model.

The current implementation is in `src/extract_cc_sc_ollama_v200.py`. It reads one or more images, preprocesses them, sends them to Ollama for structured extraction, normalizes the results, and saves a JSON report plus an annotated image.

## What the project does

- Accepts image files or a folder of images
- Searches each image for identifiers matching `No.<digits>`
- Uses a local vision model to infer whether each identifier is associated with `CC`, `SC`, or no clear classification
- Normalizes model output into a clean JSON structure
- Saves:
  - per-document results JSON
  - combined summary JSON
  - annotated output image

## Project structure

```text
CC_SC_Extraction/
├── input_images/              # input engineering drawing images
├── output/                    # generated JSON and annotated images
│   ├── all_results.json
│   ├── cc_sc_test_1_cc_sc.json
│   └── cc_sc_test_1_annotated.png
├── src/
│   ├── extract_cc_sc_ollama_v200.py   # current extractor
│   └── extract_cc_sc_ollama.py        # earlier/legacy version
├── tests/
│   ├── smoke_test.py
│   └── test_ollama.py
├── README.md
└── .git/
```

## Example output

The output JSON is structured like this:

```json
{
  "document_id": "cc_sc_test_1",
  "document_convention": null,
  "results": [
    {
      "dimension_number": "No.1503",
      "classification": "SC",
      "evidence_text": "[SC]",
      "interpretation": "The row for No. 1503 contains [SC] in the SC/CC column.",
      "confidence": 1.0,
      "needs_review": false,
      "source_crop": "full",
      "evidence_count": 1
    }
  ],
  "unresolved": []
}
```

`all_results.json` contains all processed documents in a single summary object.

## Prerequisites

Before running the extractor, make sure the following are available:

1. Python 3.11+
2. A working Ollama installation locally on the machine
3. A supported vision model pulled in Ollama, such as `qwen3-vl:8b`
4. Required Python packages installed

### Install Python dependencies

From the project root run:

```powershell
python -m pip install opencv-python numpy pillow pydantic requests
```

If you are using the same environment as the project setup in this workspace, you may already have it activated.

## Step-by-step: run the project

### 1. Start Ollama

Make sure Ollama is running on the default local endpoint:

```powershell
ollama serve
```

Then pull the model used by the script:

```powershell
ollama pull qwen3-vl:8b
```

### 2. Activate your environment

For the workspace environment shown here:

```powershell
conda activate cc_sc_extraction
```

### 3. Place your images in the input folder

Put the engineering drawing files in `input_images/`.

Examples:

```text
input_images/
├── cc_sc_test_1.jpg
├── cc_sc_test_2.png
└── ...
```

### 4. Run the extractor

Use the current script:

```powershell
python src\extract_cc_sc_ollama_v200.py input_images --output output --model qwen3-vl:8b --timeout 300 --retries 1
```

You can also target a single file:

```powershell
python src\extract_cc_sc_ollama_v200.py input_images\cc_sc_test_1.jpg --output output --model qwen3-vl:8b
```

### 5. Review the generated output

The script will create files in `output/`, including:

- `output/all_results.json` — overall summary for all processed documents
- `output/<image_name>_cc_sc.json` — per-image extraction result
- `output/<image_name>_annotated.png` — processed image with the extraction output saved

## Usage notes

- The default script argument is `--model qwen3-vl:8b`.
- The script expects Ollama to be available at `http://localhost:11434`.
- This project is designed for local, offline use and does not require cloud APIs.
- If a model response is incomplete, empty, or malformed, the code includes fallback recovery logic to keep processing going.

## Troubleshooting

### Ollama connection errors

If you see errors about not reaching Ollama:

```powershell
ollama serve
```

Then verify the endpoint is available:

```powershell
http://localhost:11434
```

### Model not installed

If the model is missing:

```powershell
ollama pull qwen3-vl:8b
```

### No files in input folder

The script only processes supported image extensions:

- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`
- `.tif`
- `.tiff`
- `.webp`

## Notes on the implementation

The current script in `src/extract_cc_sc_ollama_v200.py` includes:

- image preprocessing and content cropping
- normalization of identifiers such as `NO 1503` or `No.1503`
- structured JSON schema validation
- model retry logic
- evidence-based reconciliation of ambiguous outputs
- saving a clean document-level summary and annotated image output

The earlier script in `src/extract_cc_sc_ollama.py` is a legacy implementation and is not the primary version to use unless you are specifically testing the older workflow.

## Quick start summary

```powershell
conda activate cc_sc_extraction
ollama serve
ollama pull qwen3-vl:8b
python src\extract_cc_sc_ollama_v200.py input_images --output output --model qwen3-vl:8b
```

This will process the images in `input_images/` and save the results under `output/`.
