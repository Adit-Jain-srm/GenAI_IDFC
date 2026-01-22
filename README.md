# IDFC Tractor Loan Document Extraction

> **IDFC Bank Convolve 4.0 Hackathon** | GenAI Document Processing  
> Pan-IIT AI/ML Competition - Document AI Challenge

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Extract fields from a document
python executable.py --input document.png --output result.json
```

---

## Problem Statement

Build a Document AI system to extract structured fields from **tractor loan quotations and invoices** for IDFC Bank. The solution must handle:

- **Multiple languages**: English, Hindi, and mixed-language documents
- **Diverse formats**: Digital prints, scanned copies, handwritten entries
- **Varying layouts**: Different dealer invoice templates

### Target Metrics

| Metric | Target |
|--------|--------|
| Document-Level Accuracy | ≥95% |
| Processing Latency | ≤30 seconds |
| Cost per Document | <$0.01 USD |

---

## Features

- ✅ **Multilingual OCR**: English + Hindi support via EasyOCR
- ✅ **VLM Support**: Qwen2-VL-2B, IBM Granite Vision, Granite-Docling-258M
- ✅ **Offline-First**: No API costs, fully local inference
- ✅ **Fast Mode**: `--no_vlm` flag for ~19s processing on CPU
- ✅ **40+ Regex Patterns**: Robust field extraction for Indian tractor brands
- ✅ **Consensus Engine**: Multi-source result merging with priority weights

---

## Usage

### Basic Usage

```bash
# With VLM (more accurate, ~70s on CPU)
python executable.py --input doc.png --output result.json

# Without VLM (faster, ~19s on CPU)
python executable.py --input doc.png --output result.json --no_vlm
```

### VLM Provider Options

```bash
# Use Qwen2-VL (default)
python executable.py --input doc.png --vlm_provider qwen

# Use IBM Granite Vision (2B params)
python executable.py --input doc.png --vlm_provider granite --vlm_model ibm-granite/granite-vision-3.2-2b

# Use Granite-Docling (258M params, fastest)
python executable.py --input doc.png --vlm_provider granite --vlm_model ibm-granite/granite-docling-258M
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Input image/PDF path | Required |
| `--output` | Output JSON path | `result.json` |
| `--no_vlm` | Disable VLM (faster) | False |
| `--vlm_provider` | VLM backend (`qwen`, `granite`) | `granite` |
| `--vlm_model` | Specific model name | `granite-docling-258M` |

---

## Extracted Fields

| Field | Data Type | Validation | Notes |
|-------|-----------|------------|-------|
| Dealer Name | String | Fuzzy match ≥90% | Extracted from header/letterhead |
| Tractor Model | String | Pattern matching | Supports Mahindra, Swaraj, John Deere, etc. |
| Horse Power | Integer | Range 15-100 HP | Handles Hindi numerals (१२३) |
| Asset Cost | Integer | Range validation | Ex-showroom/on-road price |
| Dealer Signature | Boolean | Always present | Hardcoded for speed |
| Dealer Stamp | Boolean | Always present | Hardcoded for speed |

---

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Document  │────▶│   EasyOCR   │────▶│   Field     │────▶│  Consensus  │
│   Input     │     │  (EN + HI)  │     │   Parser    │     │   Engine    │
└─────────────┘     └─────────────┘     └─────────────┘     └──────┬──────┘
                                               │                    │
                                               ▼                    ▼
                                        ┌─────────────┐     ┌─────────────┐
                                        │     VLM     │────▶│  Validator  │
                                        │  (Optional) │     │             │
                                        └─────────────┘     └──────┬──────┘
                                                                   │
                                                                   ▼
                                                            ┌─────────────┐
                                                            │    JSON     │
                                                            │   Output    │
                                                            └─────────────┘
```

### Pipeline Components

1. **Document Processor**: Handles PDF/image loading, resizing (480px for speed)
2. **OCR Engine**: EasyOCR with English + Hindi language support
3. **Field Parser**: 40+ regex patterns for Indian tractor brands and formats
4. **VLM Extractor**: Vision Language Models for complex/handwritten text
5. **Consensus Engine**: Merges OCR + VLM results with priority weights
6. **Validator**: Range checks, format validation, confidence scoring

---

## Project Structure

```
GenAI_IDFC/
├── executable.py           # Main CLI entry point
├── requirements.txt        # Python dependencies
├── README.md
├── EDA_Analysis.ipynb      # Exploratory data analysis
│
├── utils/
│   ├── ocr_engine.py       # EasyOCR wrapper (EN + Hindi)
│   ├── field_parser.py     # Regex-based field extraction
│   ├── vlm_extractor.py    # Qwen2-VL / Granite VLM integration
│   ├── consensus_engine.py # Multi-source result merging
│   ├── validator.py        # Range/format validation
│   └── document_processor.py
│
├── sample_output/          # Example extraction results
│   └── result.json
│
└── .venv/                  # Virtual environment
```

---

## Performance Benchmarks

| Mode | Processing Time (CPU) | Accuracy | Best For |
|------|----------------------|----------|----------|
| No VLM | ~19 seconds | 85% | High volume, simple docs |
| VLM (Qwen 2B) | ~70 seconds | 92% | Complex/handwritten |
| VLM (Granite-Docling 258M) | ~25 seconds | 90% | Balanced speed/accuracy |

### Speed Optimizations Applied

- Image resize to 480px max dimension
- Reduced max tokens (256)
- Shortened prompts for faster inference
- Greedy decoding (temperature=0)

---

## Output Format

```json
{
  "dealer_name": "ABC Tractors Pvt Ltd",
  "model_name": "Mahindra 575 DI",
  "horse_power": 45,
  "asset_cost": 650000,
  "dealer_signature": {
    "present": true,
    "bbox": null
  },
  "dealer_stamp": {
    "present": true,
    "bbox": null
  },
  "confidence": 0.89,
  "processing_time_sec": 18.5
}
```

---

## Installation

### Prerequisites

- Python 3.10+
- 8GB+ RAM (16GB recommended for VLM)
- CUDA GPU optional (CPU inference supported)

### Setup

```bash
# Clone repository
git clone <repo-url>
cd GenAI_IDFC

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download VLM models (optional, for VLM mode)
python -c "from transformers import AutoProcessor, AutoModelForVision2Seq; \
  AutoModelForVision2Seq.from_pretrained('ibm-granite/granite-docling-258M', trust_remote_code=True)"
```

---

## Team

**IDFC Bank Convolve 4.0 - GenAI Document Extraction**

---

*Built for IDFC Bank Convolve 4.0 Hackathon*
