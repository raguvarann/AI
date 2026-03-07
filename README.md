# AI/ML Examples & Experimentation Suite

**About**: Collection of production-ready Python scripts and model artifacts demonstrating end-to-end machine learning workflows—from data preprocessing to model deployment—with focus on NLP, computer vision (OCR), predictive analytics, and retrieval-augmented generation (RAG) systems.

## Technology Stack

| Category | Technologies |
|----------|------------------|
| **Deep Learning** | TensorFlow/Keras 2.13+, PyTorch 2.0+ |
| **NLP & Transformers** | Hugging Face Transformers 4.30+, spaCy 3.6+ (NER, POS, dependency parsing) |
| **ML Algorithms** | scikit-learn 1.3+, ensemble methods, SVM, clustering |
| **Document Processing** | PyMuPDF (fitz), python-docx, pdfplumber, pytesseract (Tesseract OCR) |
| **Computer Vision** | Pillow 10.0+, OpenCV-compatible image processing |
| **LLM & RAG** | LangChain 0.1+, Ollama, custom RAG pipelines |
| **Cloud & APIs** | Google Vertex AI 1.33+, REST integrations |
| **Data Processing** | Pandas 1.5+, NumPy 1.24+ |
| **Web Framework** | Streamlit 1.28+ (rapid prototyping) |

## Project Structure

```
AI/
├── requirements.txt                    # All dependencies with versions
├── models/
│   ├── ragustructured/                 # spaCy NER/NLP model (trained on domain data)
│   └── raguunstructured/               # Transformer-based unstructured text model
├── tf_env/                             # Pre-configured TensorFlow virtual environment
│
├── [NLP & Text Processing]
├── nlp.py                              # spaCy pipeline demos (POS, NER, dependency parsing)
├── Structured.py                       # Structured information extraction
├── Unstructured.py                     # Unstructured text + BERT-based NER (dslim/bert-base-NER)
├── modelnlp.py                         # Domain-specific NLP utilities
│
├── [Document Analysis & OCR]
├── ocr.py                              # Tesseract OCR + Pillow image processing
├── doc classification.py                # LSTM-based document classification (multi-class)
├── Upload.py                           # Document upload handler
│
├── [Predictive Modeling]
├── Predictive.py                       # scikit-learn classification/regression pipelines
├── predictive_userinteraction.py       # Interactive ML model demos
│
├── [Retrieval-Augmented Generation]
├── raguapp.py                          # Streamlit web UI for document analysis
├── raguwrapper.py                      # RAG pipeline coordinator
├── ragumodelimport.py                  # Import/export RAG models
├── ragumodelmerge.py                   # Merge/combine multiple RAG models
│
├── [LLM & Cloud Integration]
├── simple_llm.py                       # LangChain + Ollama local LLM integration
├── simple_vertex.py                    # Google Vertex AI client initialization
│
└── Huggingface/                        # Docker deployment + Streamlit UI
    ├── Dockerfile                      # Container image definition
    ├── requirements.txt                # Streamlit-optimized dependencies
    └── src/
        ├── raguapp.py                  # Web interface for RAG analysis
        └── raguwrapper.py              # RAG pipeline backend
```

## Installation & Setup

### Option 1: Create Fresh Virtual Environment

```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
```

### Option 2: Use Pre-configured TensorFlow Environment

```powershell
# Activate existing tf_env
.tf_env\Scripts\Activate.ps1

# Install any missing packages
pip install -r requirements.txt
```

### Additional Setup for OCR

Ensure Tesseract-OCR is installed:
- **Windows**: Download from [Tesseract OCR releases](https://github.com/UB-Mannheim/tesseract/wiki)
- Update pytesseract path if needed: `pytesseract.pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'`

## Module Descriptions

### NLP & Text Processing

- **`nlp.py`**: spaCy pipeline examples
  - Named Entity Recognition (NER)
  - Part-of-speech (POS) tagging
  - Dependency parsing
  - Custom pipeline components

- **`Structured.py`**: Structured information extraction from semi-structured text
  - Pattern matching
  - Entity relationship extraction

- **`Unstructured.py`**: Transformer-based NER using `dslim/bert-base-NER`
  - BERT token classification
  - Multi-label entity extraction with confidence scores
  - Context-aware entity aggregation

### Document Analysis & OCR

- **`ocr.py`**: End-to-end OCR pipeline
  - Image preprocessing (grayscale, contrast, thresholding)
  - Tesseract OCR integration
  - Post-processing and confidence scoring

- **`doc classification.py`**: Multi-class document classification
  - LSTM neural network architecture
  - Tokenization & embedding layers
  - Training/validation with Keras

### Predictive Modeling

- **`Predictive.py`**: scikit-learn ML pipelines
  - Classification: Logistic Regression, Random Forest, SVM
  - Regression: Linear, Ridge, Gradient Boosting
  - Cross-validation & hyperparameter tuning

- **`predictive_userinteraction.py`**: Interactive demo
  - Real-time model predictions
  - Feature importance visualization
  - Model comparison utilities

### Retrieval-Augmented Generation (RAG)

- **`raguapp.py`**: Streamlit web interface
  - PDF/DOCX document upload
  - Document parsing (PyMuPDF, python-docx)
  - Entity extraction & analysis
  - CSV/TXT export functionality

- **`raguwrapper.py`**: Main RAG pipeline
  - Coordinates structured + unstructured extraction
  - Merges results from multiple models
  - Confidence scoring & filtering

- **`ragumodelimport.py`** / **`ragumodelmerge.py`**: Model utilities
  - Import custom spaCy/HF models
  - Combine multiple model outputs

### LLM & Cloud Integration

- **`simple_llm.py`**: LangChain + Ollama integration
  - Local LLM inference (no API calls)
  - Prompt chaining
  - Document question-answering

- **`simple_vertex.py`**: Google Vertex AI initialization
  - Cloud ML model inference
  - Project/location configuration

## Model Artifacts

The `models/` directory contains pre-trained models:

### `ragustructured/` (spaCy Model)
- **Format**: spaCy v3+ binary model
- **Components**: 
  - `tok2vec`: Token-to-vector encoder
  - `tagger`: POS tagger
  - `parser`: Dependency parser
  - `ner`: Named Entity Recognition
  - `entity_ruler`: Rule-based entity patterns
  - `lemmatizer`: Lemmatization with lookups
- **Size**: ~40MB
- **Usage**: Structured data extraction with rule-based + ML components

### `raguunstructured/` (Hugging Face Model)
- **Format**: BERT-compatible tokenizer + model weights
- **Type**: Token classification for NER
- **Size**: ~350MB (model.safetensors)
- **Usage**: Transformer-based NER on free-form text

## Running Examples

```powershell
# Activate environment
.tf_env\Scripts\Activate.ps1

# NLP Examples
python nlp.py                           # spaCy pipeline demo
python Unstructured.py                  # BERT-based NER

# OCR Example
python ocr.py                           # Tesseract OCR on sample images

# Predictive Modeling
python Predictive.py                    # scikit-learn classification

# Launch Streamlit Web UI (RAG)
streamlit run raguapp.py                # Access at http://localhost:8501

# LLM Integration
python simple_llm.py                    # LangChain + Ollama demo
        (requires Ollama running locally)
```

## Docker Deployment

```bash
# Build Docker image (Huggingface/)
cd Huggingface/
docker build -t ragu-analysis:latest .

# Run Streamlit container
docker run -p 7860:7860 ragu-analysis:latest
```

## Dependencies Breakdown

| Library | Purpose | Version |
|---------|---------|---------|
| `torch` | PyTorch tensors & DL | 2.0+ |
| `tensorflow` | Keras neural networks | 2.13+ |
| `transformers` | BERT, NER, seq2seq models | 4.30+ |
| `spacy` | NLP pipeline, production-grade | 3.6+ |
| `scikit-learn` | Classical ML algorithms | 1.3+ |
| `pandas` | Data manipulation | 1.5+ |
| `numpy` | Numerical arrays | 1.24+ |
| `Pillow` | Image processing | 10.0+ |
| `pytesseract` | OCR wrapper | 0.3.10+ |
| `PyMuPDF` (fitz) | PDF text extraction | 1.23+ |
| `python-docx` | DOCX parsing | 0.8.11+ |
| `pdfplumber` | Advanced PDF extraction | 0.9+ |
| `langchain` | LLM orchestration framework | 0.1+ |
| `ollama` | Local LLM inference | 0.1+ |
| `google-cloud-aiplatform` | Vertex AI SDK | 1.33+ |
| `streamlit` | Interactive web framework | 1.28+ |

## Performance Considerations

- **GPU Acceleration**: TensorFlow/PyTorch utilize CUDA 11.8+ for GPU inference (~5-10x speedup)
- **Model Size**: spaCy model ~40MB, Transformer models 300-500MB
- **Memory**: Recommend 8GB+ RAM for concurrent model loading
- **OCR**: Tesseract performs best on high-DPI (200+ DPI) grayscale documents
- **RAG Pipeline**: Optimize document chunking for token limits (~512 tokens per chunk)

## Troubleshooting

1. **Tesseract not found**: Install Tesseract-OCR separately; update system PATH
2. **spaCy model errors**: Run `python -m spacy download en_core_web_sm`
3. **CUDA/GPU issues**: Verify CUDA 11.8+ is installed; ensure GPU drivers are current
4. **Memory overflow**: Reduce batch size or use model quantization
5. **Streamlit timeout**: Increase `client.maxMessageSize` in `.streamlit/config.toml`

## License & Attribution

This collection experiments with publicly available models:
- spaCy models: Apache 2.0
- Hugging Face Transformers: Apache 2.0
- Tesseract: Apache 2.0

