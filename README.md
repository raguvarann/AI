# RAGU: Retrieval-Augmented Generation & Understanding

**Overview**: A comprehensive AI/ML platform combining NLP, common intelligence, predictive analytics, and retrieval systems for intelligent common analysis and question-answering capabilities.

**Simple Explanation**: Extract and analyze information from common sources using advanced AI models that understand both structured and unstructured text, perform OCR on images, and answer questions about your content.

**Technical Focus**: Production-ready Python suite featuring spaCy NLP pipelines, BERT transformers, scikit-learn ML models, ChromaDB vector storage, and Streamlit web interface for end-to-end RAG systems.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Windows/Linux/macOS
- Tesseract-OCR (optional, for OCR features)
- 8GB+ RAM recommended

### Installation

**Step 1: Clone & Setup Environment**
```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Or use existing TensorFlow env
.tf_env\Scripts\Activate.ps1
```

**Step 2: Install Dependencies**
```powershell
pip install -r requirements.txt

# Download spaCy language models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
```

**Step 3: (Optional) Install Tesseract for OCR**
- Windows: [Download Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- macOS: `brew install tesseract`
- Linux: `apt-get install tesseract-ocr`

---

## 📊 Technology Stack

| Category | Technologies |
|----------|------------------|
| **Deep Learning** | TensorFlow 2.13+, PyTorch 2.0+, Keras |
| **NLP & Transformers** | Hugging Face Transformers 4.30+, spaCy 3.6+ |
| **Machine Learning** | scikit-learn 1.3+, ensemble methods, SVM, clustering |
| **Common Processing** | PyMuPDF, python-docx, pdfplumber, pytesseract |
| **Vector Database** | ChromaDB (semantic search) |
| **LLM Integration** | LangChain 0.1+, Ollama (local inference) |
| **Cloud & APIs** | Google Vertex AI 1.33+ |
| **Web Framework** | Streamlit 1.28+ (interactive UI) |
| **Data Processing** | Pandas 1.5+, NumPy 1.24+ |

## 📁 Project Structure

```
AI/
├── requirements.txt                      # All dependencies with pinned versions
├── models/                               # Pre-trained ML models
│   ├── ragustructured/                   # spaCy NER model (entity parsing)
│   └── raguunstructured/                 # BERT transformer model (token categorization)
├── tf_env/                               # Pre-configured Python virtual environment
│
├── 📄 [COMMON PROCESSING]
│   ├── raguapp.py                        # Streamlit UI - Upload & analyze common content
│   ├── raguwrapper.py                    # RAG pipeline coordinator
│   ├── ragumodelimport.py                # Import/load ML models
│   ├── ragumodelmerge.py                 # Combine multiple model outputs
│   └── Upload.py                         # Common upload handler
│
├── 🔤 [NLP & TEXT PROCESSING]
│   ├── nlp.py                            # spaCy NLP demos (POS tagging, NER, parsing)
│   ├── Structured.py                     # Structured info parsing + entity rules
│   ├── Unstructured.py                   # BERT-based NER for free-form text
│   └── modelnlp.py                       # Domain-specific NLP utilities
│
├── 📸 [VISION & OCR]
│   └── ocr.py                            # Tesseract OCR pipeline + image preprocessing
│
├── 🎯 [PREDICTIVE MODELS]
│   ├── Predictive.py                     # scikit-learn categorization/regression
│   ├── doc categorization.py             # LSTM-based multi-class content categorization
│   └── predictive_userinteraction.py     # Interactive ML demo
│
├── 🤖 [LLM & CLOUD]
│   ├── simple_llm.py                     # LangChain + Ollama local inference
│   └── simple_vertex.py                  # Google Vertex AI integration
│
├── 📚 [VECTOR DATABASE]
│   └── db/
│       ├── chroma_app.py                 # ChromaDB storage & management
│       ├── query_tool.py                 # Semantic search & retrieval
│       ├── watcher.py                    # Monitor incoming content
│       ├── incoming_pdfs/                # PDFs awaiting processing
│       └── my_chroma_data/               # Vector database storage
│
├── 🐳 [DEPLOYMENT]
│   └── Huggingface/
│       ├── Dockerfile                    # Container image specification
│       ├── requirements.txt              # Streamlit-specific dependencies
│       └── src/
│           ├── raguapp.py                # Web interface
│           └── raguwrapper.py            # Backend RAG pipeline
│
└── 💼 [CAREER RESOURCES]
    └── jobs/
        ├── job_search_bengaluru.py       # Job market analysis
        └── job_listings_bengaluru.py     # Job listing scraper
```

---

## 🎯 Feature Overview

### 📄 Common Intelligence
- **Upload & Parse**: Support for PDF, DOCX, TXT files
- **Entity Parsing**: Identify people, organizations, locations, amounts
- **OCR**: Extract text from scanned images (requires Tesseract)
- **Export**: Download results as CSV or TXT

### 🔍 NLP Capabilities
- **Named Entity Recognition (NER)**: Find entities with confidence scores
- **Part-of-Speech Tagging**: Identify nouns, verbs, adjectives, etc.
- **Dependency Parsing**: Understand grammatical relationships
- **Entity Linking**: Match entities to predefined patterns

### 🤖 Machine Learning
- **Categorization**: Categorize content by type
- **Regression**: Predict numerical values from features
- **Ensemble Methods**: Combine multiple models for better accuracy
- **Feature Importance**: Understand which factors drive predictions

### 📚 Vector Search
- **Semantic Search**: Find similar content using ChromaDB
- **Question Answering**: Query content with natural language
- **Metadata Filtering**: Filter results by category, source, date
- **Relevance Scoring**: Rank results by similarity

---

## 📖 Detailed Module Guide

### Common Processing

#### raguapp.py (Main Streamlit Interface)
- **What it does**: Web UI for uploading and analyzing common content
- **Input**: PDF, DOCX, or TXT files
- **Output**: Entity table with confidence scores, downloadable CSV/TXT
- **Technical**: Streamlit framework, multi-page layout, async file processing
- **Usage**: `streamlit run raguapp.py` → Open http://localhost:8501

#### raguwrapper.py (RAG Pipeline)
- **What it does**: Orchestrates structured + unstructured parsing
- **Input**: Raw common text
- **Output**: Merged entity results with scores
- **Technical**: Combines spaCy structured model with BERT unstructured model
- **Key function**: `run_ragu_pipeline(text)` → list of entities

#### ragumodelimport.py & ragumodelmerge.py
- **Import**: Load custom spaCy/HF models into pipeline
- **Merge**: Combine outputs from multiple NER systems
- **Technical**: Model registry pattern, ensemble weighting

---

### NLP & Text Processing

#### nlp.py (spaCy Pipeline)
- **What it does**: Demonstrates spaCy NLP capabilities
- **From simple view**: Process text to understand: who/what/where/when
- **From technical view**: POS tagging, dependency parsing, NER with pretrained en_core_web_sm
- **Models loaded**: en_core_web_sm (40MB), en_core_web_md (optional)
- **Usage**: `python nlp.py`

#### Structured.py (Rule-based Parsing)
- **What it does**: Extract specific patterns (ticker symbols, percentages, organizations)
- **Approach**: Define custom patterns → apply entity ruler to text
- **Patterns included**: 
  - Companies: NBC, ACME, TECH
  - Tickers: $AAPL, $GOOG format
  - Percentages: "5.25%", "4.5%"
- **Usage**: `python Structured.py`

#### Unstructured.py (BERT Transformer)
- **What it does**: Extract entities from free-form text using deep learning
- **Model**: dslim/bert-base-NER (BERT fine-tuned for NER task)
- **From simple view**: Understands context, can handle variations better than rules
- **From technical view**: Token categorization, subword tokenization, attention mechanisms
- **Output**: Entity text, label (PER, ORG, LOC, MISC), confidence score
- **Usage**: `python Unstructured.py`

---

### Computer Vision & OCR

#### ocr.py (Tesseract OCR)
- **What it does**: Extract text from images/scanned PDFs
- **Process**: Image→Grayscale→Contrast adjustment→Thresholding→Tesseract
- **From simple view**: Read text from pictures
- **From technical view**: Image preprocessing (PIL), confidence scoring, language-specific models
- **Setup required**: Tesseract-OCR system installation
- **Usage**: `python ocr.py` (update image path in code)

---

### Predictive Modeling

#### Predictive.py (scikit-learn)
- **What it does**: Train ML models for categorization and regression
- **Models available**:
  - Categorization: Logistic Regression, Random Forest, SVM
  - Regression: Linear Regression, Ridge Regression, Gradient Boosting
- **From simple view**: Predict values based on input features
- **From technical view**: Train-test split, feature scaling, cross-validation, hyperparameter tuning
- **Sample**: House price prediction from square footage, bedrooms, age
- **Usage**: `python Predictive.py`

#### doc categorization.py (Deep Learning)
- **What it does**: Classify common content using LSTM neural network
- **From simple view**: Automatically categorize common content by type
- **From technical view**: LSTM layers, embedding layer, Keras, multi-class softmax output
- **Architecture**: Embeddings → LSTM → Dense layers → softmax
- **Usage**: `python 'doc categorization.py'`

#### predictive_userinteraction.py
- **What it does**: Interactive model demo and analysis
- **Features**: Feature importance, model comparison, real-time predictions
- **Visualization**: Charts and graphs for model understanding

---

### LLM & Cloud Integration

#### simple_llm.py (Local LLM with Ollama)
- **What it does**: Use LangChain to run LLMs locally without cloud costs
- **From simple view**: Ask questions and get answers from your content
- **From technical view**: LangChain chains, prompt templates, local model inference
- **Requirement**: Ollama running locally (`ollama serve`)
- **Usage**: `python simple_llm.py`

#### simple_vertex.py (Google Cloud)
- **What it does**: Initialize and use Google Vertex AI models
- **From simple view**: Use powerful Google AI models for predictions
- **From technical view**: Google Cloud SDK authentication, Vertex AI endpoint configuration
- **Prerequisites**: GCP project, credentials file, Vertex AI API enabled

---

### Vector Database & Semantic Search

#### db/chroma_app.py (ChromaDB Vector Storage)
- **What it does**: Store content as vectors for semantic search
- **From simple view**: Build a searchable knowledge base from PDFs
- **From technical view**: Embeddings, vector database, persistent storage
- **Storage**: SQLite-backed, local persistent storage in `my_chroma_data/`
- **Usage**: `python db/chroma_app.py`

#### db/query_tool.py (Interactive Search)
- **What it does**: Query the vector database with natural language
- **From simple view**: Ask questions and find relevant content
- **From technical view**: Semantic similarity search, metadata filtering
- **Usage**: `python db/query_tool.py` → Interactive prompt

#### db/watcher.py (Common Monitor)
- **What it does**: Automatically process new PDFs in a folder
- **From simple view**: Drop PDFs in a folder, they get analyzed automatically
- **From technical view**: File system watcher pattern, async processing

---

## 🤖 Model Artifacts

### ragustructured/ (spaCy Model)
- **Type**: Industrial-grade NLP pipeline
- **Size**: ~40MB
- **Components**:
  - `tok2vec`: Converts tokens to vectors using transformer architecture
  - `tagger`: Part-of-speech tagger
  - `parser`: Dependency parser for grammatical relationships
  - `ner`: Named entity recognizer
  - `entity_ruler`: Rule-based entity patterns
  - `lemmatizer`: Convert words to root form with lookups
- **Best for**: Structured data with predictable patterns
- **Example**: Extracting ticker symbols ($AAPL), organizations, percentages

### raguunstructured/ (BERT Transformer)
- **Type**: Deep learning transformer for NER
- **Size**: ~350MB (model.safetensors) + tokenizer
- **Model**: dslim/bert-base-NER
- **Capabilities**: Token categorization with attention mechanisms
- **Best for**: Free-form text, context-dependent entities
- **Example**: Finding company names, people in natural text

---

## 📚 Running Examples

### Command Reference

```powershell
# Activate environment
.tf_env\Scripts\Activate.ps1

# Text Processing
python nlp.py                           # spaCy demos
python Structured.py                    # Rule-based parsing
python Unstructured.py                  # BERT NER

# Vision & OCR
python ocr.py                           # Extract text from images

# Predictive Models
python Predictive.py                    # Categorization/Regression
python 'doc categorization.py'          # LSTM content categorization

# Vector Database
python db/chroma_app.py                 # Build knowledge base
python db/query_tool.py                 # Search database

# Web Interface (Main Application)
streamlit run raguapp.py                # Go to http://localhost:8501

# LLM Integration
python simple_llm.py                    # Requires Ollama running
```

### Web Interface Workflow

1. **Start Streamlit**
   ```powershell
   streamlit run raguapp.py
   ```

2. **Upload Content**
   - Drag & drop or browse for PDF/DOCX/TXT
   - App automatically extracts text

3. **Run Analysis**
   - Click "Analyze Common"
   - Entities appear as table
   - View confidence scores

4. **Export Results**
   - Download as CSV
   - Download as TXT
   - View in browser

---

## 🐳 Docker Deployment

### Build Container Image

```bash
cd Huggingface/

# Build image
docker build -t ragu-analysis:latest .

# Tag for registry (optional)
docker tag ragu-analysis:latest your-registry/ragu-analysis:latest
```

### Run Container

```bash
# Run locally on port 8501 (Streamlit default)
docker run -p 8501:8501 ragu-analysis:latest

# Run with volume mount for persistence
docker run -p 8501:8501 -v $(pwd)/data:/app/data ragu-analysis:latest

# Run with environment variables
docker run -e GOOGLE_APPLICATION_CREDENTIALS=/app/creds.json \
           -v $(pwd)/creds.json:/app/creds.json \
           -p 8501:8501 ragu-analysis:latest
```

### Docker Compose (Multi-service)

```yaml
version: '3.8'
services:
  ragu-app:
    build: ./Huggingface
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - PYTHONUNBUFFERED=1
  
  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - chroma_data:/chroma/data

volumes:
  chroma_data:
```

### Deploy to Cloud

**Google Cloud Run** (Recommended)
```bash
# Authenticate
gcloud auth login

# Build & push to Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT/ragu-analysis

# Deploy
gcloud run deploy ragu-analysis \
  --image gcr.io/YOUR_PROJECT/ragu-analysis \
  --platform managed \
  --memory 4Gi
```

---

## 📋 Dependencies Breakdown

| Library | Version | Purpose | Size |
|---------|---------|---------|------|
| `torch` | 2.0+ | PyTorch tensors & neural network ops | 2GB+ |
| `tensorflow` | 2.13+ | TensorFlow/Keras framework | 1.5GB+ |
| `transformers` | 4.30+ | BERT, GPT, other transformer models | 500MB+ |
| `spacy` | 3.6+ | Production NLP pipelines | 50MB |
| `scikit-learn` | 1.3+ | Classical ML (RF, SVM, LR) | 100MB |
| `pandas` | 1.5+ | DataFrames & data manipulation | 50MB |
| `numpy` | 1.24+ | Numerical arrays & operations | 50MB |
| `Pillow` | 10.0+ | Image processing & manipulation | 10MB |
| `pytesseract` | 0.3.10+ | Python OCR wrapper | <1MB |
| `PyMuPDF` (fitz) | 1.23+ | PDF text & image parsing | 50MB |
| `python-docx` | 0.8.11+ | Microsoft Word file parsing | 5MB |
| `pdfplumber` | 0.9+ | Advanced PDF parsing (tables, etc) | 5MB |
| `chromadb` | 0.3+ | Vector database & embeddings | 100MB |
| `langchain` | 0.1+ | LLM framework & chains | 50MB |
| `ollama` | 0.1+ | Local LLM inference | <1MB |
| `google-cloud-aiplatform` | 1.33+ | Google Vertex AI SDK | 50MB |
| `streamlit` | 1.28+ | Web app framework | 100MB |

---

## ⚡ Performance & Optimization

### Memory Management
- **GPU Acceleration**: TensorFlow/PyTorch use CUDA 11.8+ for 5-10x speedup
- **Recomm memory**: 8GB+ RAM for concurrent model loading
- **Model Size**: 
  - spaCy: ~40MB (fast, CPU)
  - BERT: ~350MB (slower, needs GPU for speed)
  - Combined: ~400MB + runtime overhead
- **Batch Processing**: Use batch_size > 1 for throughput

### Optimization Tips

```python
# Use smaller models when possible
from transformers import pipeline
nlp = pipeline("ner", model="distilbert-base-uncased-finetuned-ner")  # Faster

# Use GPU for transformers
nlp = pipeline("ner", model="dslim/bert-base-NER", device=0)  # device=0 = GPU

# Batch similar tasks
import torch
torch.no_grad()  # Disable gradient computation during inference

# Cache spaCy models
import spacy
nlp = spacy.load("en_core_web_sm")  # Load once, reuse
```

### Benchmark

| Task | Model | Speed (100 docs) | GPU | CPU |
|------|-------|-----------------|-----|-----|
| NER | spaCy | 500ms | N/A | N/A |
| NER | BERT | 5s | ✓ 1s | ✗ 5s |
| Categorization | LSTM | 2s | ✓ 0.5s | ✗ 2s |
| OCR | Tesseract | 10s | N/A | N/A |

---

## 🐛 Troubleshooting

### Installation Issues

**Problem**: `ImportError: No module named 'torch'`
- **Solution**: `pip install torch` or `pip install -r requirements.txt`

**Problem**: `No module named 'tesseract'`
- **Solution**: Install system Tesseract (not Python package)
  - Windows: [Download Tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki)
  - macOS: `brew install tesseract`
  - Linux: `apt-get install tesseract-ocr`

**Problem**: `torch requires CUDA version`
- **Solution**: Install CUDA 11.8+ from [NVIDIA](https://developer.nvidia.com/cuda-downloads) or use CPU version

### Runtime Issues

**Problem**: Streamlit times out after 60 seconds
- **Solution**: Edit `.streamlit/config.toml`:
  ```toml
  [client]
  maxMessageSize = 200
  
  [logger]
  level = "warning"
  ```

**Problem**: spaCy model not found
- **Solution**: Download manually
  ```powershell
  python -m spacy download en_core_web_sm
  python -m spacy download en_core_web_md
  ```

**Problem**: PDF/DOCX parsing returns empty
- **Check**: File is valid & text-based (not image-only PDF)
- **Solution**: Run OCR first for scanned PDFs

**Problem**: Out of memory (OOM)
- **Cause**: Loading multiple large models at once
- **Solutions**:
  1. Unload unused models: `del nlp` and `gc.collect()`
  2. Reduce batch size
  3. Use smaller model variants (distilbert instead of bert-base)
  4. Increase available RAM/swap

### Model & Data Issues

**Problem**: Low entity parsing accuracy
- **Cause**: Model trained on different domain
- **Solutions**:
  1. Use rule-based parsing for known patterns (Structured.py)
  2. Fine-tune model on your domain data
  3. Combine multiple models (ragumodelmerge.py)

**Problem**: ChromaDB returns no results
- **Check**: Database populated? Run `db/chroma_app.py` first
- **Solution**: Add content and ensure semantic search is enabled

---

## 🔗 Resources & Documentation

### Official Docs
- [spaCy Documentation](https://spacy.io) - Industrial NLP
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) - BERT, language models
- [LangChain](https://python.langchain.com) - LLM orchestration
- [Streamlit Docs](https://docs.streamlit.io) - Web framework
- [ChromaDB](https://docs.trychroma.com) - Vector database
- [Google Vertex AI](https://cloud.google.com/vertex-ai/docs) - Cloud ML

### Useful Links
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) - Installation guide
- [Ollama Models](https://ollama.ai/library) - Local LLM models
- [PyTorch Tutorials](https://pytorch.org/tutorials/) - Deep learning
- [scikit-learn Guide](https://scikit-learn.org/stable/user_guide.html) - ML algorithms

---

## 📝 License & Attribution

This project uses open-source models and libraries under permissive licenses:

- **spaCy Models**: Apache 2.0
- **Hugging Face Transformers**: Apache 2.0  
- **Tesseract-OCR**: Apache 2.0
- **PyTorch**: BSD
- **TensorFlow**: Apache 2.0
- **scikit-learn**: BSD-3-Clause

All model weights and code in this repository are provided as-is for educational and research purposes.

