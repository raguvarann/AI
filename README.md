About: I am an AI enthusiast with hands-on experience building machine learning and deep learning solutions focused on predictive modeling, computer vision, and natural language processing. I experiment with real-world datasets, optimize models, and build deployable AI systems that extract actionable insights.

Small collection of Python scripts demonstrating ML, NLP, OCR, and LLM integrations.

Core technologies: Python, NumPy, Pandas, scikit-learn, TensorFlow/Keras, Hugging Face Transformers, spaCy, pytesseract (Tesseract OCR), Pillow, LangChain (Ollama), Google Vertex AI client.
 
Quick start:

# AI Examples (small collection)

## Overview

This repository contains a collection of small Python scripts and model artifacts that demonstrate common AI tasks: natural language processing (spaCy, Hugging Face), OCR (pytesseract), simple predictive modeling (scikit-learn / TensorFlow), and basic LLM / Vertex AI examples.

## Quick start

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Notes

- This repo includes a `models/` folder with prebuilt spaCy and RAG model files used by several scripts.
- A Python virtual environment `tf_env/` is present in the workspace; you can use it instead of creating a new one if desired.

## Common scripts (quick descriptions)

- `nlp.py`: small spaCy examples and pipeline usage.
- `Unstructured.py`: uses transformer-based models / NER examples.
- `doc classification.py`: simple document classification demo (Keras / LSTM).
- `ocr.py`: OCR demo using `pytesseract` and `Pillow`.
- `Predictive.py`, `predictive_userinteraction.py`: scikit-learn predictive demos and user interaction examples.
- `simple_llm.py`: example LLM integration (LangChain / Ollama).
- `simple_vertex.py`: starter for Google Vertex AI client usage.
- `ragumodelimport.py`, `ragumodelmerge.py`: utilities for importing/merging RAG model artifacts.
- `modelnlp.py`, `Structured.py`, `Unstructured.py`: other NLP examples and structured/unstructured data handling.

## Models

The `models/` directory contains two subfolders (`ragustructured` and `raguunstructured`) with tokenizer configs and trained model files. Those are used by the RAG-related scripts.

## Running a script

Example: run the OCR demo

```powershell
.\.venv\Scripts\Activate.ps1
python ocr.py
```

## Additional setup

- If you use spaCy scripts, you may need to download a model, e.g.:

```powershell
python -m spacy download en_core_web_sm
```

If you want, I can add example commands or small README sections for any specific script—tell me which one to document next.

