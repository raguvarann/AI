import streamlit as st
import spacy
from transformers import pipeline
import logging
import torch
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STRUCTURED_MODEL_PATH = "models/ragustructured"
UNSTRUCTURED_MODEL_PATH = "models/raguunstructured"

@st.cache_resource
def load_nlp_resources():
    logger.info("Loading models...")
    nlp_struct = spacy.load(STRUCTURED_MODEL_PATH)
    
    device = 0 if torch.cuda.is_available() else -1
    nlp_unstruct = pipeline(
        "ner", 
        model=UNSTRUCTURED_MODEL_PATH, 
        tokenizer=UNSTRUCTURED_MODEL_PATH,
        aggregation_strategy="simple",
        device=device
    )
    return nlp_struct, nlp_unstruct

def chunk_text(text, chunk_size=1500):
    """Split text into chunks."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def run_ragu_pipeline(file_bytes):
    nlp_struct, nlp_unstruct = load_nlp_resources()
    text = file_bytes.decode("utf-8", errors="ignore")
    
    final_entities = {}
    
    # 1. Structured Inference (SpaCy)
    try:
        doc_struct = nlp_struct(text)
        for ent in doc_struct.ents:
            if ent.label_ in ["SYMBOL", "RATIO", "NETWORK"]:
                final_entities[(ent.start_char, ent.end_char)] = {
                    "Text": ent.text, "Label": ent.label_, "Method": "Structured"
                }
    except Exception as e:
        logger.error(f"SpaCy error: {e}")

    # 2. Unstructured Inference (Transformer + Offset Adjustment)
    chunks = chunk_text(text)
    current_offset = 0
    for chunk in chunks:
        try:
            results = nlp_unstruct(chunk)
            for ent in results:
                global_span = (ent['start'] + current_offset, ent['end'] + current_offset)
                if global_span not in final_entities:
                    final_entities[global_span] = {
                        "Text": ent['word'], "Label": ent['entity_group'], "Method": "Unstructured"
                    }
        except Exception as e:
            logger.warning(f"Chunk error: {e}")
        current_offset += len(chunk)
                
    return list(final_entities.values())