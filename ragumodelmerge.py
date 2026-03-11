import os
import spacy
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Define your paths
STRUCTURED_PATH = "D:/AI/models/ragustructured"
UNSTRUCTURED_PATH = "D:/AI/models/raguunstructured"

# Create directories if they don't exist
os.makedirs(STRUCTURED_PATH, exist_ok=True)
os.makedirs(UNSTRUCTURED_PATH, exist_ok=True)

print("Downloading and saving models to D: drive... this may take a minute.")

# 1. Save the spaCy "Structured" model
nlp = spacy.load("en_core_web_sm")
nlp.to_disk(STRUCTURED_PATH)
print(f"✓ Saved Structured model to {STRUCTURED_PATH}")

# 2. Save the BERT "Unstructured" model
model_name = "dslim/bert-base-NER"
model = AutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(UNSTRUCTURED_PATH)
tokenizer.save_pretrained(UNSTRUCTURED_PATH)
print(f"✓ Saved Unstructured model to {UNSTRUCTURED_PATH}")

print("\nSetup complete! You can now run ragumodelmerge.py")