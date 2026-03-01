import spacy
from transformers import pipeline

# Structured pipeline using spaCy, from ragustructured
nlp_ragu_struct = spacy.load("D:/AI/models/ragustructured")

# Unstructured pipeline using Hugging Face Transformers, from ragumodels
nlp_ragu_unstruct = pipeline(
    "ner", 
    model="D:/AI/models/raguunstructured", 
    tokenizer="D:/AI/models/raguunstructured",
    aggregation_strategy="simple"
)

# Sample text
sample = "The RBI intervention caused a 2.75% shift in the repo rate, affecting Banks."

# Processing through both pipelines
doc_struct = nlp_ragu_struct(sample)
res_unstruct = nlp_ragu_unstruct(sample)

print("--- RAGU PIPELINE EXPLORATION ---")
# Show Structured Hits
for ent in doc_struct.ents:
    if ent.label_ in ["TICKER", "PERCENTAGE", "FIN_INST"]:
        print(f"[Structured Match]   {ent.text:<10} | Label: {ent.label_}")

# Show Unstructured Hits
for ent in res_unstruct:
    print(f"[Unstructured Match] {ent['word']:<10} | Label: {ent['entity_group']}")