import os
import spacy
from transformers import pipeline


STRUCTURED_PATH = "D:/AI/models/ragustructured"
UNSTRUCTURED_PATH = "D:/AI/models/raguunstructured"


os.makedirs("D:/AI/models", exist_ok=True)

nlp = spacy.load("en_core_web_sm")
ruler = nlp.add_pipe("entity_ruler", before="ner")
patterns = [
    {"label": "TICKER", "pattern": [{"TEXT": {"REGEX": r"^\$[A-Z]{1,5}$"}}]},
    {"label": "PERCENTAGE", "pattern": [{"LIKE_NUM": True}, {"ORTH": "%"}]}
]
ruler.add_patterns(patterns)
nlp.to_disk(STRUCTURED_PATH)
print(f"Successfully saved Structured Model to: {STRUCTURED_PATH}")


pipe = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
pipe.model.save_pretrained(UNSTRUCTURED_PATH)
pipe.tokenizer.save_pretrained(UNSTRUCTURED_PATH)
print(f"Successfully saved Unstructured Model to: {UNSTRUCTURED_PATH}")