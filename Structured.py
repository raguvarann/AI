import spacy
from spacy.pipeline import EntityRuler

nlp = spacy.load("en_core_web_sm")


patterns = [
    {"label": "FIN_INST", "pattern": "RBI"},
    {"label": "FIN_INST", "pattern": "HDFC"},
    {"label": "FIN_INST", "pattern": "AXIS"},
    {"label": "TICKER", "pattern": [{"TEXT": {"REGEX": r"^\$[A-Z]{1,5}$"}}]},
    {"label": "PERCENTAGE", "pattern": [{"LIKE_NUM": True}, {"ORTH": "%"}]}
]


ruler = nlp.add_pipe("entity_ruler", before="ner")
ruler.add_patterns(patterns)


nlp.to_disk("D:/AI/models/ragustructured") 

print("Successfully saved new rules to D:/AI/models/ragustructured")

# doc = nlp("Check the quarterly performance for RBI and the 4.5% yield.")

# print("\n--- Structured Path Results ---")
# for ent in doc.ents:
#     if ent.label_ in ["TICKER", "PERCENTAGE"]:
#         print(f"Structured Match: {ent.text} | Label: {ent.label_}")