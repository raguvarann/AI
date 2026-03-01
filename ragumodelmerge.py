import spacy
from transformers import pipeline
import os

#ragu models
STRUCTURED_MODEL_PATH = "D:/AI/models/ragustructured"
UNSTRUCTURED_MODEL_PATH = "D:/AI/models/raguunstructured"

print("Loading Ragu Models... please wait.")

# Load the spaCy "Structured" model
nlp_ragu_struct = spacy.load(STRUCTURED_MODEL_PATH)

# Load the BERT "Unstructured" model
nlp_ragu_unstruct = pipeline(
    "ner", 
    model=UNSTRUCTURED_MODEL_PATH, 
    tokenizer=UNSTRUCTURED_MODEL_PATH,
    aggregation_strategy="simple"
)

# MERGE FUNCTION
def get_ragu_entities(text):

    # getting raw hits from both "Ragu" models
    doc_struct = nlp_ragu_struct(text)
    res_unstruct = nlp_ragu_unstruct(text)
    
    # dictionary to store entities by their character position (start, end)
    final_entities = {}

    # Process Structured first (THE RULES)
    for ent in doc_struct.ents:
        # We only prioritize our high-confidence custom labels
        if ent.label_ in ["TICKER", "PERCENTAGE", "FIN_INST"]:
            span = (ent.start_char, ent.end_char)
            final_entities[span] = {
                "text": ent.text,
                "label": ent.label_,
                "method": "Structured (Verified)"
            }

    # Process Unstructured (THE AI GUESSTIMATES)
    for ent in res_unstruct:
        span = (ent['start'], ent['end'])
        # IF THIS POSITION IS NOT ALREADY TAKEN BY A RULE, ADD IT
        if span not in final_entities:
            final_entities[span] = {
                "text": ent['word'],
                "label": ent['entity_group'],
                "method": "Unstructured (Detected)"
            }
            
    return list(final_entities.values())

# TEST THE PIPELINE
if __name__ == "__main__":
    print("\n--- RAGU MASTER PIPELINE ACTIVE ---\n")
    
    test_text = "The RBI reported that Banks are seeing a 2.75% growth."
    
    # Running the merge function
    clean_results = get_ragu_entities(test_text)

    # Print a clean, formatted table
    print(f"{'Text':<15} | {'Label':<12} | {'Method'}")
    print("-" * 50)
    for e in clean_results:
        print(f"{e['text']:<15} | {e['label']:<12} | {e['method']}")