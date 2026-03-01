from transformers import pipeline

unstructured_nlp = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

text = "RBI is expected to maintain the 5.25% interest rate."

# Extract entities based on context
entities = unstructured_nlp(text)

print("--- Unstructured Path Results ---")
for ent in entities:
    print(f"Entity: {ent['word']} | Label: {ent['entity_group']} | Score: {ent['score']:.2f}")