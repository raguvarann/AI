import spacy

# Load English model
nlp = spacy.load("en_core_web_sm")

text = "Ragu started loving NLP in 2025."

doc = nlp(text)

for ent in doc.ents:
    print(ent.text, "-", ent.label_)