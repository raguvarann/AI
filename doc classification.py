import os
import pdfplumber
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Example dataset
data = {
    "text": [
        "Employee salary for February 2026: $1000",
        "Rental payment for apartment 12B, February 2026",
        "Invoice #1234: Purchase of office supplies",
        "Payslip for March 2026",
        "Landlord receipt for February rent"
    ],
    "label": [
        "payslip",
        "rental",
        "invoice",
        "payslip",
        "rental"
    ]
}

df = pd.DataFrame(data)

# Encode labels
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label_encoded'], test_size=0.2, random_state=42
)

# Tokenize text
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences
max_len = 20
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

# LSTM model
model = Sequential([
    Embedding(input_dim=1000, output_dim=64, input_length=max_len),
    LSTM(64),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train
model.fit(X_train_pad, y_train, epochs=20, validation_split=0.1)

# Evaluate
loss, acc = model.evaluate(X_test_pad, y_test)
print("Test Accuracy:", acc)

# Predict function
def classify_document(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(pad)
    label = le.inverse_transform([pred.argmax()])[0]
    return label

# Test prediction
doc_text = "This month’s salary statement for John Doe"
print("Document Type:", classify_document(doc_text))
