import streamlit as st
import fitz
import docx
import pandas as pd
import io
from raguwrapper import run_ragu_pipeline

def extract_text(uploaded_file):
    filename = uploaded_file.name.lower()
    file_content = io.BytesIO(uploaded_file.getvalue())
    
    if filename.endswith(".pdf"):
        doc = fitz.open(stream=file_content, filetype="pdf")
        return "\n".join([page.get_text() for page in doc]).encode("utf-8")
    elif filename.endswith(".docx"):
        doc = docx.Document(file_content)
        return "\n".join([p.text for p in doc.paragraphs]).encode("utf-8")
    return uploaded_file.getvalue()

st.set_page_config(page_title="RAGU Analysis", layout="wide")
st.title("🔍 RAGU Common Analysis")

uploaded_file = st.file_uploader("Upload any common", type=None)

if uploaded_file and st.button("Analyze Common"):
    with st.spinner("Processing..."):
        try:
            results = run_ragu_pipeline(extract_text(uploaded_file))
            if results:
                df = pd.DataFrame(results)
                st.dataframe(df)
                c1, c2 = st.columns(2)
                c1.download_button("📥 CSV", df.to_csv(index=False), "results.csv", "text/csv")
                c2.download_button("📥 TXT", df.to_string(index=False), "results.txt", "text/plain")
            else:
                st.info("No entities detected.")
        except Exception as e:
            st.error(f"Error: {e}")