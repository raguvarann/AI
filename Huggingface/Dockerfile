FROM python:3.13.5-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY src/ ./src/
COPY models/ ./models/

RUN pip3 install -r requirements.txt

EXPOSE 7860

HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health

# Use 7860 as the port
ENTRYPOINT ["streamlit", "run", "src/raguapp.py", \
            "--server.port=7860", \
            "--server.address=0.0.0.0", \
            "--server.enableCORS=false", \
            "--server.enableXsrfProtection=false"]