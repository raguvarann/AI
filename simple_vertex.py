from google.cloud import aiplatform

aiplatform.init(
    project="vertex-ai-practice-487809",
    location="us-central1"
)

print("Ragu loves Vertex AI!")
