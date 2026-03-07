import sys
sys.path.append(r'C:\Users\raguv\AppData\Roaming\Python\Python311\site-packages')

from huggingface_hub import HfApi

token = "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" 

api = HfApi(token=token)

api.upload_folder(
    folder_path = "D:/AI/Huggingface",
    repo_id="raguaillm/DocumentAnalysis",
    repo_type="space",
    # path_in_repo="src"
)
print("Upload complete!")