"""
Download and setup PPE detection model for backend
"""
from huggingface_hub import hf_hub_download
import os

def download_ppe_model(cache_dir="./models"):
    """Download PPE detection model from HuggingFace"""
    repo_id = "Hansung-Cho/yolov8-ppe-detection"
    filename = "best.pt"
    
    print(f"📥 Downloading PPE detection model...")
    print(f"   Repository: {repo_id}")
    print(f"   Cache directory: {cache_dir}")
    
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir
    )
    
    print(f"✅ Model downloaded successfully!")
    print(f"   Path: {model_path}")
    
    return model_path

if __name__ == "__main__":
    model_path = download_ppe_model()
    print(f"\n✅ Ready to use!")
    print(f"   Model path: {model_path}")
