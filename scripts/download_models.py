import os
import requests
from pathlib import Path

def download_model(url: str, output_path: str):
    """Download a model file if it doesn't exist."""
    if not os.path.exists(output_path):
        print(f"Downloading model to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete!")
    else:
        print(f"Model already exists at {output_path}")

def main():
    """Download required local models."""
    models = {
        "llama-2-7b.Q4_K_M.gguf": "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf",
    }

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    for model_name, url in models.items():
        output_path = model_dir / model_name
        download_model(url, str(output_path))

if __name__ == "__main__":
    main()