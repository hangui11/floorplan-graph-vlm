"""
Download Qwen3-VL-8B-Instruct model from Hugging Face and save locally.

Usage:
    python download_model.py
"""
from pathlib import Path
from huggingface_hub import snapshot_download

# MODEL_REPO = "Qwen/Qwen3-VL-4B-Instruct"
# LOCAL_DIR = Path(__file__).resolve().parent / "models" / "Qwen3-VL-4B-Instruct"

MODEL_REPO = "Qwen/Qwen3-VL-8B-Instruct"
LOCAL_DIR = Path(__file__).resolve().parent / "models" / "Qwen3-VL-8B-Instruct"

def main():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {MODEL_REPO} to {LOCAL_DIR}...")
    print("This may take a while (~16 GB).\n")

    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=str(LOCAL_DIR),
        local_dir_use_symlinks=False,
    )

    print(f"\nDone. Model saved to: {LOCAL_DIR}")


if __name__ == "__main__":
    main()
