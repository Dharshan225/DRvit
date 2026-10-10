"""
DR ViT - Hugging Face Model Downloader

Downloads the ONNX model and its external weights
into the local models directory.
"""

from pathlib import Path
from huggingface_hub import hf_hub_download
import shutil
import sys


# =========================================================
# CONFIGURATION
# =========================================================

REPO_ID = "dharshan25/drvit_model"

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_FILES = [
    "DR_ViT_model.onnx",
    "DR_ViT_model.onnx.data",
]


# =========================================================
# DOWNLOAD MODEL FILES
# =========================================================

def download_model():
    """Download model files from Hugging Face when necessary."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DR ViT - HUGGING FACE MODEL DOWNLOADER")
    print("=" * 60)

    for filename in MODEL_FILES:
        destination = MODEL_DIR / filename

        # Reuse a local file if it already exists and is non-empty.
        if destination.is_file() and destination.stat().st_size > 0:
            print(f"\nAlready available: {filename}")
            print(f"Location: {destination}")
            continue

        print(f"\nDownloading: {filename}")

        try:
            cached_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="model",
            )

            # Copy the verified cached file into the project.
            shutil.copy2(cached_path, destination)

            if not destination.is_file() or destination.stat().st_size == 0:
                raise RuntimeError(
                    f"Downloaded file is missing or empty: {filename}"
                )

            print(f"Downloaded successfully: {destination}")

        except Exception as error:
            # Remove an incomplete destination file, if present.
            if destination.exists():
                destination.unlink()

            raise RuntimeError(
                f"Could not download {filename} from "
                f"Hugging Face repository {REPO_ID}.\n"
                f"Check your internet connection and repository files.\n"
                f"Original error: {error}"
            ) from error

    print("\n" + "=" * 60)
    print("MODEL FILE DOWNLOAD CHECK COMPLETED")
    print(f"Model directory: {MODEL_DIR}")
    print("=" * 60)


# =========================================================
# RUN DIRECTLY
# =========================================================

if __name__ == "__main__":
    try:
        download_model()
    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)