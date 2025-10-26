"""
Download pretrained nanochat model from Hugging Face.
This downloads the community-trained model from sdobson/nanochat.

Usage:
python -m scripts.download_pretrained
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download

def main():
    print("\n" + "="*80)
    print("Download Pretrained nanochat Model")
    print("="*80)

    # Determine cache directory
    cache_dir = Path(os.path.expanduser("~/.cache/nanochat"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCache directory: {cache_dir}")
    print("\nDownloading model from Hugging Face: sdobson/nanochat")
    print("This may take several minutes depending on your connection...\n")

    try:
        # Download the model
        model_path = snapshot_download(
            repo_id="sdobson/nanochat",
            cache_dir=cache_dir,
            local_dir=cache_dir / "community_model",
            local_dir_use_symlinks=False
        )

        print(f"\n✓ Model downloaded successfully!")
        print(f"Location: {model_path}")

        print("\n" + "="*80)
        print("Next Steps")
        print("="*80)
        print("\n1. Inspect the downloaded files:")
        print(f"   ls {cache_dir / 'community_model'}")

        print("\n2. You may need to adapt the checkpoint format to work with nanochat")
        print("   The community model might have a different structure")

        print("\n3. Alternatively, use the Hugging Face Space for testing:")
        print("   https://huggingface.co/spaces/sdobson/nanochat")

        print("\n" + "="*80)

    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print("\nAlternative: Use the online demo at:")
        print("https://huggingface.co/spaces/sdobson/nanochat")

if __name__ == "__main__":
    main()
