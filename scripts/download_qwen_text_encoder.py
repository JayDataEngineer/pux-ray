#!/usr/bin/env python3
"""Download qwen_3_06b_base.safetensors text encoder for Anima model."""
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

def main():
    ckpts_dir = Path("/mnt/data/models")
    qwen_dir = ckpts_dir / "Qwen3-0.6B"
    qwen_dir.mkdir(parents=True, exist_ok=True)
    
    target_file = qwen_dir / "qwen_3_06b_base.safetensors"
    
    if target_file.exists():
        print(f"✓ Text encoder already exists: {target_file}")
        return
    
    print("Downloading qwen_3_06b_base.safetensors text encoder for Anima...")
    try:
        file_path = hf_hub_download(
            repo_id="circlestone-labs/Anima",
            filename="split_files/text_encoders/qwen_3_06b_base.safetensors",
            local_dir=ckpts_dir,
            local_dir_use_symlinks=False
        )
        print(f"✓ Downloaded to: {file_path}")
        print(f"✓ Target location: {target_file}")
        
        # Verify the file exists
        if target_file.exists():
            print("✓ Text encoder successfully downloaded and verified!")
        else:
            # The file might be downloaded to split_files/text_encoders/
            # Create a symlink to the expected location
            downloaded_file = ckpts_dir / "split_files" / "text_encoders" / "qwen_3_06b_base.safetensors"
            if downloaded_file.exists():
                os.symlink(downloaded_file, target_file)
                print(f"✓ Created symlink: {target_file} -> {downloaded_file}")
            else:
                print("✗ File downloaded but not found at expected location")
                sys.exit(1)
            
    except Exception as e:
        print(f"✗ Download failed: {e}")
        print("\nManual download required:")
        print("1. Visit: https://huggingface.co/circlestone-labs/Anima")
        print("2. Download: split_files/text_encoders/qwen_3_06b_base.safetensors")
        print(f"3. Place in: {qwen_dir}/")
        sys.exit(1)

if __name__ == "__main__":
    main()
