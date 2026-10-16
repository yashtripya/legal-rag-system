"""
Utility script to package ChromaDB into split-archive binaries (< 50MB each)
suitable for Git tracking and Streamlit Cloud deployment.
"""

import os
import sys
import io
import shutil
from pathlib import Path

# Fix Windows console UTF-8 output
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma_db"
PREBUILT_DIR = STORAGE_DIR / "prebuilt"
TEMP_ZIP = STORAGE_DIR / "chroma_temp.zip"

CHUNK_SIZE = 45 * 1024 * 1024  # 45 MB per split part (well below GitHub's 100 MB limit)


def package_chroma_db():
    print(f"📦 Packaging ChromaDB from: {CHROMA_DIR}")
    if not CHROMA_DIR.exists() or not list(CHROMA_DIR.iterdir()):
        raise RuntimeError(f"ChromaDB directory '{CHROMA_DIR}' is empty or does not exist!")

    PREBUILT_DIR.mkdir(parents=True, exist_ok=True)

    # Clean existing prebuilt parts
    for existing in PREBUILT_DIR.glob("part_*.bin"):
        existing.unlink()

    # Step 1: Compress to zip
    print("Compressing vector database directory to zip...")
    shutil.make_archive(str(TEMP_ZIP).replace(".zip", ""), "zip", str(CHROMA_DIR))

    total_zip_size = os.path.getsize(TEMP_ZIP)
    print(f"Compressed archive size: {total_zip_size / (1024 * 1024):.2f} MB")

    # Step 2: Split into parts
    print("Splitting into parts under 50 MB for GitHub tracking...")
    part_idx = 0
    with open(TEMP_ZIP, "rb") as f_in:
        while True:
            chunk = f_in.read(CHUNK_SIZE)
            if not chunk:
                break
            part_path = PREBUILT_DIR / f"part_{part_idx:03d}.bin"
            with open(part_path, "wb") as f_out:
                f_out.write(chunk)
            print(f"  Created: {part_path.name} ({len(chunk) / (1024 * 1024):.2f} MB)")
            part_idx += 1

    # Cleanup temporary zip
    if TEMP_ZIP.exists():
        TEMP_ZIP.unlink()

    print(f"✅ Successfully packaged {part_idx} parts into: {PREBUILT_DIR}")


if __name__ == "__main__":
    package_chroma_db()
