#!/usr/bin/env python3
"""
Download ISIMIP3b daily rsds NetCDF4 files and generate ssrd TIFs.

Usage (inside the container):
    python3 /tmp/download_and_process_ssrd.py \
        --static-dir /app/src/waterpath_data_service/static \
        --sources /app/src/waterpath_data_service/static/data/hydrology/hydrology_original/rsds_sources.txt

Downloads each URL from sources.txt into rsds_daily/ (skipping files that
already exist with a non-zero size), then runs generate_hydrology_models.py
which processes the downloaded NCs into 12-month ssrd_m{NN}.tif files.
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path


def download_file(url: str, dest: Path) -> None:
    """Download *url* to *dest*, printing progress every 10%."""
    tmp = dest.with_suffix(".part")
    try:
        def _progress(block_count, block_size, total_size):
            if total_size <= 0:
                return
            done = block_count * block_size
            pct = min(100, int(done * 100 / total_size))
            mb_done = done / 1024 / 1024
            mb_total = total_size / 1024 / 1024
            if pct % 10 == 0:
                print(f"    {pct}%  ({mb_done:.0f} / {mb_total:.0f} MB)", flush=True)

        urllib.request.urlretrieve(url, tmp, reporthook=_progress)
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Download rsds NCs and generate ssrd TIFs.")
    parser.add_argument("--static-dir", required=True, help="Path to static/ directory.")
    parser.add_argument(
        "--sources",
        default=None,
        help="Path to rsds_sources.txt. Defaults to "
             "<static-dir>/data/hydrology/hydrology_original/rsds_sources.txt",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download and re-process files that already exist.",
    )
    args = parser.parse_args()

    static_dir = Path(args.static_dir).resolve()
    sources_path = Path(args.sources) if args.sources else (
        static_dir / "data" / "hydrology" / "hydrology_original" / "rsds_sources.txt"
    )
    rsds_dir = static_dir / "data" / "hydrology" / "hydrology_original" / "rsds_daily"
    rsds_dir.mkdir(parents=True, exist_ok=True)

    if not sources_path.is_file():
        print(f"ERROR: sources file not found: {sources_path}")
        sys.exit(1)

    urls = [line.strip() for line in sources_path.read_text().splitlines() if line.strip()]
    print(f"Sources  : {sources_path}")
    print(f"rsds_dir : {rsds_dir}")
    print(f"Files    : {len(urls)}\n")

    # ------------------------------------------------------------------ #
    # Download                                                             #
    # ------------------------------------------------------------------ #
    for i, url in enumerate(urls, 1):
        filename = url.split("/")[-1]
        dest = rsds_dir / filename
        if not args.overwrite and dest.is_file() and dest.stat().st_size > 0:
            print(f"[{i}/{len(urls)}] SKIP (exists): {filename}")
            continue
        print(f"[{i}/{len(urls)}] Downloading: {filename}", flush=True)
        try:
            download_file(url, dest)
            size_gb = dest.stat().st_size / 1024 ** 3
            print(f"  ✅ {filename}  ({size_gb:.2f} GB)", flush=True)
        except Exception as exc:
            print(f"  ❌ FAILED: {exc}", flush=True)
            # Continue with remaining files rather than aborting.

    print("\nAll downloads complete. Starting ssrd TIF generation...\n")

    # ------------------------------------------------------------------ #
    # Generate ssrd TIFs                                                   #
    # ------------------------------------------------------------------ #
    # Re-use the generate_hydrology_models.py script (expected at /tmp/).
    generate_script = Path("/tmp/generate_hydrology_models.py")
    if not generate_script.is_file():
        # Fallback: look alongside this script.
        generate_script = Path(__file__).parent / "generate_hydrology_models.py"
    if not generate_script.is_file():
        print("ERROR: generate_hydrology_models.py not found.")
        sys.exit(1)

    # Import and call main() directly so we stay in the same process.
    import importlib.util
    spec = importlib.util.spec_from_file_location("gen_hydro", generate_script)
    gen_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen_module)

    # Patch sys.argv so argparse inside the module sees the right args.
    overwrite_flag = ["--overwrite"] if args.overwrite else []
    sys.argv = ["generate_hydrology_models.py", "--static-dir", str(static_dir)] + overwrite_flag
    gen_module.main()


if __name__ == "__main__":
    main()
