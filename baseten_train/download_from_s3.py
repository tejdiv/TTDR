"""Download checkpoint artifacts from presigned S3 URLs.

Reads the JSON from `truss train get_checkpoint_urls` and downloads
all files into a target directory, preserving the relative path structure.
Uses parallel downloads for speed.

Usage:
    python download_from_s3.py --json checkpoints.json --output_dir /path/to/output
"""

import argparse
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_file(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        return dest_path, "skipped"
    urllib.request.urlretrieve(url, dest_path)
    return dest_path, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to checkpoint URLs JSON")
    parser.add_argument("--output_dir", required=True, help="Where to download files")
    parser.add_argument("--workers", type=int, default=32, help="Parallel downloads")
    args = parser.parse_args()

    with open(args.json) as f:
        data = json.load(f)

    artifacts = data["checkpoint_artifacts"]
    print(f"Downloading {len(artifacts)} files to {args.output_dir}...")

    total_bytes = sum(a["size_bytes"] for a in artifacts)
    print(f"Total size: {total_bytes / 1e9:.1f} GB")

    futures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for art in artifacts:
            # relative_file_name is like "rlds/bridge_dataset/1.0.0/bridge_dataset/1.0.0/file"
            dest = os.path.join(args.output_dir, art["relative_file_name"])
            fut = pool.submit(download_file, art["url"], dest)
            futures[fut] = art["relative_file_name"]

        done = 0
        for fut in as_completed(futures):
            done += 1
            path, status = fut.result()
            if done % 100 == 0 or done == len(artifacts):
                print(f"  {done}/{len(artifacts)} ({status})")

    print("Download complete.")


if __name__ == "__main__":
    main()
