"""Download the showcase source clip (CC BY 3.0 Big Buck Bunny excerpt) and verify its SHA-256.

The recorded traces in demo/showcase/ were produced from this exact file; the server refuses any
other bytes. Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RIGHTS = HERE / "showcase/rights.json"
TARGET = HERE / "showcase/media/bbb-excerpt-0033-0459.webm"
URL = "https://github.com/tolyaho/StreamingVideoMemoryBuffer/releases/download/showcase-v1/bbb-excerpt-0033-0459.webm"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=URL)
    parser.add_argument("--output", type=Path, default=TARGET)
    args = parser.parse_args()
    expected = json.loads(RIGHTS.read_text())["source_sha256"]
    if args.output.is_file() and sha256(args.output) == expected:
        print(f"Already present and verified: {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".part")
    print(f"Downloading {args.url}")
    with urllib.request.urlopen(args.url) as response, partial.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    actual = sha256(partial)
    if actual != expected:
        partial.unlink()
        sys.exit(f"SHA-256 mismatch: got {actual}, expected {expected}. Nothing was kept.")
    partial.replace(args.output)
    print(f"Verified {args.output} ({args.output.stat().st_size} bytes, SHA-256 {actual})")


if __name__ == "__main__":
    main()
