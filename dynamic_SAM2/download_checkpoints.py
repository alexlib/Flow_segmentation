"""
download_checkpoints.py
-----------------------
Pulls the SAM 2.1 checkpoint weights from HuggingFace Hub into ./checkpoints/
on first run, so the repo itself can stay small enough for GitHub.

Usage:

    python download_checkpoints.py              # downloads "large" only (~900 MB)
    python download_checkpoints.py --size tiny  # one of: tiny, small, base_plus, large
    python download_checkpoints.py --all        # pull every size (~2.3 GB total)
    python download_checkpoints.py --check      # just list what is already there

Weights come from the official Meta / FAIR repo on HuggingFace, so no separate
account or auth is needed. The files are byte-identical to the ones at
https://dl.fbaipublicfiles.com/segment_anything_2/ .
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"

# (huggingface repo id, filename on disk, expected byte size)
WEIGHTS = {
    "tiny":      ("facebook/sam2.1-hiera-tiny",       "sam2.1_hiera_tiny.pt",       156008466),
    "small":     ("facebook/sam2.1-hiera-small",      "sam2.1_hiera_small.pt",      184416285),
    "base_plus": ("facebook/sam2.1-hiera-base-plus",  "sam2.1_hiera_base_plus.pt",  323606802),
    "large":     ("facebook/sam2.1-hiera-large",      "sam2.1_hiera_large.pt",      898083611),
}


def _ensure_hf_hub():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("Installing huggingface_hub ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])


def _status(name: str):
    spec = WEIGHTS[name]
    _, fname, size = spec
    p = CKPT_DIR / fname
    if p.exists() and p.stat().st_size == size:
        return p, True
    return p, False


def download(size: str):
    _ensure_hf_hub()
    from huggingface_hub import hf_hub_download

    repo_id, fname, expected = WEIGHTS[size]
    dst = CKPT_DIR / fname
    if dst.exists() and dst.stat().st_size == expected:
        print(f"[skip] {fname} already present ({expected/1e6:.1f} MB)")
        return dst

    CKPT_DIR.mkdir(exist_ok=True)
    print(f"[get ] {repo_id}/{fname}  ({expected/1e6:.1f} MB)")
    cached = hf_hub_download(repo_id=repo_id, filename=fname)
    # Hugging Face caches elsewhere; copy into ./checkpoints/ so the rest of
    # the code finds it at a stable local path.
    import shutil
    shutil.copyfile(cached, dst)
    got = dst.stat().st_size
    if got != expected:
        raise RuntimeError(f"size mismatch for {fname}: got {got}, expected {expected}")
    print(f"[ok  ] {dst} ({got/1e6:.1f} MB)")
    return dst


def main():
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--size", choices=list(WEIGHTS), default="large",
                     help="Which checkpoint to download (default: large)")
    grp.add_argument("--all", action="store_true", help="Download every size")
    ap.add_argument("--check", action="store_true",
                    help="List what is already in checkpoints/ and exit")
    args = ap.parse_args()

    if args.check:
        for name in WEIGHTS:
            p, ok = _status(name)
            tag = "present" if ok else "missing"
            print(f"  {name:10s} {tag:8s} {p.name}")
        return

    sizes = list(WEIGHTS) if args.all else [args.size]
    for s in sizes:
        download(s)
    print("done.")


if __name__ == "__main__":
    main()
