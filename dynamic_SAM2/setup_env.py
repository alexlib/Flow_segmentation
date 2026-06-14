"""
setup_env.py  --  one-shot, fully-automated bootstrap for this bundle
---------------------------------------------------------------------

Works unchanged on:
    Windows  10 / 11   with NVIDIA CUDA  or  CPU only
    Linux              with NVIDIA CUDA  or  CPU only
    macOS              Apple Silicon (MPS)  or  Intel (CPU)
    Google Colab       (CUDA already installed by Colab)

Run it once from the bundle folder:

    python setup_env.py

or, if your Python is exposed as `python3`:

    python3 setup_env.py

What it does, in order:
    1. Detects OS, CPU arch, Python version, and Google Colab
    2. Detects NVIDIA GPU via `nvidia-smi` (reports the CUDA version)
    3. Installs torch + torchvision with the matching wheel:
            NVIDIA (Windows/Linux)  -> https://download.pytorch.org/whl/cu121
            macOS  -> default PyPI (ships with MPS support)
            Colab  -> leaves the pre-installed torch alone
       Falls back to `pip install --user` when the system site-packages is
       not writable, so no admin / sudo is ever needed.
    4. Installs every other dependency from requirements.txt
    5. Verifies the presence and exact byte size of every checkpoint file
    6. Wires the bundle folder into Python's search path via a .pth file,
       so `import sam2` and `import build` work from any notebook without
       the user touching sys.path
    7. Prints a clear READY / NOT READY summary

Flags:
    python setup_env.py --check        # verify only, skip installs
    python setup_env.py --no-torch     # skip torch (keep whatever is there)
    python setup_env.py --cpu          # force CPU torch even if NVIDIA is present
    python setup_env.py --cuda cu118   # force a specific CUDA wheel index
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Exact byte sizes for integrity check
EXPECTED_CHECKPOINTS = {
    "sam2.1_hiera_large.pt":        898083611,
    "sam2.1_hiera_base_plus.pt":    323606802,
    "sam2.1_hiera_small.pt":        184416285,
    "sam2.1_hiera_tiny.pt":         156008466,
}

EXPECTED_CONFIGS = [
    "sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_l.yaml",
    "sam2.1_hiera_s.yaml",
    "sam2.1_hiera_t.yaml",
]

# In the public repo only the "large" model is expected by default; the rest
# can be pulled on demand via download_checkpoints.py --all.
PUBLIC_CHECKPOINTS_DEFAULT = ["sam2.1_hiera_large.pt"]

DEFAULT_CUDA_TAG = "cu121"   # works for CUDA 12.1 through 12.6 drivers
# ----------------------------------------------------------------------


def info(msg):  print(f"  {msg}")
def ok(msg):    print(f"  [ OK ] {msg}")
def warn(msg):  print(f"  [WARN] {msg}")
def fail(msg):  print(f"  [FAIL] {msg}")


def section(title):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def detect_env():
    in_colab = "google.colab" in sys.modules or "COLAB_GPU" in os.environ
    system = platform.system()      # Windows / Linux / Darwin
    machine = platform.machine()
    py = sys.version.split()[0]
    info(f"OS            : {system} ({machine})")
    info(f"Python        : {py}")
    info(f"Google Colab  : {in_colab}")
    return {"colab": in_colab, "system": system, "machine": machine, "python": py}


def detect_nvidia():
    """Return CUDA major.minor string like '12.4' if nvidia-smi works, else None."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.check_output([exe], stderr=subprocess.STDOUT, timeout=8).decode(errors="ignore")
    except Exception:
        return None
    for line in out.splitlines():
        if "CUDA Version" in line:
            try:
                return line.split("CUDA Version:")[1].strip().split()[0]
            except Exception:
                return "unknown"
    return "unknown"


def pip_install(args):
    """pip install <args> -- tries system first, falls back to --user on failure."""
    base = [sys.executable, "-m", "pip", "install", *args]
    info(" ".join(base))
    try:
        subprocess.check_call(base)
        return
    except subprocess.CalledProcessError:
        warn("system install failed, retrying with --user ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", *args])


def ensure_torch(env, cuda_tag=DEFAULT_CUDA_TAG, force_cpu=False):
    """Install the right torch/torchvision pair if they are not already there."""
    if env["colab"]:
        info("Colab detected -> keeping Colab's bundled torch (CUDA already there).")
        return

    try:
        import torch  # noqa: F401
        ok(f"torch {torch.__version__} already installed -> leaving alone")
        return
    except ImportError:
        pass

    cuda_version = None if force_cpu else detect_nvidia()
    system = env["system"]

    if cuda_version and system in ("Windows", "Linux"):
        info(f"NVIDIA driver reports CUDA {cuda_version} -> installing torch wheel {cuda_tag}")
        idx = f"https://download.pytorch.org/whl/{cuda_tag}"
        pip_install(["torch", "torchvision", "--index-url", idx])
    elif system == "Darwin":
        info("macOS detected -> installing default torch (MPS / CPU wheel)")
        pip_install(["torch", "torchvision"])
    else:
        info("No NVIDIA GPU detected -> installing CPU-only torch wheel")
        idx = "https://download.pytorch.org/whl/cpu"
        pip_install(["torch", "torchvision", "--index-url", idx])


def install_other_deps():
    req = PROJECT_ROOT / "requirements.txt"
    if not req.exists():
        fail(f"requirements.txt not found at {req}")
        return
    info(f"Installing the rest from {req.name}")
    pip_install(["-r", str(req)])


def ensure_checkpoint(size="large"):
    """Pull the requested SAM 2.1 checkpoint from HuggingFace if it is not there."""
    dl = PROJECT_ROOT / "download_checkpoints.py"
    if not dl.exists():
        warn("download_checkpoints.py not found -- skipping auto-download")
        return
    info(f"Making sure checkpoint '{size}' is on disk ...")
    try:
        subprocess.check_call([sys.executable, str(dl), "--size", size])
    except subprocess.CalledProcessError:
        warn(f"checkpoint download failed. Run manually: python download_checkpoints.py --size {size}")


def detect_backend():
    try:
        import torch
    except ImportError:
        warn("torch still not importable after install -> something went wrong")
        return None
    if torch.cuda.is_available():
        dev = f"cuda (device 0: {torch.cuda.get_device_name(0)})"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev = "mps (Apple GPU)"
    else:
        dev = "cpu"
    info(f"Torch         : {torch.__version__}")
    info(f"Backend       : {dev}")
    return dev


def verify_tree():
    problems = 0

    ck_dir = PROJECT_ROOT / "checkpoints"
    if not ck_dir.is_dir():
        fail("checkpoints/ folder missing")
        problems += 1
    else:
        # in the public repo we only require the 'large' checkpoint by default;
        # the rest are optional and can be pulled later with
        #   python download_checkpoints.py --all
        for name in PUBLIC_CHECKPOINTS_DEFAULT:
            expected = EXPECTED_CHECKPOINTS[name]
            f = ck_dir / name
            if not f.exists():
                fail(f"checkpoints/{name} missing -- run: python download_checkpoints.py")
                problems += 1
            elif f.stat().st_size != expected:
                fail(f"checkpoints/{name} wrong size "
                     f"(got {f.stat().st_size}, expected {expected})")
                problems += 1
            else:
                ok(f"checkpoints/{name} ({expected/1e6:.1f} MB)")

    cf_dir = PROJECT_ROOT / "configs"
    for name in EXPECTED_CONFIGS:
        f = cf_dir / name
        if not f.exists():
            fail(f"configs/{name} missing")
            problems += 1
        else:
            ok(f"configs/{name}")

    if (PROJECT_ROOT / "sam2" / "__init__.py").exists():
        ok("sam2/ package present")
    else:
        fail("sam2/ package missing")
        problems += 1

    if (PROJECT_ROOT / "build.py").exists():
        ok("build.py present")
    else:
        fail("build.py missing")
        problems += 1

    blade = PROJECT_ROOT / "videos" / "blade"
    if blade.is_dir():
        jpgs = sorted(p for p in blade.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
        if jpgs:
            ok(f"videos/blade/ has {len(jpgs)} frames")
        else:
            warn("videos/blade/ is empty")

    return problems


def add_to_syspath():
    """Write a .pth so 'import sam2' / 'import build' works from any CWD."""
    try:
        import site
        target = Path(site.getusersitepackages())
        target.mkdir(parents=True, exist_ok=True)
        pth = target / "ai_predictor_propagation.pth"
        pth.write_text(str(PROJECT_ROOT) + "\n")
        ok(f"PROJECT_ROOT wired into Python via {pth}")
    except Exception as e:
        warn(f"could not write .pth: {e}")
        info("Workaround: in your script/notebook do")
        info(f"    import sys; sys.path.insert(0, r'{PROJECT_ROOT}')")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify only, do not pip install anything")
    ap.add_argument("--no-torch", action="store_true",
                    help="skip the torch install step")
    ap.add_argument("--cpu", action="store_true",
                    help="force CPU torch wheel even if NVIDIA is detected")
    ap.add_argument("--cuda", default=DEFAULT_CUDA_TAG,
                    help=f"CUDA wheel tag, default {DEFAULT_CUDA_TAG} (try cu118, cu121, cu124)")
    ap.add_argument("--size", default="large",
                    choices=["tiny", "small", "base_plus", "large"],
                    help="Which checkpoint to download (default: large ~900 MB)")
    ap.add_argument("--skip-download", action="store_true",
                    help="Do not download any checkpoint automatically")
    args = ap.parse_args()

    section("1. Environment")
    env = detect_env()

    section("2. GPU / CUDA")
    cuda_ver = detect_nvidia()
    if cuda_ver:
        info(f"nvidia-smi reports CUDA {cuda_ver}")
    else:
        info("No NVIDIA driver found (this is fine on Mac or CPU machines)")

    section("3. Dependencies")
    if args.check:
        info("--check given -> skipping installs")
    else:
        if not args.no_torch:
            ensure_torch(env, cuda_tag=args.cuda, force_cpu=args.cpu)
        install_other_deps()

    section("4. Checkpoint")
    if args.check or args.skip_download:
        info("skipping checkpoint download")
    else:
        ensure_checkpoint(args.size)

    section("5. Torch backend")
    detect_backend()

    section("6. Project tree check")
    problems = verify_tree()

    section("7. Python path")
    add_to_syspath()

    section("Summary")
    if problems == 0:
        ok("READY. You can now open the notebooks:")
        info("    python -m jupyter lab")
        info("Recommended first notebook: video_predictor_colab.ipynb")
        sys.exit(0)
    else:
        fail(f"NOT READY -- {problems} issue(s) above. Fix them, then re-run:")
        info("    python setup_env.py --check")
        sys.exit(1)


if __name__ == "__main__":
    main()
