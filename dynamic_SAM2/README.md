# Flow segmentation  --  dynamic (SAM 2 video propagation)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AliRKhojasteh/Flow_segmentation/blob/main/dynamic_SAM2/video_predictor_colab.ipynb)

Dynamic companion to the
[Flow_segmentation](https://github.com/AliRKhojasteh/Flow_segmentation) repo,
which hosts the static SAM 1 flow segmentation work. This folder demonstrates
**SAM 2.1 video propagation** on a short blade-tip sequence: click a point
on the first frame, and the mask is tracked through every subsequent frame
without any further input.

Runs as a one-click Colab notebook, or locally on Windows / Linux / macOS
with GPU or CPU.

---

## Try it in Colab

Click the badge above. The notebook's first cell clones the repo, installs
every dependency with the right torch wheel for Colab's GPU, and downloads
the SAM 2.1 hiera-large checkpoint (~900 MB) from HuggingFace. Nothing else
to do -- press Run All.

---

## Run it locally

```
git clone https://github.com/AliRKhojasteh/Flow_segmentation.git
cd Flow_segmentation/dynamic_SAM2
python setup_env.py
```

That command detects the OS (Windows / Linux / macOS / Colab) and the
hardware (NVIDIA GPU / Apple MPS / CPU), installs torch with the matching
wheel, installs every other pip dependency, downloads the SAM 2.1 hiera-large
checkpoint, and prints READY.

Then launch Jupyter:

```
python -m jupyter lab
```

and open `video_predictor_colab.ipynb`.

### Flags on setup_env.py

    --size tiny|small|base_plus|large   # which checkpoint to pull (default: large)
    --skip-download                     # do not fetch any checkpoint
    --cpu                               # force the CPU torch wheel
    --cuda cu118                        # older NVIDIA driver

### Pull more checkpoint sizes later

```
python download_checkpoints.py --all      # every size, ~2.3 GB total
python download_checkpoints.py --size tiny
```

---

## What is inside

    dynamic_SAM2/
    |-- README.md                       this file
    |-- LICENSE                         MIT for this repo, attributes SAM 2 / hydra / omegaconf
    |-- .gitignore                      excludes .pt / .pickle
    |-- setup_env.py                    one-shot bootstrap (env + deps + checkpoint)
    |-- download_checkpoints.py         standalone weight puller
    |-- requirements.txt                pip deps (torch pinned separately by setup_env.py)
    |-- environment.yml                 conda alternative
    |
    |-- build.py                        constructs the SAM 2 video predictor
    |
    |-- video_predictor_colab.ipynb     main notebook (Colab-ready)
    |-- test.ipynb                      tiny import sanity check
    |-- add_remove.ipynb                interactive mask editor; LOCAL only (needs ipympl)
    |
    |-- configs/                        Hydra YAMLs for each SAM 2.1 size
    |-- functions/                      custom automatic_mask_generator + image_predictor
    |-- sam2/                           the SAM 2 package (from facebookresearch/sam2)
    |-- hydra_Gokul/                    patched hydra-core fork used by build.py
    |-- omegaconf_Gokul/                patched omegaconf fork used by build.py
    |-- videos/blade/                   7 sample frames of the blade demo
    |-- pkl/B0001.png                   one sample still for add_remove.ipynb
    |-- checkpoints/                    weights land here on first run (git-ignored)

---

## Target machines

| Machine                            | Backend auto-picked | Setup time    |
|------------------------------------|---------------------|---------------|
| Google Colab (T4 / A100)           | CUDA                | ~1 min + DL   |
| Windows 10 / 11 + NVIDIA GPU       | CUDA                | ~2 min + DL   |
| Linux + NVIDIA GPU                 | CUDA                | ~2 min + DL   |
| macOS (Apple Silicon)              | MPS (Apple GPU)     | ~1 min + DL   |
| Any CPU                            | CPU                 | ~1 min + DL   |

"DL" is the one-time ~900 MB checkpoint download.

---

## Troubleshooting

**Colab says "cannot find build.py".**
The bootstrap cell assumes the repo subfolder is called `dynamic_SAM2`.
If you push to a different folder name, edit the `SUBDIR` variable in the
first code cell of the notebook.

**CUDA detected but `cuda.is_available() == False`.**
Older NVIDIA driver. Run:
```
python setup_env.py --cuda cu118
```

**`add_remove.ipynb` shows a static plot.**
That notebook uses `%matplotlib widget` and needs `ipympl` plus a kernel
restart. Works in local Jupyter; not recommended in Colab.

**Out-of-memory on a small GPU.**
Default config is SAM 2.1 hiera-large (~2 GB VRAM). Switch to tiny in the
notebook:
```python
cfg        = PROJECT_ROOT / "configs" / "sam2.1_hiera_t.yaml"
checkpoint = PROJECT_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
```
then `python download_checkpoints.py --size tiny`.

---

## Citation / attribution

If you use this code for academic work, please cite the underlying SAM 2
paper and the Flow_segmentation repository.

SAM 2 source code in `sam2/` is from Meta AI's public release of
[Segment Anything 2](https://github.com/facebookresearch/sam2) under
the Apache 2.0 licence.
