# Flow segmentation  --  dynamic

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AliRKhojasteh/Flow_segmentation/blob/main/dynamic_SAM2/video_predictor_colab.ipynb)

Time-resolved companion to the main
[Flow_segmentation](https://github.com/AliRKhojasteh/Flow_segmentation) repo.
The static pipeline segments one PIV snapshot at a time; this one propagates
a user-selected mask through an entire image sequence. One click on the
first frame, and the region of interest (a blade, bubble, vortex core,
particle cluster, shear layer, wake, ...) is tracked across every subsequent
frame without further annotation.

Useful for:
* isolating a moving object in time-resolved PIV / PTV recordings,
* producing consistent per-frame masks for conditional statistics,
* building ground-truth for downstream training without click-by-click work.

Runs as a one-click Colab notebook, or locally on Windows / Linux / macOS
with GPU or CPU.

---

## Try it in Colab

Click the badge above. The notebook's first cell clones the repo, installs
every dependency with the right torch wheel for Colab's GPU, pulls the
hiera-large checkpoint (~900 MB) from HuggingFace, and runs. Press Run All.

## Run it locally

```
git clone https://github.com/AliRKhojasteh/Flow_segmentation.git
cd Flow_segmentation/dynamic_SAM2
python setup_env.py
python -m jupyter lab
```

`setup_env.py` detects the OS, picks the right torch wheel (CUDA on
Windows/Linux with NVIDIA, MPS on Apple Silicon, CPU otherwise), installs
the rest of the dependencies, downloads the default hiera-large checkpoint,
and prints READY.

Flags:

    --size tiny|small|base_plus|large   # which checkpoint to pull (default: large)
    --skip-download                     # do not fetch any checkpoint
    --cpu                               # force the CPU torch wheel
    --cuda cu118                        # older NVIDIA driver

Pull more sizes later:

```
python download_checkpoints.py --all      # every size, ~2.3 GB total
python download_checkpoints.py --size tiny
```

---

## What is inside

    dynamic_SAM2/
    |-- README.md
    |-- LICENSE
    |-- .gitignore
    |-- setup_env.py                    one-shot bootstrap
    |-- download_checkpoints.py         standalone weight puller
    |-- requirements.txt, environment.yml
    |
    |-- build.py                        constructs the video predictor
    |-- video_predictor_colab.ipynb     main notebook, Colab-ready
    |-- test.ipynb                      import sanity check
    |-- add_remove.ipynb                interactive mask editor, local Jupyter only
    |
    |-- configs/                        Hydra YAMLs for each model size
    |-- functions/                      custom automatic mask generator + image predictor
    |-- sam2/                           the upstream model package
    |-- hydra_Gokul/, omegaconf_Gokul/  patched forks used by build.py
    |-- videos/blade/                   7 sample frames of a blade recording
    |-- pkl/B0001.png                   one still frame for add_remove.ipynb
    |-- checkpoints/                    weights land here on first run (git-ignored)

---

## Target machines

| Machine                            | Backend auto-picked | Setup time    |
|------------------------------------|---------------------|---------------|
| Google Colab (T4 / A100)           | CUDA                | ~1 min + DL   |
| Windows 10/11 + NVIDIA GPU         | CUDA                | ~2 min + DL   |
| Linux + NVIDIA GPU                 | CUDA                | ~2 min + DL   |
| macOS (Apple Silicon)              | MPS (Apple GPU)     | ~1 min + DL   |
| Any CPU                            | CPU                 | ~1 min + DL   |

"DL" is the one-time ~900 MB checkpoint download.

---

## Troubleshooting

**Colab says "cannot find build.py".** The bootstrap cell assumes the repo
subfolder is called `dynamic_SAM2`. If you rename it, edit the `SUBDIR`
variable in the first code cell of the notebook.

**CUDA detected but `cuda.is_available() == False`.** Older NVIDIA driver:
```
python setup_env.py --cuda cu118
```

**`add_remove.ipynb` shows a static plot.** That notebook uses
`%matplotlib widget` and needs `ipympl` plus a kernel restart. Works in
local Jupyter; not recommended in Colab.

**Out-of-memory on a small GPU.** Default is the large model (~2 GB VRAM).
Switch to tiny in the notebook:
```python
cfg        = PROJECT_ROOT / "configs" / "sam2.1_hiera_t.yaml"
checkpoint = PROJECT_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
```
then `python download_checkpoints.py --size tiny`.

---

## Citation

If you use this code in academic work, please cite the
[Flow_segmentation](https://github.com/AliRKhojasteh/Flow_segmentation) repository
and the underlying Segment Anything 2 paper
([facebookresearch/sam2](https://github.com/facebookresearch/sam2), Apache 2.0).
