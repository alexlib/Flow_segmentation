import logging
import os
from pathlib import Path

import torch
from hydra_Gokul import compose
from hydra_Gokul.initialize import initialize_config_dir
from hydra_Gokul.core.global_hydra import GlobalHydra as GokulGlobalHydra
from hydra.core.global_hydra import GlobalHydra as CoreGlobalHydra
from hydra_Gokul.utils import instantiate
from omegaconf_Gokul import OmegaConf




def _compose_config(config_file, overrides, job_name):
    cfg_path = Path(config_file)
    if cfg_path.suffix.lower() in {".yaml", ".yml"} and cfg_path.is_file():
        gh = GokulGlobalHydra.instance()
        if gh.is_initialized():
            gh.clear()
        core_gh = CoreGlobalHydra.instance()
        if core_gh.is_initialized():
            core_gh.clear()
        with initialize_config_dir(
            config_dir=str(cfg_path.parent.resolve()),
            job_name=job_name,
            version_base=None,
        ):
            return compose(config_name=cfg_path.name, overrides=overrides)
    return compose(config_name=config_file, overrides=overrides)


def _resolve_config(cfg):
    try:
        OmegaConf.resolve(cfg)
    except Exception:
        from omegaconf import OmegaConf as StdOmegaConf

        StdOmegaConf.resolve(cfg)
    return cfg


def build_segment(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = _compose_config(
        config_file=config_file,
        overrides=hydra_overrides_extra,
        job_name="build_segment",
    )
    cfg = _resolve_config(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    vos_optimized=False,
    **kwargs,
):
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if vos_optimized:
        hydra_overrides = [
            "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictorVOS",
            "++model.compile_image_encoder=True",  # Let sam2_base handle this
        ]

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = _compose_config(
        config_file=config_file,
        overrides=hydra_overrides,
        job_name="build_predictor",
    )
    cfg = _resolve_config(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _hf_download(model_id):
    from huggingface_hub import hf_hub_download

    config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return config_name, ckpt_path


def build_segment_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_segment(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def build_predictor_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")
