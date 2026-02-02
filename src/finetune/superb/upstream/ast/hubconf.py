import os

from s3prl.utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert

ROOT = os.path.expanduser("~/ssast/src/pretrain/exp")

FRAME_MODEL = "mask01-tiny-f128-t2-b64-lr5e-4-m400-e75-pretrain_mpmhb-librispeech360-mpmhb1.0-mpg1.0-mpc1.0"
PATCH_MODEL = "mask01-tiny-f16-t16-b64-lr5e-4-m400-e200-pretrain_mpmhb-librispeech360"

FRAME_MODEL_PATH = os.path.join(ROOT, FRAME_MODEL, "models", "best_audio_model.pth")
PATCH_MODEL_PATH = os.path.join(ROOT, PATCH_MODEL, "models", "best_audio_model.pth")

def _check_model_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"SSAST Pretrained model not found at: {path}. Please download it.")

# Frame-based SSAST
# 1s for speech commands, 6s for IEMOCAP, 10s for SID
def ssast_frame_base_1s(ckpt, *args, **kwargs):
    _check_model_exists(FRAME_MODEL_PATH)
    kwargs['model_size'] = 'base_f'
    kwargs['pretrain_path'] = FRAME_MODEL_PATH
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ssast_frame_base_6s(ckpt, *args, **kwargs):
    _check_model_exists(FRAME_MODEL_PATH)
    kwargs['model_size'] = 'base_f'
    kwargs['pretrain_path'] = FRAME_MODEL_PATH
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ssast_frame_base_10s(ckpt, *args, **kwargs):
    _check_model_exists(FRAME_MODEL_PATH)
    kwargs['model_size'] = 'base_f'
    kwargs['pretrain_path'] = FRAME_MODEL_PATH
    kwargs["target_length"] = 1000
    return _UpstreamExpert(ckpt, *args, **kwargs)

# Patch-based SSAST
# 1s for speech commands, 6s for IEMOCAP, 10s for SID
def ssast_patch_base_1s(ckpt, *args, **kwargs):
    _check_model_exists(PATCH_MODEL_PATH)
    kwargs['model_size'] = 'base_p'
    kwargs['pretrain_path'] = PATCH_MODEL_PATH
    kwargs["target_length"] = 100
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ssast_patch_base_6s(ckpt, *args, **kwargs):
    _check_model_exists(PATCH_MODEL_PATH)
    kwargs['model_size'] = 'base_p'
    kwargs['pretrain_path'] = PATCH_MODEL_PATH
    kwargs["target_length"] = 600
    return _UpstreamExpert(ckpt, *args, **kwargs)

def ssast_patch_base_10s(ckpt, *args, **kwargs):
    _check_model_exists(PATCH_MODEL_PATH)
    kwargs['model_size'] = 'base_p'
    kwargs['pretrain_path'] = PATCH_MODEL_PATH
    kwargs["target_length"] = 1000
    return _UpstreamExpert(ckpt, *args, **kwargs)
