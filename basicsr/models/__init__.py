import importlib
from copy import deepcopy
from os import path as osp

from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import MODEL_REGISTRY

__all__ = ['build_model']

_model_folder = osp.dirname(osp.abspath(__file__))
_model_filenames = sorted(
    osp.splitext(osp.basename(v))[0]
    for v in scandir(_model_folder)
    if v.endswith('_model.py')
)

_model_modules = [
    importlib.import_module(f'basicsr.models.{name}')
    for name in _model_filenames
]

def build_model(opt):
    opt = deepcopy(opt)
    model_cls = MODEL_REGISTRY.get(opt['model_type'])
    model = model_cls(opt)
    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model
