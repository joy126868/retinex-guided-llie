import importlib
from copy import deepcopy
from os import path as osp

from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import ARCH_REGISTRY

__all__ = ['build_network']

_arch_folder = osp.dirname(osp.abspath(__file__))
_arch_filenames = sorted(
    osp.splitext(osp.basename(v))[0]
    for v in scandir(_arch_folder)
    if v.endswith('_arch.py')
)

_arch_modules = [
    importlib.import_module(f'basicsr.archs.{name}')
    for name in _arch_filenames
]


def build_network(opt):
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    net_cls = ARCH_REGISTRY.get(network_type)
    net = net_cls(**opt)
    logger = get_root_logger()
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net
