from sslmethods.base import BaseModel
from sslmethods.simclr import SimCLR
from sslmethods.moco import MoCo
from sslmethods.byol import BYOL
from sslmethods.simsiam import SimSiam
from sslmethods.swav import SwAV
from sslmethods.dino import DINO
from sslmethods.mae import MAE

_method_class_map = {
    'base': BaseModel,
    'simclr': SimCLR,
    'moco': MoCo,
    'byol': BYOL,
    'simsiam': SimSiam,
    'swav': SwAV,
    'dino': DINO,
    'mae': MAE
}


def get_method_class(key):
    if key in _method_class_map:
        return _method_class_map[key]
    else:
        raise ValueError('Invalid method: {}'.format(key))
