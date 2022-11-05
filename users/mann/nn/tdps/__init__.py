from .base import *
from .feature import *
from .label import *
from .initialization import *

_archs = {
    **FEATURE_ARCHS,
    # **LABEL_ARCHS,
}

_builders = {
    **LABEL_MODEL_BUILDER,
    **FEATURE_MODEL_BUILDERS,
}

_init_builders = {
    **INITIALIZATION_BUILDERS,
}

_softmax_archs = SOFTMAX_ARCHS

def get_tdp_layer(num_classes, arch):
    return _archs[arch](num_classes)
    

def get_model(num_classes, arch, init=None, dependency_reduction=None):
    init_tdps = None
    if init is not None:
        init = init.copy()
        type_ = init.pop("type")
        init_tdps = _init_builders[type_].build(num_classes, **init)

    try:
        return _builders[arch].build(num_classes, init_tdps)
    except TypeError:
        return _builders[arch].build(num_classes, init_tdps, system)
