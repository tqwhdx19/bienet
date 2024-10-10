# Copyright (c) OpenMMLab. All rights reserved.
from .fcn_head import FCNHead
from .psp_head import PSPHead
from .segformer_head import SegformerHead
from .bieb_head import BiebHead

__all__ = [
    'FCNHead',  'PSPHead', 'SegformerHead',
    'BiebHead'
]
