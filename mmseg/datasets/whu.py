# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset, BaseCDDataset


@DATASETS.register_module()
class WHUDataset(BaseCDDataset):
# class WHUDataset(BaseSegDataset):
    """ISPRS dataset.

    In segmentation map annotation for WHU, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    METAINFO = dict(
        classes=('background', 'building'),
        palette=[[0, 0, 0], [0, 255, 0]])

    def __init__(self,
                 img_suffix='.png',
                 img_suffix2='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:

        super().__init__(
            img_suffix=img_suffix,
            img_suffix2=img_suffix2,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

        # self.img_suffix2 = img_suffix2
