# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset, BaseCDDataset

import os.path as osp
from typing import Callable, Dict, List, Optional, Sequence, Union

import mmengine
import mmengine.fileio as fileio
import numpy as np
from torch.utils.data import Subset

@DATASETS.register_module()
class InriaDataset(BaseCDDataset):
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

    # def load_data_list(self) -> List[dict]:
    #     """Load annotation from directory or annotation file.
    #
    #     Returns:
    #         list[dict]: All data info of dataset.
    #     """
    #     data_list = []
    #     img_dir = self.data_prefix.get('img_path', None)
    #     img_dir2 = self.data_prefix.get('img_path2', None)
    #     ann_dir = self.data_prefix.get('seg_map_path', None)
    #     if osp.isfile(self.ann_file):
    #         lines = mmengine.list_from_file(
    #             self.ann_file, backend_args=self.backend_args)
    #         for line in lines:
    #             img_name = line.strip()
    #             if '.' in osp.basename(img_name):
    #                 img_name, img_ext = osp.splitext(img_name)
    #                 self.img_suffix = img_ext
    #                 self.img_suffix2 = img_ext
    #             data_info = dict(
    #                 img_path=osp.join(img_dir, img_name + self.img_suffix),
    #                 img_path2=osp.join(img_dir2, img_name + self.img_suffix2))
    #
    #             if ann_dir is not None:
    #                 seg_map = img_name + self.seg_map_suffix
    #                 data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
    #             data_info['label_map'] = self.label_map
    #             data_info['reduce_zero_label'] = self.reduce_zero_label
    #             data_info['seg_fields'] = []
    #             data_list.append(data_info)
    #     else:
    #         for img in fileio.list_dir_or_file(
    #                 dir_path=img_dir,
    #                 list_dir=False,
    #                 suffix=self.img_suffix,
    #                 recursive=True,
    #                 backend_args=self.backend_args):
    #
    #             if '.' in osp.basename(img):
    #                 img, img_ext = osp.splitext(img)
    #                 self.img_suffix = img_ext
    #                 self.img_suffix2 = img_ext
    #             data_info = dict(
    #                 img_path=osp.join(img_dir, img + self.img_suffix),
    #                 img_path2=osp.join(img_dir2, img + self.img_suffix2))
    #             if ann_dir is not None:
    #                 seg_map = img + self.seg_map_suffix
    #                 data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
    #             data_info['label_map'] = self.label_map
    #             data_info['reduce_zero_label'] = self.reduce_zero_label
    #             data_info['seg_fields'] = []
    #             data_list.append(data_info)
    #         data_list = sorted(data_list, key=lambda x: x['img_path'])
    #
    #     # data_list = self.get_subset(data_list, fraction=0.5, seed=42)
    #
    #     return data_list

    # 创建数据子集
    # def get_subset(self, data_list, fraction=0.5, seed=42):
    #     np.random.seed(seed)
    #
    #     indices = np.random.choice(len(data_list), int(len(data_list) * fraction), replace=False)
    #     subset = [data_list[i] for i in indices]
    #     # indices = np.random.choice(len(data_list), int(len(data_list) * fraction), replace=False)
    #     # subset = Subset(data_list, indices)
    #     # subset = subset.dataset
    #
    #     return subset