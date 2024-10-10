
from mmcv import imread
from mmcv.transforms.loading import LoadImageFromFile
from mmcv.transforms.processing import Resize, RandomResize
import numpy as np
from typing import Optional
import mmcv
import mmengine.fileio as fileio
from mmseg.registry import TRANSFORMS
from mmcv.image.geometric import _scale_size
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


@TRANSFORMS.register_module()
class LoadImageOtherFromFile(LoadImageFromFile):
    def __init__(self, type_name="img_path2", *args, **kwargs):

        self.type_name = type_name
        super(LoadImageOtherFromFile, self).__init__(*args, **kwargs)

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path2']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        if self.to_float32:
            img = img.astype(np.float32)

        results['img_aug'] = img

        # results['img_shape'] = img.shape[:2]
        # results['ori_shape'] = img.shape[:2]
        return results


@TRANSFORMS.register_module()
class MyResize(Resize):
    """Resize images & bbox & seg & keypoints.

    This transform resizes the input image according to ``scale`` or
    ``scale_factor``. Bboxes, seg map and keypoints are then resized with the
    same scale factor.
    if ``scale`` and ``scale_factor`` are both set, it will use ``scale`` to
    resize.

    Required Keys:

    - img
    - gt_bboxes (optional)
    - gt_seg_map (optional)
    - gt_keypoints (optional)

    Modified Keys:

    - img
    - gt_bboxes
    - gt_seg_map
    - gt_keypoints
    - img_shape

    Added Keys:

    - scale
    - scale_factor
    - keep_ratio

    Args:
        scale (int or tuple): Images scales for resizing. Defaults to None
        scale_factor (float or tuple[float]): Scale factors for resizing.
            Defaults to None.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Defaults to False.
        clip_object_border (bool): Whether to clip the objects
            outside the border of the image. In some dataset like MOT17, the gt
            bboxes are allowed to cross the border of images. Therefore, we
            don't need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def __init__(self, **kwargs) -> None:
        super(MyResize, self).__init__(**kwargs)

    def _resize_img_aug(self, results: dict) -> None:
        """Resize images with ``results['scale']``."""

        if results.get('img_aug', None) is not None:
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results['img_aug'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results['img_aug'].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results['img_aug'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
            results['img_aug'] = img


    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes, semantic
        segmentation map and keypoints.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'scale', 'scale_factor', 'img_shape',
            and 'keep_ratio' keys are updated in result dict.
        """

        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['img'].shape[:2]
            results['scale'] = _scale_size(img_shape[::-1], self.scale_factor)  # type: ignore
        self._resize_img(results)
        self._resize_img_aug(results)
        self._resize_bboxes(results)
        self._resize_seg(results)
        self._resize_keypoints(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'scale_factor={self.scale_factor}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'clip_object_border={self.clip_object_border}), '
        repr_str += f'backend={self.backend}), '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@TRANSFORMS.register_module()
class RandomResizeWithAug(RandomResize):
    """Random resize images & bbox & keypoints.

    How to choose the target scale to resize the image will follow the rules
    below:

    - if ``scale`` is a sequence of tuple

    .. math::
        target\\_scale[0] \\sim Uniform([scale[0][0], scale[1][0]])
    .. math::
        target\\_scale[1] \\sim Uniform([scale[0][1], scale[1][1]])

    Following the resize order of weight and height in cv2, ``scale[i][0]``
    is for width, and ``scale[i][1]`` is for height.

    - if ``scale`` is a tuple

    .. math::
        target\\_scale[0] \\sim Uniform([ratio\\_range[0], ratio\\_range[1]])
            * scale[0]
    .. math::
        target\\_scale[1] \\sim Uniform([ratio\\_range[0], ratio\\_range[1]])
            * scale[1]

    Following the resize order of weight and height in cv2, ``ratio_range[0]``
    is for width, and ``ratio_range[1]`` is for height.

    - if ``keep_ratio`` is True, the minimum value of ``target_scale`` will be
      used to set the shorter side and the maximum value will be used to
      set the longer side.

    - if ``keep_ratio`` is False, the value of ``target_scale`` will be used to
      reisze the width and height accordingly.

    Required Keys:

    - img
    - gt_bboxes
    - gt_seg_map
    - gt_keypoints

    Modified Keys:

    - img
    - gt_bboxes
    - gt_seg_map
    - gt_keypoints
    - img_shape

    Added Keys:

    - scale
    - scale_factor
    - keep_ratio

    Args:
        scale (tuple or Sequence[tuple]): Images scales for resizing.
            Defaults to None.
        ratio_range (tuple[float], optional): (min_ratio, max_ratio).
            Defaults to None.
        resize_type (str): The type of resize class to use. Defaults to
            "Resize".
        **resize_kwargs: Other keyword arguments for the ``resize_type``.

    Note:
        By defaults, the ``resize_type`` is "Resize", if it's not overwritten
        by your registry, it indicates the :class:`mmcv.Resize`. And therefore,
        ``resize_kwargs`` accepts any keyword arguments of it, like
        ``keep_ratio``, ``interpolation`` and so on.

        If you want to use your custom resize class, the class should accept
        ``scale`` argument and have ``scale`` attribution which determines the
        resize shape.
    """

    def __init__(
        self, scale: Union[Tuple[int, int], Sequence[Tuple[int, int]]],
            ratio_range: Tuple[float, float] = None,
        resize_type: str = 'MyResize',
        **resize_kwargs
    ) -> None:
        super(RandomResizeWithAug, self).__init__(scale= scale, ratio_range=ratio_range, **resize_kwargs)

        # self.ratio_range = ratio_range

        self.resize_cfg = dict(type=resize_type, **resize_kwargs)

        # create a empty Reisize object
        self.resize = TRANSFORMS.build({'scale': 0, **self.resize_cfg})

    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, ``img``, ``gt_bboxes``, ``gt_semantic_seg``,
            ``gt_keypoints``, ``scale``, ``scale_factor``, ``img_shape``, and
            ``keep_ratio`` keys are updated in result dict.
        """
        results['scale'] = self._random_scale()
        self.resize.scale = results['scale']
        results = self.resize(results)
        return results
