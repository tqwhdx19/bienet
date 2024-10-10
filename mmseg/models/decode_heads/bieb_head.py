# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize
from typing import List, Tuple
from torch import Tensor
from mmseg.utils import ConfigType, SampleList
from ..losses import accuracy

@MODELS.register_module()
class BiebHead(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear',
                 loss_rec=dict(
                     type='mmdet.models.losses.L2Loss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                  **kwargs):

        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        self.convs_second = nn.ModuleList()

        # in_channels=[64, 128, 320, 512], channels=128
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    # concatenation enhanced information
                    in_channels=self.in_channels[i] + self.channels,
                    # in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        # reconstruction
        for i in range(num_inputs):
            self.convs_second.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        # reconstruction
        self.fusion_conv_second = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        # reconstruction output
        self.conv_rec = nn.Conv2d(self.channels, self.out_channels+1, kernel_size=1)

        self.conv_mlp = ConvModule(
                    in_channels=self.out_channels + 1,
                    out_channels=self.out_channels + 1,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)

        self.loss_rec = MODELS.build(loss_rec)
        self.sigmoid = nn.Sigmoid()

        self.number = 0

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []

        # reconstruction
        outs_rec = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs_second[idx]
            middle_out = conv(x)
            outs_rec.append(middle_out)

        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            x_fuse = torch.cat((x, outs_rec[idx].detach()), dim=1)
            outs.append(
                resize(
                    # input=conv(x),
                    input=conv(x_fuse),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        out = self.cls_seg(out)

        outs_rec_resize = []
        for idx in range(len(inputs)):
            middle_out = outs_rec[idx]
            outs_rec_resize.append(
                resize(
                    input=middle_out,
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out_rec = self.fusion_conv_second(torch.cat(outs_rec_resize, dim=1))
        out_rec = self.conv_rec(out_rec)

        # ssim rec
        out_rec = self.sigmoid(out_rec)

        return out, out_rec

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_logits, rec_logits = self.forward(inputs)

        losses = self.loss_by_feat(seg_logits, rec_logits, batch_data_samples)

        return losses

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        seg_logits, rec_logits = self.forward(inputs)
        seg_results = self.predict_by_feat(seg_logits, batch_img_metas)

        return seg_results

    def _stack_batch_rec(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_aug = [
            data_sample.img_aug for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_aug, dim=0)

    def _stack_batch_original_img(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_original = [
            data_sample.original_img for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_original, dim=0)

    def loss_by_feat(self, seg_logits: Tensor,  rec_logits:Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples)
        img_aug_label = self._stack_batch_rec(batch_data_samples)
        original_img = self._stack_batch_original_img(batch_data_samples)
        loss = dict()
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        # reconstruction
        rec_logits = resize(
            input=rec_logits,
            size=img_aug_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        # loss['loss_aug'] = self.loss_rec(rec_logits, img_aug_label, seg_label)
        rec_logits = rec_logits + original_img
        rec_logits = self.conv_mlp(rec_logits)
        loss['loss_aug_ssim_l1'] = self.loss_rec(rec_logits, img_aug_label)

        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)

        return loss

    def predict_by_feat_add_original(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """

        if isinstance(batch_img_metas[0]['img_shape'], torch.Size):
            # slide inference
            size = batch_img_metas[0]['img_shape']
        elif 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape'][:2]
        else:
            size = batch_img_metas[0]['img_shape']

        seg_logits = resize(
            input=seg_logits,
            size=size,
            mode='bilinear',
            align_corners=self.align_corners)

        original_img = batch_img_metas[0]['original_img']
        original_img = original_img/255
        rec_logits = seg_logits + original_img
        rec_logits = self.conv_mlp(rec_logits)

        return rec_logits