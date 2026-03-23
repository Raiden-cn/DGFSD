
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from functools import partial
from pcdet.models.model_utils.dgfsd_utils import post_act_block_sparse_3d, post_act_block_sparse_2d
from pcdet.models.model_utils.dgfsd_utils import SparseBasicBlock3D, SparseBasicBlock2D

from ...utils.spconv_utils import replace_feature, spconv
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils.loss_utils import focal_loss_sparse, FocalLossCenterNet


from torch.nn.init import kaiming_normal_
from ..model_utils import centernet_utils

norm_fn_1d = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
norm_fn_2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)

class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, init_bias=-2.19, use_bias=False, norm_func=None):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    nn.BatchNorm2d(input_channels) if norm_func is None else norm_func(input_channels), # 
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict

class SEDLayer(spconv.SparseModule):
    def __init__(self, dim: int, down_kernel_size: list, down_stride: list, num_SBB: list, indice_key, xy_only=False):
        super().__init__()

        block = SparseBasicBlock2D if xy_only else SparseBasicBlock3D
        post_act_block = post_act_block_sparse_2d if xy_only else post_act_block_sparse_3d

        self.encoder = nn.ModuleList(
            [spconv.SparseSequential(
                *[block(dim, indice_key=f"{indice_key}_0") for _ in range(num_SBB[0])])]
        )

        num_levels = len(down_stride)
        for idx in range(1, num_levels):
            cur_layers = [
                post_act_block(
                    dim, dim, down_kernel_size[idx], down_stride[idx], down_kernel_size[idx] // 2,
                    conv_type='spconv', indice_key=f'spconv_{indice_key}_{idx}'),

                *[block(dim, indice_key=f"{indice_key}_{idx}") for _ in range(num_SBB[idx])]
            ]
            self.encoder.append(spconv.SparseSequential(*cur_layers))

        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        for idx in range(num_levels - 1, 0, -1):
            self.decoder.append(
                post_act_block(
                    dim, dim, down_kernel_size[idx],
                    conv_type='inverseconv', indice_key=f'spconv_{indice_key}_{idx}'))
            self.decoder_norm.append(norm_fn_1d(dim))

    def forward(self, x):
        feats = []
        for conv in self.encoder:
            x = conv(x)
            feats.append(x)

        x = feats[-1]
        for deconv, norm, up_x in zip(self.decoder, self.decoder_norm, feats[:-1][::-1]):
            x = deconv(x)
            x = replace_feature(x, norm(x.features + up_x.features))
        return x

class BaseDGFSD(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, class_names, voxel_size, point_cloud_range, **kwargs):
        super().__init__()

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        sed_dim = model_cfg.SED_FEATURE_DIM
        sed_num_layers = model_cfg.SED_NUM_LAYERS
        sed_num_SBB = model_cfg.SED_NUM_SBB
        sed_down_kernel_size = model_cfg.SED_DOWN_KERNEL_SIZE
        sed_down_stride = model_cfg.SED_DOWN_STRIDE
        assert sed_down_stride[0] == 1
        assert len(sed_num_SBB) == len(sed_down_kernel_size) == len(sed_down_stride)

        afd_dim = model_cfg.AFD_FEATURE_DIM
        afd_num_layers = model_cfg.AFD_NUM_LAYERS
        afd_num_SBB = model_cfg.AFD_NUM_SBB
        afd_down_kernel_size = model_cfg.AFD_DOWN_KERNEL_SIZE
        afd_down_stride = model_cfg.AFD_DOWN_STRIDE
        assert afd_down_stride[0] == 1
        assert len(afd_num_SBB) == len(afd_down_stride)

        post_act_block = post_act_block_sparse_3d
        self.stem = spconv.SparseSequential(
            post_act_block(input_channels, 16, 3, 1, 1, indice_key='subm1', conv_type='subm'),

            SparseBasicBlock3D(16, indice_key='conv1'),
            SparseBasicBlock3D(16, indice_key='conv1'),
            post_act_block(16, 32, 3, 2, 1, indice_key='spconv1', conv_type='spconv'),

            SparseBasicBlock3D(32, indice_key='conv2'),
            SparseBasicBlock3D(32, indice_key='conv2'),
            post_act_block(32, 64, 3, 2, 1, indice_key='spconv2', conv_type='spconv'),

            SparseBasicBlock3D(64, indice_key='conv3'),
            SparseBasicBlock3D(64, indice_key='conv3'),
            SparseBasicBlock3D(64, indice_key='conv3'),
            SparseBasicBlock3D(64, indice_key='conv3'),
            post_act_block(64, sed_dim, 3, (1, 2, 2), 1, indice_key='spconv3', conv_type='spconv'),
        )

        self.sed_layers = nn.ModuleList()
        for idx in range(sed_num_layers):
            layer = SEDLayer(
                sed_dim, sed_down_kernel_size, sed_down_stride, sed_num_SBB,
                indice_key=f'sedlayer{idx}', xy_only=kwargs.get('xy_only', False))
            self.sed_layers.append(layer)

        self.dense_encoder = spconv.SparseSequential(
            post_act_block(sed_dim, afd_dim, (3, 1, 1), (2, 1, 1), 0, indice_key='spconv4', conv_type='spconv'),
            post_act_block(afd_dim, afd_dim, (3, 1, 1), (2, 1, 1), 0, indice_key='spconv5', conv_type='spconv'),
        )

        self.adaptive_feature_diffusion = model_cfg.get('AFD', False)
        if self.adaptive_feature_diffusion:
            self.class_names = class_names
            self.voxel_size = voxel_size
            self.point_cloud_range = point_cloud_range
            self.fg_thr = model_cfg['FG_THRESHOLD']
            self.featmap_stride = model_cfg['FEATMAP_STRIDE']
            self.group_pooling_kernel_size = model_cfg['GREOUP_POOLING_KERNEL_SIZE']
            self.detach_feature = model_cfg.get('DETACH_FEATURE', False)

            self.group_class_names = []
            for names in model_cfg['GROUP_CLASS_NAMES']:
                self.group_class_names.append([x for x in names if x in class_names])

            self.cls_conv = spconv.SparseSequential(
                spconv.SubMConv2d(afd_dim, afd_dim, 3, stride=1, padding=1, bias=False, indice_key='conv_cls'),
                norm_fn_1d(afd_dim),
                nn.ReLU(),
                spconv.SubMConv2d(afd_dim, len(self.group_class_names), 1, bias=True, indice_key='cls_out')
            )
            self.forward_ret_dict = {}

        self.afd_layers = nn.ModuleList()
        for idx in range(afd_num_layers):
            layer = SEDLayer(
                afd_dim, afd_down_kernel_size, afd_down_stride, afd_num_SBB,
                indice_key=f'afdlayer{idx}', xy_only=True)
            self.afd_layers.append(layer)

        self.num_point_features = afd_dim
        self.init_weights()

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, (spconv.SubMConv2d, spconv.SubMConv3d)):
                nn.init.kaiming_normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if self.adaptive_feature_diffusion:
            self.cls_conv[-1].bias.data.fill_(-2.19)

    def assign_target(self, batch_spatial_indices, batch_gt_boxes):
        all_names = np.array(['bg', *self.class_names])
        inside_box_target = batch_spatial_indices.new_zeros((len(self.group_class_names), batch_spatial_indices.shape[0]))

        for gidx, names in enumerate(self.group_class_names):
            batch_inside_box_mask = []
            for bidx in range(len(batch_gt_boxes)):
                spatial_indices = batch_spatial_indices[batch_spatial_indices[:, 0] == bidx][:, [2, 1]]
                points = spatial_indices.clone() + 0.5
                points[:, 0] = points[:, 0] * self.featmap_stride * self.voxel_size[0] + self.point_cloud_range[0]
                points[:, 1] = points[:, 1] * self.featmap_stride * self.voxel_size[1] + self.point_cloud_range[1]
                points = torch.cat([points, points.new_zeros((points.shape[0], 1))], dim=-1) # z坐标置0

                gt_boxes = batch_gt_boxes[bidx].clone()
                gt_boxes = gt_boxes[(gt_boxes[:, 3] > 0) & (gt_boxes[:, 4] > 0)]
                gt_class_names = all_names[gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []
                for _, name in enumerate(gt_class_names):
                    if name in names:
                        gt_boxes_single_head.append(gt_boxes[_])

                inside_box_mask = points.new_zeros((points.shape[0]))
                if len(gt_boxes_single_head) > 0:
                    boxes = torch.stack(gt_boxes_single_head)[:, :7] # 10 --> 7
                    boxes[:, 2] = 0 # setting box center of z value to zero
                    inside_box_mask[roiaware_pool3d_utils.points_in_boxes_gpu(points[None], boxes[None])[0] > -1] = 1
                batch_inside_box_mask.append(inside_box_mask)
            inside_box_target[gidx] = torch.cat(batch_inside_box_mask)
        return inside_box_target

    def get_loss(self):
        spatial_indices = self.forward_ret_dict['spatial_indices'] # [11220, 3]
        batch_size = self.forward_ret_dict['batch_size'] # int 2
        batch_index = spatial_indices[:, 0]

        inside_box_pred = self.forward_ret_dict['inside_box_pred'] # 前景score [3, 11220]
        inside_box_target = self.forward_ret_dict['inside_box_target'] # 二值 1 or 0 [3, 11220]
        inside_box_pred = torch.cat([inside_box_pred[:, batch_index == bidx] for bidx in range(batch_size)], dim=1)
        inside_box_pred = torch.clamp(inside_box_pred.sigmoid(), min=1e-4, max=1 - 1e-4)

        cls_loss = 0.0
        recall_dict = {}
        for gidx in range(len(self.group_class_names)):
            group_cls_loss = focal_loss_sparse(inside_box_pred[gidx], inside_box_target[gidx].float())
            cls_loss += group_cls_loss

            fg_mask = inside_box_target[gidx] > 0
            pred_mask = inside_box_pred[gidx][fg_mask] > self.fg_thr
            recall_dict[f'afd_recall_{gidx}'] = (pred_mask.sum() / fg_mask.sum().clamp(min=1.0)).item()
            recall_dict[f'afd_cls_loss_{gidx}'] = group_cls_loss.item()

        return cls_loss, recall_dict

    def to_bev(self, x):
        x = self.dense_encoder(x)

        features = x.features
        indices = x.indices[:, [0, 2, 3]]
        spatial_shape = x.spatial_shape[1:]

        indices_unique, _inv = torch.unique(indices, dim=0, return_inverse=True)
        features_unique = features.new_zeros((indices_unique.shape[0], features.shape[1]))
        features_unique.index_add_(0, _inv, features)

        x = spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=spatial_shape,
            batch_size=x.batch_size
        )
        return x

    def feature_diffusion(self, x, batch_dict):
        if not self.adaptive_feature_diffusion:
            return x

        detached_x = x
        if self.detach_feature:
            detached_x = spconv.SparseConvTensor(
                features=x.features.detach(),
                indices=x.indices,
                spatial_shape=x.spatial_shape,
                batch_size=x.batch_size
            )

        inside_box_pred = self.cls_conv(detached_x).features.permute(1, 0)

        if self.training:
            inside_box_target = self.assign_target(x.indices, batch_dict['gt_boxes'])
            self.forward_ret_dict['batch_size'] = x.batch_size
            self.forward_ret_dict['spatial_indices'] = x.indices
            self.forward_ret_dict['inside_box_pred'] = inside_box_pred
            self.forward_ret_dict['inside_box_target'] = inside_box_target

        group_inside_mask = inside_box_pred.sigmoid() > self.fg_thr
        bg_mask = ~group_inside_mask.max(dim=0, keepdim=True)[0] # [1, N] if have bg=True
        group_inside_mask = torch.cat([group_inside_mask, bg_mask], dim=0) # [4, 17710]

        one_mask = x.features.new_zeros((x.batch_size, 1, x.spatial_shape[0], x.spatial_shape[1]))
        for gidx, inside_mask in enumerate(group_inside_mask):
            selected_indices = x.indices[inside_mask] 
            single_one_mask = spconv.SparseConvTensor(
                features=x.features.new_ones(selected_indices.shape[0], 1),
                indices=selected_indices,
                spatial_shape=x.spatial_shape,
                batch_size=x.batch_size
            ).dense()
            pooling_size = self.group_pooling_kernel_size[gidx]
            single_one_mask = F.max_pool2d(single_one_mask, kernel_size=pooling_size, stride=1, padding=pooling_size // 2) # simulating diffusion
            one_mask = torch.maximum(one_mask, single_one_mask) # 

        zero_indices = (one_mask[:, 0] > 0).nonzero().int()
        zero_features = x.features.new_zeros((len(zero_indices), x.features.shape[1]))
        cat_indices = torch.cat([x.indices, zero_indices], dim=0)
        cat_features = torch.cat([x.features, zero_features], dim=0)
        indices_unique, _inv = torch.unique(cat_indices, dim=0, return_inverse=True)
        features_unique = x.features.new_zeros((indices_unique.shape[0], x.features.shape[1]))
        features_unique.index_add_(0, _inv, cat_features)

        x = spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=x.spatial_shape,
            batch_size=x.batch_size
        )
        return x

    def forward(self, batch_dict):
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.stem(x)
        for layer in self.sed_layers:
            x = layer(x)

        x = self.to_bev(x)
        x = self.feature_diffusion(x, batch_dict)
        for layer in self.afd_layers:
            x = layer(x)

        batch_dict.update({'spatial_features_2d': x})
        return batch_dict

class PillarDGFSD(BaseDGFSD): # for nuscenes
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        kwargs['xy_only'] = True
        super().__init__(model_cfg, input_channels, grid_size, **kwargs)
        self.sparse_shape = grid_size[[1, 0]]
        sed_dim = model_cfg.SED_FEATURE_DIM
        afd_dim = model_cfg.AFD_FEATURE_DIM
        
        self.detach_feature = model_cfg.get('DETACH_FEATURE', False)
        self.class_names = kwargs.get('class_names', None)
        self.voxel_size = kwargs.get('voxel_size', None)
        self.point_cloud_range = kwargs.get('point_cloud_range', None)
        del self.stem
        self.dense_encoder = post_act_block_sparse_2d(
            sed_dim, afd_dim, 3, 2, 1, conv_type='spconv', indice_key='dense_encoder')

        # dense head -------------------------
        cur_head_dict = {}
        self.class_names_each_head = []
        for cur_class_names in model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names])
        self.cls_heads_list = nn.ModuleList()
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=model_cfg.NUM_HM_CONV)
            self.cls_heads_list.append(
                SeparateHead(
                    input_channels=afd_dim,
                    sep_head_dict=cur_head_dict,
                    init_bias=-2.19,
                    use_bias=model_cfg.get('USE_BIAS_BEFORE_NORM', False),
                    # norm_func=norm_func
                )
            )
        self.forward_ret_dict = {}
        #----------------------------------------

        self.init_weights()
        self.build_losses()
        self.model_cfg = model_cfg
        
    def build_losses(self):
        self.add_module('hm_loss_func', FocalLossCenterNet())

    def forward(self, batch_dict):
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        
        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords[:, [0, 2, 3]].int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        for layer in self.sed_layers:
            x = layer(x)
        x = self.dense_encoder(x)
        x = self.dense_knowledge_injection(x, batch_dict)
        x = self.feature_diffusion(x, batch_dict)

        for layer in self.afd_layers:
            x = layer(x)

        batch_dict.update({'spatial_features_2d': x})
        return batch_dict

    def feature_diffusion(self, x, batch_dict):
        if not self.adaptive_feature_diffusion:
            return x
        detached_x = x
        if self.detach_feature:
            detached_x = spconv.SparseConvTensor(
                features=x.features.detach(),
                indices=x.indices,
                spatial_shape=x.spatial_shape,
                batch_size=x.batch_size
            )

        inside_box_pred = self.cls_conv(detached_x).features.permute(1, 0)

        if self.training:
            inside_box_target = self.assign_target_sparse(x.indices, batch_dict['gt_boxes'])
            self.forward_ret_dict['batch_size'] = x.batch_size
            self.forward_ret_dict['spatial_indices'] = x.indices
            self.forward_ret_dict['inside_box_pred'] = inside_box_pred # predict value foreground score
            self.forward_ret_dict['inside_box_target'] = inside_box_target

        group_inside_mask = inside_box_pred.sigmoid() > self.fg_thr
        bg_mask = ~group_inside_mask.max(dim=0, keepdim=True)[0]
        group_inside_mask = torch.cat([group_inside_mask, bg_mask], dim=0)

        one_mask = x.features.new_zeros((x.batch_size, 1, x.spatial_shape[0], x.spatial_shape[1]))
        for gidx, inside_mask in enumerate(group_inside_mask):
            selected_indices = x.indices[inside_mask]
            single_one_mask = spconv.SparseConvTensor(
                features=x.features.new_ones(selected_indices.shape[0], 1),
                indices=selected_indices,
                spatial_shape=x.spatial_shape,
                batch_size=x.batch_size
            ).dense()
            pooling_size = self.group_pooling_kernel_size[gidx]
            single_one_mask = F.max_pool2d(single_one_mask, kernel_size=pooling_size, stride=1, padding=pooling_size // 2)
            one_mask = torch.maximum(one_mask, single_one_mask)

        zero_indices = (one_mask[:, 0] > 0).nonzero().int()
        zero_features = x.features.new_zeros((len(zero_indices), x.features.shape[1]))

        cat_indices = torch.cat([x.indices, zero_indices], dim=0)
        cat_features = torch.cat([x.features, zero_features], dim=0)
        indices_unique, _inv = torch.unique(cat_indices, dim=0, return_inverse=True)
        features_unique = x.features.new_zeros((indices_unique.shape[0], x.features.shape[1]))
        features_unique.index_add_(0, _inv, cat_features)

        x = spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=x.spatial_shape,
            batch_size=x.batch_size
        )
        return x

    def dense_knowledge_injection(self, x, batch_dict):
        detached_x = x
        if self.detach_feature:
            detached_x = spconv.SparseConvTensor(
                features = x.features.detach(),
                indices = x.indices,
                spatial_shape = x.spatial_shape,
                batch_size = x.batch_size
            )
        x_dense = detached_x.dense() # [1, 128, 180, 180]

        # 1. predicte headmap
        B, C, _, _ = x_dense.size()

        k = self.model_cfg.K # TopK-K value
        hm_dicts = []
        for head in self.cls_heads_list:
            hm_dicts.append(head(x_dense))

        # auxiliary dense head assign
        if self.training:
            dense_target_dict = self.assign_target(
                batch_dict['gt_boxes'], feature_map_size=x_dense.size()[2:],
                feature_map_stride=batch_dict.get('spatial_features_2d_strides', None)
            )
            self.forward_ret_dict['dense_target_dict'] = dense_target_dict

        self.forward_ret_dict['hm_dicts'] = hm_dicts

        # 2. locate center point 
        centerpoints = []
        for idx, hm_dict in enumerate(hm_dicts):
            heatmap = hm_dict['hm'].sigmoid()
            scores, inds, class_ids, ys, xs = centernet_utils._topk(heatmap, K=k)
            centerpoint = torch.stack([ys, xs], dim=2) # [b, c, ys, xs]
            centerpoints.append(centerpoint)
        centerpoints = torch.cat(centerpoints, dim=1) # [b, num_cp, [ys,xs]]

        # 3. dense knowledege injection
        sparse_indices = []
        num_points = centerpoints.size(1)
        cp_feature =  x_dense.new_zeros(B, C, num_points)
        for bs_index in range(B):
            cur_bev =  x_dense[bs_index]
            cur_cp = centerpoints[bs_index].long()
            cp_feature[bs_index] = cur_bev[:, cur_cp[:,0], cur_cp[:,1]]
            sparse_indices.append(torch.cat([(x_dense.new_ones(num_points) * bs_index)[:, None], cur_cp], dim=1))

        sparse_indices = torch.cat(sparse_indices).int()
        x = x.replace_feature(torch.cat([x.features, cp_feature.contiguous().view(-1, C)])) # TODO check: cp_feature.permute(0, 2, 1).contiguous().view(-1, C) 
        x.indices = torch.cat([x.indices, sparse_indices])

        return x

    def assign_target(self, gt_boxes, feature_map_size=None, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)
            range_image_polar: (B, 3, H, W)
            feature_map_size: (2) [H, W]
            spatial_cartesian: (B, 4, H, W)
        Returns:
            
        """
        feature_map_size = feature_map_size[::-1]  # [H, W] ==> [x, y]
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'inds': [],
            'masks': [],
            'heatmap_masks': [],
            'target_boxes_src': [], 
        }

        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head): # head-wise match cls name
            heatmap_list, inds_list, masks_list, target_boxes_src_list = [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names: # match current head
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0: 
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap, _, inds, mask, ret_boxes_src = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.cpu(),
                    feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device)) 
                target_boxes_src_list.append(ret_boxes_src.to(gt_boxes_single_head.device))

            ret_dict['heatmaps'].append(torch.stack(heatmap_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
            ret_dict['target_boxes_src'].append(torch.stack(target_boxes_src_list, dim=0))
        return ret_dict

    def assign_target_of_single_head(self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500, gaussian_overlap=0.1, min_radius=2):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()
        ret_boxes_src = gt_boxes.new_zeros(num_max_objs, gt_boxes.shape[-1])
        ret_boxes_src[:gt_boxes.shape[0]] = gt_boxes

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap) 
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue 

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())

            inds[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k] = 1

            ret_boxes[k, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]

        return heatmap, ret_boxes, inds, mask, ret_boxes_src

    def assign_target_sparse(self, batch_spatial_indices, batch_gt_boxes):
        all_names = np.array(['bg', *self.class_names])
        inside_box_target = batch_spatial_indices.new_zeros((len(self.group_class_names), batch_spatial_indices.shape[0]))

        for gidx, names in enumerate(self.group_class_names):
            batch_inside_box_mask = []
            for bidx in range(len(batch_gt_boxes)):
                spatial_indices = batch_spatial_indices[batch_spatial_indices[:, 0] == bidx][:, [2, 1]]
                points = spatial_indices.clone() + 0.5
                points[:, 0] = points[:, 0] * self.featmap_stride * self.voxel_size[0] + self.point_cloud_range[0]
                points[:, 1] = points[:, 1] * self.featmap_stride * self.voxel_size[1] + self.point_cloud_range[1]
                points = torch.cat([points, points.new_zeros((points.shape[0], 1))], dim=-1)

                gt_boxes = batch_gt_boxes[bidx].clone()
                gt_boxes = gt_boxes[(gt_boxes[:, 3] > 0) & (gt_boxes[:, 4] > 0)]
                gt_class_names = all_names[gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []
                for _, name in enumerate(gt_class_names):
                    if name in names:
                        gt_boxes_single_head.append(gt_boxes[_])

                inside_box_mask = points.new_zeros((points.shape[0]))
                if len(gt_boxes_single_head) > 0:
                    boxes = torch.stack(gt_boxes_single_head)[:, :7]
                    boxes[:, 2] = 0
                    inside_box_mask[roiaware_pool3d_utils.points_in_boxes_gpu(points[None], boxes[None])[0] > -1] = 1
                batch_inside_box_mask.append(inside_box_mask)
            inside_box_target[gidx] = torch.cat(batch_inside_box_mask)
        return inside_box_target

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['hm_dicts']
        target_dict = self.forward_ret_dict['dense_target_dict']

        tb_dict = {}
        loss_dense = 0
        
        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = torch.clamp(pred_dict['hm'].sigmoid(), min=1e-4, max=1 - 1e-4)
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dict['heatmaps'][idx])
            hm_loss *= self.model_cfg.DENSE_LOSS_WEIGHTS
            loss_dense += hm_loss
            tb_dict['dense_loss_head_%d' % idx] = hm_loss.item()

        # sparse get_loss
        spatial_indices = self.forward_ret_dict['spatial_indices'] # [11220, 3]
        batch_size = self.forward_ret_dict['batch_size'] # int 2
        batch_index = spatial_indices[:, 0]

        inside_box_pred = self.forward_ret_dict['inside_box_pred'] # foreground score [3, 11220]
        inside_box_target = self.forward_ret_dict['inside_box_target'] # binary value  1 or 0 [3, 11220]
        inside_box_pred = torch.cat([inside_box_pred[:, batch_index == bidx] for bidx in range(batch_size)], dim=1)
        inside_box_pred = torch.clamp(inside_box_pred.sigmoid(), min=1e-4, max=1 - 1e-4)

        for gidx in range(len(self.group_class_names)):
            group_cls_loss = focal_loss_sparse(inside_box_pred[gidx], inside_box_target[gidx].float())
            loss_dense += group_cls_loss
            fg_mask = inside_box_target[gidx] > 0
            pred_mask = inside_box_pred[gidx][fg_mask] > self.fg_thr
            tb_dict[f'afd_recall_{gidx}'] = (pred_mask.sum() / fg_mask.sum().clamp(min=1.0)).item()
            tb_dict[f'afd_cls_loss_{gidx}'] = group_cls_loss.item()

        return loss_dense, tb_dict

class VoxelDGFSD(nn.Module): # for av2
    def __init__(self, model_cfg, input_channels, grid_size, class_names, voxel_size, point_cloud_range, **kwargs):
        super().__init__()

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        sed_dim = model_cfg.SED_FEATURE_DIM
        sed_num_layers = model_cfg.SED_NUM_LAYERS
        sed_num_SBB = model_cfg.SED_NUM_SBB
        sed_down_kernel_size = model_cfg.SED_DOWN_KERNEL_SIZE
        sed_down_stride = model_cfg.SED_DOWN_STRIDE
        assert sed_down_stride[0] == 1
        assert len(sed_num_SBB) == len(sed_down_kernel_size) == len(sed_down_stride)

        afd_dim = model_cfg.AFD_FEATURE_DIM
        afd_num_layers = model_cfg.AFD_NUM_LAYERS
        afd_num_SBB = model_cfg.AFD_NUM_SBB
        afd_down_kernel_size = model_cfg.AFD_DOWN_KERNEL_SIZE
        afd_down_stride = model_cfg.AFD_DOWN_STRIDE
        assert afd_down_stride[0] == 1
        assert len(afd_num_SBB) == len(afd_down_stride)

        post_act_block = post_act_block_sparse_3d
        self.stem = spconv.SparseSequential(
            post_act_block(input_channels, 16, 3, 1, 1, indice_key='subm1', conv_type='subm'),

            SparseBasicBlock3D(16, indice_key='conv1'),
            SparseBasicBlock3D(16, indice_key='conv1'),
            post_act_block(16, 32, 3, 2, 1, indice_key='spconv1', conv_type='spconv'),

            SparseBasicBlock3D(32, indice_key='conv2'),
            SparseBasicBlock3D(32, indice_key='conv2'),
            post_act_block(32, 64, 3, 2, 1, indice_key='spconv2', conv_type='spconv'),

            SparseBasicBlock3D(64, indice_key='conv3'),
            SparseBasicBlock3D(64, indice_key='conv3'),
            SparseBasicBlock3D(64, indice_key='conv3'),
            SparseBasicBlock3D(64, indice_key='conv3'),
            post_act_block(64, sed_dim, 3, (1, 2, 2), 1, indice_key='spconv3', conv_type='spconv'),
        )

        self.sed_layers = nn.ModuleList()
        for idx in range(sed_num_layers):
            layer = SEDLayer(
                sed_dim, sed_down_kernel_size, sed_down_stride, sed_num_SBB,
                indice_key=f'sedlayer{idx}', xy_only=kwargs.get('xy_only', False))
            self.sed_layers.append(layer)

        self.dense_encoder = spconv.SparseSequential(
            post_act_block(sed_dim, afd_dim, (3, 1, 1), (2, 1, 1), 0, indice_key='spconv4', conv_type='spconv'),
            post_act_block(afd_dim, afd_dim, (3, 1, 1), (2, 1, 1), 0, indice_key='spconv5', conv_type='spconv'),
        )

        self.adaptive_feature_diffusion = model_cfg.get('AFD', False)
        if self.adaptive_feature_diffusion:
            self.class_names = class_names
            self.voxel_size = voxel_size
            self.point_cloud_range = point_cloud_range
            self.fg_thr = model_cfg['FG_THRESHOLD']
            self.featmap_stride = model_cfg['FEATMAP_STRIDE']
            self.group_pooling_kernel_size = model_cfg['GREOUP_POOLING_KERNEL_SIZE']
            self.detach_feature = model_cfg.get('DETACH_FEATURE', False)
            self.group_class_names = []
            for names in model_cfg['GROUP_CLASS_NAMES']:
                self.group_class_names.append([x for x in names if x in class_names])

            self.cls_conv = spconv.SparseSequential(
                spconv.SubMConv2d(afd_dim, afd_dim, 3, stride=1, padding=1, bias=False, indice_key='conv_cls'),
                norm_fn_1d(afd_dim),
                nn.ReLU(),
                spconv.SubMConv2d(afd_dim, len(self.group_class_names), 1, bias=True, indice_key='cls_out')
            )
            self.forward_ret_dict = {}

        # dense head
        cur_head_dict = {}
        self.class_names_each_head = []
        for cur_class_names in model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names])
        self.cls_heads_list = nn.ModuleList()
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=model_cfg.NUM_HM_CONV)
            self.cls_heads_list.append(
                SeparateHead(
                    input_channels=afd_dim,
                    sep_head_dict=cur_head_dict,
                    init_bias=-2.19,
                    use_bias=model_cfg.get('USE_BIAS_BEFORE_NORM', False),
                    # norm_func=norm_func
                )
            )
        self.forward_ret_dict = {}


        self.afd_layers = nn.ModuleList()
        for idx in range(afd_num_layers):
            layer = SEDLayer(
                afd_dim, afd_down_kernel_size, afd_down_stride, afd_num_SBB,
                indice_key=f'afdlayer{idx}', xy_only=True)
            self.afd_layers.append(layer)

        self.num_point_features = afd_dim
        self.init_weights()
        self.build_losses()
        self.model_cfg = model_cfg

    def build_losses(self):
        self.add_module('hm_loss_func', FocalLossCenterNet())

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, (spconv.SubMConv2d, spconv.SubMConv3d)):
                nn.init.kaiming_normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if self.adaptive_feature_diffusion:
            self.cls_conv[-1].bias.data.fill_(-2.19)

    def assign_target(self, gt_boxes, feature_map_size=None, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)
            range_image_polar: (B, 3, H, W)
            feature_map_size: (2) [H, W]
            spatial_cartesian: (B, 4, H, W)
        Returns:
            
        """

        feature_map_size = feature_map_size[::-1]  # [H, W] ==> [x, y]
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        # feature_map_size = self.grid_size[:2] // target_assigner_cfg.FEATURE_MAP_STRIDE

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'inds': [],
            'masks': [],
            'heatmap_masks': [],
            'target_boxes_src': [], 
        }
        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head): # match category head by head
            heatmap_list, inds_list, masks_list, target_boxes_src_list = [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names: 
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0: 
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap, _, inds, mask, ret_boxes_src = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.cpu(),
                    feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))
                target_boxes_src_list.append(ret_boxes_src.to(gt_boxes_single_head.device))

            ret_dict['heatmaps'].append(torch.stack(heatmap_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
            ret_dict['target_boxes_src'].append(torch.stack(target_boxes_src_list, dim=0))
        return ret_dict

    def assign_target_of_single_head(self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500, gaussian_overlap=0.1, min_radius=2):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()
        ret_boxes_src = gt_boxes.new_zeros(num_max_objs, gt_boxes.shape[-1])
        ret_boxes_src[:gt_boxes.shape[0]] = gt_boxes

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue 

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())

            inds[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k] = 1

            ret_boxes[k, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]

        return heatmap, ret_boxes, inds, mask, ret_boxes_src

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['hm_dicts']
        target_dict = self.forward_ret_dict['dense_target_dict']

        tb_dict = {}
        loss_dense = 0
        
        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = torch.clamp(pred_dict['hm'].sigmoid(), min=1e-4, max=1 - 1e-4)
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dict['heatmaps'][idx])
            hm_loss *= self.model_cfg.DENSE_LOSS_WEIGHTS
            loss_dense += hm_loss
            tb_dict['dense_loss_head_%d' % idx] = hm_loss.item()

        # sparse get_loss
        spatial_indices = self.forward_ret_dict['spatial_indices'] # [11220, 3]
        batch_size = self.forward_ret_dict['batch_size'] # int 2
        batch_index = spatial_indices[:, 0]

        inside_box_pred = self.forward_ret_dict['inside_box_pred'] # foreground score [3, 11220]
        inside_box_target = self.forward_ret_dict['inside_box_target'] # binary 1 or 0 [3, 11220]
        inside_box_pred = torch.cat([inside_box_pred[:, batch_index == bidx] for bidx in range(batch_size)], dim=1)
        inside_box_pred = torch.clamp(inside_box_pred.sigmoid(), min=1e-4, max=1 - 1e-4)

        for gidx in range(len(self.group_class_names)):
            group_cls_loss = focal_loss_sparse(inside_box_pred[gidx], inside_box_target[gidx].float())
            loss_dense += group_cls_loss
            fg_mask = inside_box_target[gidx] > 0
            pred_mask = inside_box_pred[gidx][fg_mask] > self.fg_thr
            tb_dict[f'afd_recall_{gidx}'] = (pred_mask.sum() / fg_mask.sum().clamp(min=1.0)).item()
            tb_dict[f'afd_cls_loss_{gidx}'] = group_cls_loss.item()

        return loss_dense, tb_dict

    def to_bev(self, x):
        x = self.dense_encoder(x)

        features = x.features
        indices = x.indices[:, [0, 2, 3]]
        spatial_shape = x.spatial_shape[1:]

        indices_unique, _inv = torch.unique(indices, dim=0, return_inverse=True)
        features_unique = features.new_zeros((indices_unique.shape[0], features.shape[1]))
        features_unique.index_add_(0, _inv, features)

        x = spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=spatial_shape,
            batch_size=x.batch_size
        )
        return x

    def feature_diffusion(self, x, batch_dict):
        if not self.adaptive_feature_diffusion:
            return x

        detached_x = x
        if self.detach_feature:
            detached_x = spconv.SparseConvTensor(
                features=x.features.detach(),
                indices=x.indices,
                spatial_shape=x.spatial_shape,
                batch_size=x.batch_size
            )

        inside_box_pred = self.cls_conv(detached_x).features.permute(1, 0)

        if self.training:
            inside_box_target = self.assign_target_sparse(x.indices, batch_dict['gt_boxes'])
            self.forward_ret_dict['batch_size'] = x.batch_size
            self.forward_ret_dict['spatial_indices'] = x.indices
            self.forward_ret_dict['inside_box_pred'] = inside_box_pred # predict value foreground score
            self.forward_ret_dict['inside_box_target'] = inside_box_target

        group_inside_mask = inside_box_pred.sigmoid() > self.fg_thr
        bg_mask = ~group_inside_mask.max(dim=0, keepdim=True)[0]
        group_inside_mask = torch.cat([group_inside_mask, bg_mask], dim=0)

        one_mask = x.features.new_zeros((x.batch_size, 1, x.spatial_shape[0], x.spatial_shape[1]))
        for gidx, inside_mask in enumerate(group_inside_mask):
            selected_indices = x.indices[inside_mask]
            single_one_mask = spconv.SparseConvTensor(
                features=x.features.new_ones(selected_indices.shape[0], 1),
                indices=selected_indices,
                spatial_shape=x.spatial_shape,
                batch_size=x.batch_size
            ).dense()
            pooling_size = self.group_pooling_kernel_size[gidx]
            single_one_mask = F.max_pool2d(single_one_mask, kernel_size=pooling_size, stride=1, padding=pooling_size // 2)
            one_mask = torch.maximum(one_mask, single_one_mask)

        zero_indices = (one_mask[:, 0] > 0).nonzero().int()
        zero_features = x.features.new_zeros((len(zero_indices), x.features.shape[1]))

        cat_indices = torch.cat([x.indices, zero_indices], dim=0)
        cat_features = torch.cat([x.features, zero_features], dim=0)
        indices_unique, _inv = torch.unique(cat_indices, dim=0, return_inverse=True)
        features_unique = x.features.new_zeros((indices_unique.shape[0], x.features.shape[1]))
        features_unique.index_add_(0, _inv, cat_features)

        x = spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=x.spatial_shape,
            batch_size=x.batch_size
        )
        return x

    def dense_knowledge_injection(self, x, batch_dict):
        """
        Args:
            x: sprase_tensor (B, C, 180, 180)
            batch_dict:
                dense_bev_feature: (B, C, 180, 180)
        """
        detached_x = x
        if self.detach_feature:
            detached_x = spconv.SparseConvTensor(
                features = x.features.detach(),
                indices = x.indices,
                spatial_shape = x.spatial_shape,
                batch_size = x.batch_size
            )
        x_dense = detached_x.dense() # [1, 128, 180, 180]

        # 1. 用heatmap完成中心点回归
        B, C, _, _ = x_dense.size()

        k = self.model_cfg.K
        hm_dicts = []
        for head in self.cls_heads_list:
            hm_dicts.append(head(x_dense))
        # assign
        if self.training:
            dense_target_dict = self.assign_target(
                batch_dict['gt_boxes'], feature_map_size=x_dense.size()[2:],
                feature_map_stride=batch_dict.get('spatial_features_2d_strides', None)
            )
            self.forward_ret_dict['dense_target_dict'] = dense_target_dict

        self.forward_ret_dict['hm_dicts'] = hm_dicts

        # 2. 确定中心点位置
        centerpoints = []
        for idx, hm_dict in enumerate(hm_dicts):
            heatmap = hm_dict['hm'].sigmoid()
            scores, inds, class_ids, ys, xs = centernet_utils._topk(heatmap, K=k)
            centerpoint = torch.stack([ys, xs], dim=2) # [b, c, ys, xs]
            centerpoints.append(centerpoint)
        centerpoints = torch.cat(centerpoints, dim=1) # [b, num_cp, [ys,xs]]

        # # 3. 中心点扩散/拷贝?
        sparse_indices = []
        num_points = centerpoints.size(1)
        cp_feature =  x_dense.new_zeros(B, C, num_points)
        for bs_index in range(B):
            cur_bev =  x_dense[bs_index]
            cur_cp = centerpoints[bs_index].long()
            cp_feature[bs_index] = cur_bev[:, cur_cp[:,0], cur_cp[:,1]]
            sparse_indices.append(torch.cat([(x_dense.new_ones(num_points) * bs_index)[:, None], cur_cp], dim=1))

        sparse_indices = torch.cat(sparse_indices).int()
        x = x.replace_feature(torch.cat([x.features, cp_feature.contiguous().view(-1, C)])) # TODO check: cp_feature.permute(0, 2, 1).contiguous().view(-1, C) 
        x.indices = torch.cat([x.indices, sparse_indices])

        return x

    def forward(self, batch_dict):
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.stem(x)
        for layer in self.sed_layers:
            x = layer(x)

        x = self.to_bev(x)
        x = self.feature_diffusion_dense_guided(x, batch_dict)
        x = self.feature_diffusion(x, batch_dict)
        for layer in self.afd_layers:
            x = layer(x)

        batch_dict.update({'spatial_features_2d': x})
        return batch_dict

    def assign_target_sparse(self, batch_spatial_indices, batch_gt_boxes):
        all_names = np.array(['bg', *self.class_names])
        inside_box_target = batch_spatial_indices.new_zeros((len(self.group_class_names), batch_spatial_indices.shape[0]))

        for gidx, names in enumerate(self.group_class_names):
            batch_inside_box_mask = []
            for bidx in range(len(batch_gt_boxes)):
                spatial_indices = batch_spatial_indices[batch_spatial_indices[:, 0] == bidx][:, [2, 1]]
                points = spatial_indices.clone() + 0.5
                points[:, 0] = points[:, 0] * self.featmap_stride * self.voxel_size[0] + self.point_cloud_range[0]
                points[:, 1] = points[:, 1] * self.featmap_stride * self.voxel_size[1] + self.point_cloud_range[1]
                points = torch.cat([points, points.new_zeros((points.shape[0], 1))], dim=-1)

                gt_boxes = batch_gt_boxes[bidx].clone()
                gt_boxes = gt_boxes[(gt_boxes[:, 3] > 0) & (gt_boxes[:, 4] > 0)]
                gt_class_names = all_names[gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []
                for _, name in enumerate(gt_class_names):
                    if name in names:
                        gt_boxes_single_head.append(gt_boxes[_])

                inside_box_mask = points.new_zeros((points.shape[0]))
                if len(gt_boxes_single_head) > 0:
                    boxes = torch.stack(gt_boxes_single_head)[:, :7]
                    boxes[:, 2] = 0
                    inside_box_mask[roiaware_pool3d_utils.points_in_boxes_gpu(points[None], boxes[None])[0] > -1] = 1
                batch_inside_box_mask.append(inside_box_mask)
            inside_box_target[gidx] = torch.cat(batch_inside_box_mask)
        return inside_box_target
