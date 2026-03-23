import torch
from ...utils import box_utils
from .point_head_template import PointHeadTemplate
from ...utils.spconv_utils import replace_feature, spconv

class Dense_Feature_Weighting(PointHeadTemplate):
    def __init__(self, num_class, input_channels, model_cfg, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )
        self.featmap_stride = model_cfg.get("FEATURE_MAP_STRIDE", None)
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

    def assign_targets(self, input_dict, point_coords):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)
        
        batch_size = gt_boxes.shape[0]
        gt_boxes[:,:,2] = 0 # set Z-axis to 0
        # for nuscenes
        gt_boxes = gt_boxes[:,:,[0, 1, 2, 3, 4, 5, 6, 9]] # [x,y,z,l,w,h,r,v1,v2,cls]-->[x,y,z,l,w,h,r,cls]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False
        )

        return targets_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss()

        point_loss = point_loss_cls
        tb_dict.update(tb_dict_1)
        return point_loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        sparse_bev = batch_dict['spatial_features_2d']
        point_features = sparse_bev.features.detach() # more stable for training
        bev_coords = sparse_bev.indices
        point_cls_preds = self.cls_layers(point_features)  # (total_points, num_class)

        ret_dict = {
            'point_cls_preds': point_cls_preds,
        }
        point_cls_scores = torch.sigmoid(point_cls_preds)
        point_cls_scores, _ = point_cls_scores.max(dim=-1) # incase of multi category
        
        weighted_sparse_bev = replace_feature(sparse_bev, sparse_bev.features * point_cls_scores.view(-1, 1))

        batch_dict['spatial_features_2d'] = weighted_sparse_bev

        if self.training:
            point_coords =  self.to_point(bev_coords)
            targets_dict = self.assign_targets(batch_dict, point_coords)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
        self.forward_ret_dict = ret_dict

        return batch_dict

    def to_point(self, bev_coords):
        """
        input:
            bev_coords: (N, 3) [bs_idx, x, y]
        return:
            point_coords: (N, 4) [bs_idx, x, y, 0]
        """
        batch_indices = bev_coords[:,[0]]
        bev_coord = bev_coords[:, [2, 1]]
        point_coord = bev_coord.clone() + 0.5
        point_coord[:, 0] = bev_coord[:, 0] * self.featmap_stride * self.voxel_size[0] + self.point_cloud_range[0]
        point_coord[:, 1] = bev_coord[:, 1] * self.featmap_stride * self.voxel_size[1] + self.point_cloud_range[1]
        point_coords = torch.cat([batch_indices, point_coord, point_coord.new_zeros((point_coord.shape[0], 1))], dim=-1)
        
        return point_coords


