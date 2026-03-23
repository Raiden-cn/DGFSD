from .detector3d_template import Detector3DTemplate


class TransFusionDGFSD(Detector3DTemplate):

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, batch_dict):
        disp_dict = {}

        loss_trans, tb_dict = batch_dict['loss'], batch_dict['tb_dict']
        tb_dict = {
            'loss_trans': loss_trans.item(),
            **tb_dict
        }
        loss = loss_trans
        if hasattr(self.backbone_3d, 'get_loss'):
            loss_dense_head, tb_dict_dense_head = self.backbone_3d.get_loss()
            loss += loss_dense_head
            tb_dict.update(tb_dict_dense_head)
            disp_dict.update({'dense_head_0' : tb_dict['dense_loss_head_0'],})
        if hasattr(self, 'point_head'): # for dfw
            if hasattr(self.point_head, 'get_loss'):
                loss_dfw, tb_dict_point = self.point_head.get_loss()
                loss += loss_dfw
                tb_dict.update(tb_dict_point)
                disp_dict.update({'point_loss_cls' : tb_dict['point_loss_cls'],})

        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict


class CenterPointDGFSD(Detector3DTemplate):

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn

        if hasattr(self.backbone_3d, 'get_loss'):
            loss_dense_head, tb_dict_dense_head = self.backbone_3d.get_loss()
            loss += loss_dense_head
            tb_dict.update(tb_dict_dense_head)
            disp_dict.update({"dense_head_0": tb_dict['dense_loss_head_0']},)
        if hasattr(self, 'point_head'):
            if hasattr(self.point_head, 'get_loss'):
                loss_dfw, tb_dict_point = self.point_head.get_loss()
                loss += loss_dfw
                tb_dict.update(tb_dict_point)
                disp_dict.update({'point_loss_cls' : tb_dict['point_loss_cls'],})

        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
