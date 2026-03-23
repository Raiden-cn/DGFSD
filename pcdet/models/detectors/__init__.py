from .detector3d_template import Detector3DTemplate
from .dgfsd import TransFusionDGFSD, CenterPointDGFSD
__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'TransFusionDGFSD': TransFusionDGFSD, # for nuScenes
    'CenterPointDGFSD': CenterPointDGFSD # for argo2

}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
