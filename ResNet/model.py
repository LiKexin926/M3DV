import torch
from torch import nn

import resnet
def generate_model(shortcut,sample_size,sample_duration,models,model_depth=50,n_classes=2,pretrain_path=False,ft_begin_index=0):
    if models == 'resnet':
        from resnet import get_fine_tuning_parameters
        if model_depth == 10:
            model = resnet.resnet10(
                num_classes=n_classes,
                shortcut_type=shortcut,
                sample_size=sample_size,
                sample_duration=sample_duration)
        elif model_depth == 18:
            model = resnet.resnet18(
                num_classes=n_classes,
                shortcut_type=shortcut,
                sample_size=sample_size,
                sample_duration=sample_duration)
        elif model_depth == 34:
            model = resnet.resnet34(
                num_classes=n_classes,
                shortcut_type=shortcut,
                sample_size=sample_size,
                sample_duration=sample_duration)
        elif model_depth == 50:
            model = resnet.resnet50(
                num_classes=n_classes,
                shortcut_type=shortcut,
                sample_size=sample_size,
                sample_duration=sample_duration)
        elif model_depth == 101:
            model = resnet.resnet101(
                num_classes=n_classes,
                shortcut_type=shortcut,
                sample_size=sample_size,
                sample_duration=sample_duration)
        elif model_depth == 152:
            model = resnet.resnet152(
                num_classes=n_classes,
                shortcut_type=shortcut,
                sample_size=sample_size,
                sample_duration=sample_duration)
        elif model_depth == 200:
            model = resnet.resnet200(
                num_classes=n_classes,
                shortcut_type=shortcut,
                sample_size=sample_size,
                sample_duration=sample_duration)

    parameters = get_fine_tuning_parameters(model, ft_begin_index)
    
    return model, parameters
