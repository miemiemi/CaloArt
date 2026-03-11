import torch
from torch.optim import AdamW

class BiasedAdamW(AdamW):
    def __init__(self, model, lr=0.0005, **kwargs):
        weight_decay = kwargs.get("weight_decay", 1e-5)
        kwargs.pop("weight_decay", None)
        # Split parameters into decay and no_decay groups
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Match the usual "no decay on bias / norm / other 1D scales" rule.
            if len(param.shape) == 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Initialize the AdamW optimizer with these grouped parameters
        super().__init__(param_groups, lr=lr, **kwargs)
