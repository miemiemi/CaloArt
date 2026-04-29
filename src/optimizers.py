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


class LayerwiseAdamW(AdamW):
    def __init__(self, model, lr=0.0005, param_group_specs=None, **kwargs):
        weight_decay = kwargs.get("weight_decay", 1e-5)
        kwargs.pop("weight_decay", None)
        param_group_specs = list(param_group_specs or [])

        def _normalize_prefixes(spec):
            prefixes = spec.get("prefixes")
            if prefixes is None:
                prefix = spec.get("prefix")
                prefixes = [] if prefix is None else [prefix]
            return [p for p in prefixes if p]

        normalized_specs = []
        for index, spec in enumerate(param_group_specs):
            prefixes = _normalize_prefixes(spec)
            normalized_specs.append(
                {
                    "name": spec.get("name", f"group_{index}"),
                    "prefixes": prefixes,
                    "lr": spec.get("lr"),
                    "lr_scale": spec.get("lr_scale", 1.0),
                    "weight_decay": spec.get("weight_decay", weight_decay),
                }
            )

        grouped = {}

        def _match_spec(param_name):
            best_spec = None
            best_prefix_len = -1
            for spec in normalized_specs:
                for prefix in spec["prefixes"]:
                    if param_name.startswith(prefix) and len(prefix) > best_prefix_len:
                        best_spec = spec
                        best_prefix_len = len(prefix)
            return best_spec

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            spec = _match_spec(name)
            group_name = "default" if spec is None else spec["name"]
            group_lr = lr if spec is None else spec["lr"]
            if group_lr is None:
                lr_scale = 1.0 if spec is None else spec["lr_scale"]
                group_lr = lr * lr_scale
            group_weight_decay = weight_decay if spec is None else spec["weight_decay"]

            decay_tag = "no_decay" if len(param.shape) == 1 or name.endswith(".bias") else "decay"
            group_key = (group_name, decay_tag, float(group_lr), float(group_weight_decay))
            if group_key not in grouped:
                grouped[group_key] = []
            grouped[group_key].append(param)

        param_groups = []
        for (group_name, decay_tag, group_lr, group_weight_decay), params in grouped.items():
            param_groups.append(
                {
                    "params": params,
                    "lr": group_lr,
                    "weight_decay": 0.0 if decay_tag == "no_decay" else group_weight_decay,
                    "group_name": group_name,
                }
            )

        super().__init__(param_groups, lr=lr, **kwargs)
