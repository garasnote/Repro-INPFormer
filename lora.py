"""
LoRA (Low-Rank Adaptation) for INP-Former.

Freezes pretrained weights and injects trainable low-rank adapters
into decoder Linear layers. For parameter-efficient finetuning
when transferring across datasets (e.g. Real-IAD → MVTec-AD).
"""
import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear with a low-rank adapter."""

    def __init__(self, original: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features
        device = original.weight.device

        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x):
        base = self.original(x)
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base + lora

    def extra_repr(self):
        return f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.3f}"


def apply_lora(model, target_modules=None, rank=4, alpha=1.0):
    """
    Inject LoRA adapters into Linear layers of specified modules.

    Args:
        model: INP_Former model
        target_modules: list of module name prefixes to target.
                       Default: ['decoder', 'bottleneck'] (not encoder — frozen DINOv2)
        rank: LoRA rank (lower = fewer params, typically 2-8)
        alpha: scaling factor
    Returns:
        list of LoRA parameter names for the optimizer
    """
    if target_modules is None:
        target_modules = ['decoder', 'bottleneck']

    lora_params = []
    replaced = 0

    # Collect first, apply after — mutating during named_modules() causes infinite recursion
    replacements = []
    for name, module in model.named_modules():
        if not any(name.startswith(t) for t in target_modules):
            continue
        for attr_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                replacements.append((name, module, attr_name, child))

    for name, module, attr_name, child in replacements:
        lora_layer = LoRALinear(child, rank=rank, alpha=alpha)
        setattr(module, attr_name, lora_layer)
        lora_params.extend([
            (f"{name}.{attr_name}.lora_A", lora_layer.lora_A),
            (f"{name}.{attr_name}.lora_B", lora_layer.lora_B),
        ])
        replaced += 1

    print(f"LoRA: injected {replaced} adapters (rank={rank}, alpha={alpha})")
    total_lora = sum(p.numel() for _, p in lora_params)
    total_model = sum(p.numel() for p in model.parameters())
    print(f"LoRA params: {total_lora:,} / {total_model:,} total ({100*total_lora/total_model:.2f}%)")

    return lora_params


def get_lora_state_dict(model):
    """Extract only LoRA parameters for saving."""
    state = {}
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            state[name] = param.data
    return state


def merge_lora(model):
    """Merge LoRA weights into original Linear layers (for inference)."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                module.original.weight.add_(
                    (module.lora_B @ module.lora_A * module.scaling)
                )
    print("LoRA weights merged into base model.")
