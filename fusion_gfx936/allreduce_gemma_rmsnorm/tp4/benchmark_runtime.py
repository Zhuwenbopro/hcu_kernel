"""Common checkpoint and device checks for the HCU benchmark."""
import json
import os
from pathlib import Path

import torch.distributed as dist
from safetensors import safe_open


def load_norm_weight(device):
    model = Path(os.environ["MODEL_PATH"])
    index = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))
    key = "model.language_model.layers.0.post_attention_layernorm.weight"
    with safe_open(str(model / index["weight_map"][key]), framework="pt") as handle:
        return handle.get_tensor(key).to(device)


def check_device_mapping(local_bdf, world_size):
    if dist.get_world_size() != world_size:
        raise RuntimeError(f"expected {world_size} ranks")
    bdfs = [None] * world_size
    dist.all_gather_object(bdfs, local_bdf)
    if len(set(bdfs)) != world_size:
        raise RuntimeError(f"each rank must own a distinct local HCU; mapped devices: {bdfs}")
