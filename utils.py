from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer
import json
import glob
import torch
from safetensors import safe_open
from typing import Tuple
import os

def load_hf_model(model_path: str, device: str) -> PaliGemmaForConditionalGeneration:
    print(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    tensors = {}

    for safetensor_file in safetensor_files:
        with safe_open(safetensor_file, framework="pt", device=device) as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    with open(os.path.join(model_path, "config.json"), "r") as f:
        config_dict = json.load(f)
        config = PaliGemmaConfig(**config_dict)

    model = PaliGemmaForConditionalGeneration(config).to(device)
    #checkpoint = torch.load('path/to/checkpoint.pt', map_location='cpu')
    #print("Checkpoint keys:", list(checkpoint.keys()))
    #print("Model state_dict keys:", list(model.state_dict().keys()))
    model.load_state_dict(tensors, strict=False)
    model.tie_weights()

    return (model, tokenizer)
