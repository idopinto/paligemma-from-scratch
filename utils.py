from paligemma.config import PaliGemmaConfig
from transformers import AutoTokenizer
from paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
import json
import glob
from safetensors import safe_open
import os
from tqdm import tqdm
import time
import torch


def load_hf_model(
    model_path: str, device: str
) -> tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    # Load the tokenizer
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"
    # print(f"Tokenizer: {tokenizer}")
    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    print(f"Found {len(safetensors_files)} safetensors files")
    # Load weights on CPU first, cast to bfloat16 immediately to halve memory.
    # Moving a pre-loaded float32 model to MPS can silently leave some tensors
    # on CPU; building on CPU then doing a single .to(device) is more reliable.
    tensors = {}
    for safetensors_file in tqdm(
        safetensors_files, desc="Loading safetensors files to dictionary"
    ):
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key).to(torch.bfloat16)

    # Load the model's config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # Build on CPU in bfloat16, load weights, then move to target device in one shot.
    model = PaliGemmaForConditionalGeneration(config).to(torch.bfloat16)
    start_time = time.time()
    missing, unexpected = model.load_state_dict(tensors, strict=False)
    end_time = time.time()
    print(f"Time taken to load state dict: {end_time - start_time:.2f}s")
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model = model.to(device)
    model.tie_weights()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    return (model, tokenizer)
