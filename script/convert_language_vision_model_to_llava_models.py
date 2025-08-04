import argparse
import json
import os
from collections import OrderedDict
from typing import Any, Dict, Optional
import torch
from transformers.modeling_utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    shard_checkpoint,
)
from transformers import AutoTokenizer

from safetensors.torch import save_file,safe_open
from tqdm import tqdm
from pathlib import Path

def load_input_state_dict(args):
    souce_state_dict: Dict[str, torch.Tensor] = OrderedDict()

    for filepath in tqdm(os.listdir(args.language_model_path), desc="Load weights"):
        if os.path.isfile(os.path.join(args.language_model_path, filepath)) and filepath.endswith(".bin"):
            shard_weight = torch.load(os.path.join(args.language_model_path, filepath), map_location="cpu")
            souce_state_dict.update(shard_weight)
        elif os.path.isfile(os.path.join(args.language_model_path, filepath)) and filepath.endswith(".safetensors"):
            with safe_open(os.path.join(args.language_model_path, filepath), framework="pt") as f:
                for k in f.keys():
                    souce_state_dict[k] = f.get_tensor(k)
    return souce_state_dict

def convert_to_BunnyPhi3MMForCausalLM(args):
    from bunny.model.language_model.bunny_phi3_m import BunnyPhi3MMConfig 
    config = BunnyPhi3MMConfig.from_pretrained(args.language_model_path)
  

    config.architectures[0] = "BunnyPhi3MMForCausalLM"
    source_state_dict = load_input_state_dict(args)

    target_state_dict: Dict[str, torch.Tensor] = OrderedDict()

    torch_dtype = None
    for key, value in tqdm(source_state_dict.items(), desc="Convert format"):
        if torch_dtype is None:
            torch_dtype = value.dtype
        # if "self_attn" in key and "rotary_emb" not in key:
        #     target_state_dict[key]= value
        #     target_state_dict[key.replace("_proj","_vision_proj")] = value.clone()
        if "model.layers" in key and "vision_tower" not in key:
            ls  = key.split('.')
            layer_idx = int(ls[2])
            if "mlp" in key and (layer_idx%2==0):
                target_state_dict[key.replace("mlp", "mlp.vision_expert")] = value.clone()
                target_state_dict[key.replace("mlp", "mlp.language_expert")] = value.clone()
            else:
                target_state_dict[key]=value
        else:
            target_state_dict[key]=value

    return target_state_dict,config,AutoTokenizer.from_pretrained(args.language_model_path)


def convert_to_BunnyPhi3ForCausalLM(args):
    from bunny.model.language_model.bunny_phi3 import BunnyPhi3Config 
    config = BunnyPhi3Config.from_pretrained(args.language_model_path)
  

    config.architectures[0] = "BunnyPhi3ForCausalLM"
    source_state_dict = load_input_state_dict(args)

    target_state_dict: Dict[str, torch.Tensor] = OrderedDict()

    torch_dtype = None
    for key, value in tqdm(source_state_dict.items(), desc="Convert format"):
        if torch_dtype is None:
            torch_dtype = value.dtype
        # if "self_attn" in key and "rotary_emb" not in key:
        #     target_state_dict[key]= value
        #     target_state_dict[key.replace("_proj","_vision_proj")] = value.clone()
        if "model.layers" in key and "vision_tower" not in key:
            ls  = key.split('.')
            layer_idx = int(ls[2])
            if "mlp" in key and (layer_idx%2!=0) and "mlp.experts.0" not in key and "mlp.experts.2" not in key and "mlp.experts.3" not in key :
                # target_state_dict[key.replace("mlp.vision_expert", "mlp")] = value.clone()
                target_state_dict[key.replace("mlp.experts.1", "mlp")] = value.clone()
            # elif "vision_expert" not in key:
            elif "mlp.experts.0" not in key and "mlp.experts.2" not in key and "mlp.experts.3" not in key:
                target_state_dict[key]=value
        else:
            target_state_dict[key]=value

    return target_state_dict,config,AutoTokenizer.from_pretrained(args.language_model_path)



def convert(args,shard_size: Optional[str] = "2GB", save_safetensors: Optional[bool] = True):


    if args.llava_model_architecture == "m-phi-3":

        model_state_dict,config,tokenizer = convert_to_BunnyPhi3MMForCausalLM(args)
    
    if args.llava_model_architecture == "phi-3":

        model_state_dict,config,tokenizer = convert_to_BunnyPhi3ForCausalLM(args)

    weights_name = SAFE_WEIGHTS_NAME if save_safetensors else WEIGHTS_NAME

    shards, index = shard_checkpoint(model_state_dict, max_shard_size=shard_size, weights_name=weights_name)
    for shard_file, shard in tqdm(shards.items(), desc="Save weights"):
        if model_state_dict:
            save_file(shard, os.path.join(args.save_model_path, shard_file), metadata={"format": "pt"})
        else:
            torch.save(shard, os.path.join(args.save_model_path, shard_file))

    if index is None:
        print("Model weights saved in {}".format(os.path.join(args.save_model_path, WEIGHTS_NAME)))
    else:
        index_name = SAFE_WEIGHTS_INDEX_NAME if save_safetensors else WEIGHTS_INDEX_NAME
        with open(os.path.join(args.save_model_path, index_name), "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, sort_keys=True)
        print("Model weights saved in {}".format(args.save_model_path))

    tokenizer.save_pretrained(args.save_model_path)
    config.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language-model-path", type=str, required=True)
    parser.add_argument("--llava-model-architecture", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)

    args = parser.parse_args()
    Path(args.save_model_path).mkdir(parents=True,exist_ok=True)
    convert(args)