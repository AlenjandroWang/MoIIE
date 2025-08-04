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




def convert_to_BunnyPhi3MoEForCausalLM(args):
    from bunny.model.language_model.bunny_phi3_moe import  BunnyPhi3MoEConfig
    config = BunnyPhi3MoEConfig.from_pretrained(args.language_model_path)

    config.architectures[0] = "BunnyPhi3MoEForCausalLM"
    config.num_experts = args.num_local_experts
    config.num_experts_per_tok = args.num_experts_per_tok
    config.router_aux_loss_coef  = args.router_aux_loss_coef
    config.output_router_logits = args.output_router_logits 
    source_state_dict = load_input_state_dict(args)

    target_state_dict: Dict[str, torch.Tensor] = OrderedDict()

    torch_dtype = None
    for key, value in tqdm(source_state_dict.items(), desc="Convert format"):
        if torch_dtype is None:
            torch_dtype = value.dtype
        if "model.layers" in key and "vision_tower" not in key:
            ls  = key.split('.')
            layer_idx = int(ls[2])
            if "mlp" in key and (layer_idx%2!=0):
                    for i in range(args.num_local_experts):
                        target_state_dict[key.replace("mlp", "mlp.experts.{}".format(str(i)))] = value.clone()
            else:
                target_state_dict[key]=value
        else:
                target_state_dict[key]=value

    return target_state_dict,config,AutoTokenizer.from_pretrained(args.language_model_path)


def convert_to_BunnyMMSPhi3MoEForCausalLM(args):
    from bunny.model.language_model.bunny_phi3_moe_mm_s import  BunnyMMSPhi3MoEConfig
    config = BunnyMMSPhi3MoEConfig.from_pretrained(args.language_model_path)

    config.architectures[0] = "BunnyMMSPhi3MoEForCausalLM"
    config.num_experts = args.num_local_experts
    config.num_experts_per_tok = args.num_experts_per_tok
    config.vis_router_aux_loss_coef  = args.vis_router_aux_loss_coef
    config.lan_router_aux_loss_coef  = args.lan_router_aux_loss_coef
    config.output_vis_router_logits = args.output_vis_router_logits
    config.output_lan_router_logits = args.output_lan_router_logits 
    source_state_dict = load_input_state_dict(args)

    target_state_dict: Dict[str, torch.Tensor] = OrderedDict()

    torch_dtype = None
    for key, value in tqdm(source_state_dict.items(), desc="Convert format"):
        if torch_dtype is None:
            torch_dtype = value.dtype
        if "model.layers" in key and "vision_tower" not in key:
            ls  = key.split('.')
            layer_idx = int(ls[2])
            if "mlp" in key and (layer_idx%2!=0):
            # if "mlp" in key:
                for i in range(args.num_local_experts):
                    target_state_dict[key.replace("mlp", "mlp.experts.{}".format(str(i)))] = value.clone()
            else:
                target_state_dict[key]=value
        else:
                target_state_dict[key]=value

    return target_state_dict,config,AutoTokenizer.from_pretrained(args.language_model_path)

def convert_to_BunnyMMSLlamaMoEForCausalLM(args):
    from bunny.model.language_model.bunny_llama_moe_mm_s import  BunnyMMSLlamaMoEConfig
    config = BunnyMMSLlamaMoEConfig.from_pretrained(args.language_model_path)

    config.architectures[0] = "BunnyMMSLlamaMoEForCausalLM"
    config.num_experts = args.num_local_experts
    config.num_experts_per_tok = args.num_experts_per_tok
    config.vis_router_aux_loss_coef  = args.vis_router_aux_loss_coef
    config.lan_router_aux_loss_coef  = args.lan_router_aux_loss_coef
    config.output_vis_router_logits = args.output_vis_router_logits
    config.output_lan_router_logits = args.output_lan_router_logits 
    source_state_dict = load_input_state_dict(args)

    target_state_dict: Dict[str, torch.Tensor] = OrderedDict()

    torch_dtype = None
    for key, value in tqdm(source_state_dict.items(), desc="Convert format"):
        if torch_dtype is None:
            torch_dtype = value.dtype
        if "model.layers" in key and "vision_tower" not in key:
            ls  = key.split('.')
            layer_idx = int(ls[2])
            if "mlp" in key and (layer_idx%2!=0):
            # if "mlp" in key:
                for i in range(args.num_local_experts):
                    target_state_dict[key.replace("mlp", "mlp.experts.{}".format(str(i)))] = value.clone()
            else:
                target_state_dict[key]=value
        else:
                target_state_dict[key]=value

    return target_state_dict,config,AutoTokenizer.from_pretrained(args.language_model_path)




def convert_to_BunnyMMPhi3MoEForCausalLM(args):
    from bunny.model.language_model.bunny_phi3_moe_mm import  BunnyMMPhi3MoEConfig
    config = BunnyMMPhi3MoEConfig.from_pretrained(args.language_model_path)

    config.architectures[0] = "BunnyMMPhi3MoEForCausalLM"
    config.num_vis_experts = args.num_vis_experts
    config.num_lan_experts = args.num_lan_experts
    config.num_experts_per_tok = args.num_experts_per_tok
    config.vis_router_aux_loss_coef  = args.vis_router_aux_loss_coef
    config.lan_router_aux_loss_coef  = args.lan_router_aux_loss_coef
    config.output_vis_router_logits = args.output_vis_router_logits
    config.output_lan_router_logits = args.output_lan_router_logits 
    source_state_dict = load_input_state_dict(args)

    target_state_dict: Dict[str, torch.Tensor] = OrderedDict()

    torch_dtype = None
    for key, value in tqdm(source_state_dict.items(), desc="Convert format"):
        if torch_dtype is None:
            torch_dtype = value.dtype
        if "model.layers" in key and "vision_tower" not in key:
            ls  = key.split('.')
            layer_idx = int(ls[2])
            if "mlp" in key and (layer_idx%2!=0):
                    for i in range(args.num_vis_experts):
                        target_state_dict[key.replace("mlp", "mlp.vis_experts.{}".format(str(i)))] = value.clone()
                    for i in range(args.num_lan_experts):
                        target_state_dict[key.replace("mlp", "mlp.lan_experts.{}".format(str(i)))] = value.clone()
            else:
                target_state_dict[key]=value
        else:
                target_state_dict[key]=value

    return target_state_dict,config,AutoTokenizer.from_pretrained(args.language_model_path)





def convert(args,shard_size: Optional[str] = "2GB", save_safetensors: Optional[bool] = True):

 
    
    if args.moe_architecture == "bunny-phi3-moe":

        model_state_dict,config,tokenizer = convert_to_BunnyPhi3MoEForCausalLM(args)

    if args.moe_architecture == "bunny-mm-phi3-moe":

        model_state_dict,config,tokenizer = convert_to_BunnyMMPhi3MoEForCausalLM(args)
    
    if args.moe_architecture == "bunny-mm-phi3-moe-s":

        model_state_dict,config,tokenizer = convert_to_BunnyMMSPhi3MoEForCausalLM(args)
    
    if args.moe_architecture == "bunny-mm-llama3-moe-s":

        model_state_dict,config,tokenizer = convert_to_BunnyMMSLlamaMoEForCausalLM(args)
    

    


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
    parser.add_argument("--num_local_experts", type=int, required=False)
    parser.add_argument("--num_vis_experts", type=int, required=False)
    parser.add_argument("--num_lan_experts", type=int, required=False)
    parser.add_argument("--num_experts_per_tok", type=int, required=False)
    parser.add_argument("--router_aux_loss_coef", type=float, required=False)
    parser.add_argument("--vis_router_aux_loss_coef", type=float, required=False)
    parser.add_argument("--lan_router_aux_loss_coef", type=float, required=False)
    parser.add_argument("--output_vis_router_logits", type=bool, required=False)
    parser.add_argument("--output_lan_router_logits", type=bool, required=False)
    parser.add_argument("--output_router_logits", type=bool, required=False)
    parser.add_argument("--moe_architecture", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)

    args = parser.parse_args()
    Path(args.save_model_path).mkdir(parents=True,exist_ok=True)
    convert(args)