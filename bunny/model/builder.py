import os
import warnings
import torch

from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig, logging,AutoModelForCausalLM

logging.set_verbosity_error()
warnings.filterwarnings('ignore')

from bunny.model import *


def load_pretrained_model(model_path, model_base, model_name, model_type, load_8bit=False, load_4bit=False,
                          device_map="auto", device="cuda", **kwargs):
    if model_type not in {'phi-1.5', 'phi-2', 'phi-3', 'stablelm-2', 'qwen1.5-1.8b', 'minicpm', 'llama3-8b','llava','llm','phi-3-moe','m-phi-3','mm-phi-3-moe','mms-phi-3-moe','mma-phi-3-moe','moe-llava-phi3','mms-llama-3-moe','llava_qwen','llava_mistral','llama-3-moe','m-llama-3'}:
        raise ValueError(f"Unknown Model Type {model_type}")

    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    # Load Bunny model
    if 'lora' in model_name.lower() and model_base is None:
        warnings.warn(
            'There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
    if 'lora' in model_name.lower() and model_base is not None:
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)

        print('Loading Bunny from base model...')
        if model_type == 'phi-1.5' or model_type == 'phi-2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyPhiForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                        config=lora_cfg_pretrained, **kwargs)
        elif model_type == 'phi-3':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyPhi3ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                         config=lora_cfg_pretrained, **kwargs)
        elif model_type == 'stablelm-2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True, trust_remote_code=True)
            model = BunnyStableLMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                             config=lora_cfg_pretrained, **kwargs)
        elif model_type == 'qwen1.5-1.8b':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyQwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained,
                                                          **kwargs)
        elif model_type =="llava_qwen":
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = LlavaQwenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained,
                                                          **kwargs)
        elif model_type == 'minicpm':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyMiniCPMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                            config=lora_cfg_pretrained,
                                                            **kwargs)
        elif model_type == 'llava_mistral':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = LlavaMistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                            config=lora_cfg_pretrained,
                                                            **kwargs)
        elif model_type == 'llama3-8b':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                          config=lora_cfg_pretrained,
                                                          **kwargs)
        elif model_type == 'mms-llama-3-moe':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyMMSLlamaMoEForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                          config=lora_cfg_pretrained,
                                                          **kwargs)
        elif model_type == 'llama-3-moe':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyLlamaMoEForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                          config=lora_cfg_pretrained,
                                                          **kwargs)
        elif model_type == 'm-llama-3':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyLlamaMMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                          config=lora_cfg_pretrained,
                                                          **kwargs)
        elif model_type == 'llava':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                          config=lora_cfg_pretrained,
                                                          **kwargs)
        elif model_type == 'phi-3-moe':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyPhi3MoEForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                          config=lora_cfg_pretrained,
                                                          **kwargs)
        elif model_type == 'mm-phi-3-moe':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyMMPhi3MoEForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                          config=lora_cfg_pretrained,
                                                          **kwargs)

        elif model_type == 'm-phi-3':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyPhi3MMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                          config=lora_cfg_pretrained,
                                                          **kwargs)
        elif model_type == 'mms-phi-3-moe':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyMMSPhi3MoEForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                          config=lora_cfg_pretrained,
                                                          **kwargs)
        elif model_type == 'mma-phi-3-moe':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyMMAPhi3MoEForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                          config=lora_cfg_pretrained,
                                                          **kwargs)
        elif model_type == 'moe-llava-phi3':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = EvalMoELLaVAPhiForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            import deepspeed
            deepspeed.init_distributed(dist_backend='nccl')
                    # Initialize the DeepSpeed-Inference engine
            ds_engine = deepspeed.init_inference(model,
                                                         # mp_size=2,
                                                         # dtype=torch.half,
                                                         checkpoint=None,
                                                         replace_with_kernel_inject=False)
            model = ds_engine.module
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(
                torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(
                torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional Bunny weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        else:
            # this is probably from HF Hub
            from huggingface_hub import hf_hub_download
            def load_from_hf(repo_id, filename, subfolder=None):
                cache_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    subfolder=subfolder)
                return torch.load(cache_file, map_location='cpu')

            non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')

        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in
                               non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in
                                   non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')
    elif model_base is not None and model_type!='llm':
        # this may be mm projector only
        print('Loading Bunny from base model...')

        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        if model_type == 'phi-1.5' or model_type == 'phi-2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyPhiForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                        config=cfg_pretrained, **kwargs)
        elif model_type == 'phi-3':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyPhi3ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                         config=cfg_pretrained, **kwargs)
        elif model_type == 'stablelm-2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True, trust_remote_code=True)
            model = BunnyStableLMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                             config=cfg_pretrained, **kwargs)
        elif model_type == 'qwen1.5-1.8b':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyQwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained,
                                                          **kwargs)
        elif model_type == 'llava_qwen':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = LlavaQwenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained,
                                                          **kwargs)
        elif model_type == 'llava_mistral':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = LlavaMistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained,
                                                          **kwargs)
        elif model_type == 'minicpm':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyMiniCPMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained,
                                                            **kwargs)
        elif model_type == 'llama3-8b':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained,
                                                          **kwargs)
        elif model_type == 'mms-llama-3-moe':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyMMSLlamaMoEForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                          config=cfg_pretrained,
                                                          **kwargs)
        elif model_type == 'llama-3-moe':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyLlamaMoEForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                          config=cfg_pretrained,
                                                          **kwargs)
        elif model_type == 'm-llama-3':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyLlamaMMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                          config=cfg_pretrained,
                                                          **kwargs)
        elif model_type == 'llava':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                          config=cfg_pretrained,
                                                          **kwargs)
        elif model_type == 'phi-3-moe':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyPhi3MoEForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                          config=cfg_pretrained,
                                                          **kwargs)
        elif model_type == 'mm-phi-3-moe':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyMMPhi3MoEForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                          config=cfg_pretrained,
                                                          **kwargs)
        elif model_type == 'm-phi-3':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyPhi3MMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                          config=cfg_pretrained,
                                                          **kwargs)
        elif model_type == 'mms-phi-3-moe':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyMMSPhi3MoEForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                          config=cfg_pretrained,
                                                          **kwargs)
        elif model_type == 'mma-phi-3-moe':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyMMAPhi3MoEForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                          config=cfg_pretrained,
                                                          **kwargs)

        elif model_type == 'moe-llava-phi3':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = EvalMoELLaVAPhiForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            import deepspeed
            deepspeed.init_distributed(dist_backend='nccl')
                    # Initialize the DeepSpeed-Inference engine
            ds_engine = deepspeed.init_inference(model,
                                                         # mp_size=2,
                                                         # dtype=torch.half,
                                                         checkpoint=None,
                                                         replace_with_kernel_inject=False)
            model = ds_engine.module

        mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
        mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
        model.load_state_dict(mm_projector_weights, strict=False)
    elif model_type=='llm':
         # Load language model
        print('Loading language model...')
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        image_processor = None
        if hasattr(model.config, "max_sequence_length"):
            context_len = model.config.max_sequence_length
        else:
            context_len = 2048
        if "llama3" in model_name.lower():
            print("this is llama3")
            tokenizer.eos_token_id = 128001
            model.generation_config.pad_token_id = tokenizer.eos_token_id

        if model.generation_config.pad_token_id is None:
            model.generation_config.pad_token_id = model.generation_config.eos_token_id
        return tokenizer, model, image_processor, context_len
    else:
        if model_type == 'phi-1.5' or model_type == 'phi-2':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyPhiForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'phi-3':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyPhi3ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'stablelm-2':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
            model = BunnyStableLMForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'qwen1.5-1.8b':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyQwen2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'llava_qwen':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = LlavaQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'llava_mistral':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = LlavaMistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True,attn_implementation="flash_attention_2", **kwargs)
        elif model_type == 'minicpm':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyMiniCPMForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'llama3-8b':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'llama-3-moe':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyLlamaMoEForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'm-llama-3':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyLlamaMMForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'llava':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True,
                                                          **kwargs)
        elif model_type == 'mms-llama-3-moe':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyMMSLlamaMoEForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True,
                                                          **kwargs)
        elif model_type == 'phi-3-moe':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyPhi3MoEForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True,
                                                          **kwargs)
        elif model_type == 'mm-phi-3-moe':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyMMPhi3MoEForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True,
                                                          **kwargs)
        elif model_type == 'mms-phi-3-moe':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyMMSPhi3MoEForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True,
                                                          **kwargs)
        elif model_type == 'mma-phi-3-moe':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyMMAPhi3MoEForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True,
                                                          **kwargs)
                                                             
        elif model_type == 'm-phi-3':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyPhi3MMForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True,
                                                          **kwargs)
        elif model_type == 'moe-llava-phi3':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
       
            model = EvalMoELLaVAPhiForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            import deepspeed
            deepspeed.init_distributed(dist_backend='nccl')
            print(model)
                    # Initialize the DeepSpeed-Inference engine
            ds_engine = deepspeed.init_inference(model,
                                                         # mp_size=2,
                                                         # dtype=torch.half,
                                                         checkpoint=None,
                                                         replace_with_kernel_inject=False)
            model = ds_engine.module
            
            

    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    vision_tower_aux = model.get_vision_tower_aux()
    if vision_tower_aux is not None:
        if not vision_tower_aux.is_loaded:
            vision_tower_aux.load_model()

    if not vision_tower.is_loaded:
        vision_tower.load_model()

    if getattr(model.config, "unfreeze_mm_vision_tower", False):
        if 'lora' in model_name.lower():
            assert model_base is not None
            vision_non_lora_trainables = {k[19:]: v for k, v in non_lora_trainables.items() if
                                          k.startswith('model.vision_tower.')}
            vision_tower.load_state_dict(vision_non_lora_trainables, strict=False)
        else:
            assert model_base is None
            from safetensors.torch import load_file
            vision_weights = {}
            for file_name in os.listdir(model_path):
                if file_name.endswith('safetensors'):
                    vision_weights.update(
                        {k[19:]: v for k, v in load_file(os.path.join(model_path, file_name)).items() if
                         k.startswith('model.vision_tower.')})
            vision_tower.load_state_dict(vision_weights, strict=True)

    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor
    image_processor_aux = None

    if vision_tower_aux is not None and hasattr(vision_tower_aux, 'image_processor'):
        vision_tower_aux.to(device=device, dtype=torch.float16)
        image_processor_aux = vision_tower_aux.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    if model_type == 'llama3-8b' or model_type == 'mms-llama-3-moe' or model_type == 'llama-3-moe' or model_type == 'm-llama-3' :
        tokenizer.eos_token_id = 128001
        model.generation_config.pad_token_id = tokenizer.eos_token_id

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    return tokenizer, model, image_processor,image_processor_aux, context_len
