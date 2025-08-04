import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from bunny.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from bunny.conversation import conv_templates, SeparatorStyle
from bunny.model.builder import load_pretrained_model
from bunny.util.utils import disable_torch_init
from bunny.util.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def count_top_experts(top2_indices, num_experts):
    # 将 [728, 2] 展平为一维张量 [728 * 2]
    flat_indices = top2_indices.view(-1)
    # 使用 torch.bincount 统计每个专家被选中的次数
    expert_counts = torch.bincount(flat_indices, minlength=num_experts)
    # 找到被选中次数最多的两个专家及其对应的次数
    top2_counts = torch.topk(expert_counts, k=2)
    return top2_counts.indices, top2_counts.values


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, image_processor_aux ,context_len = load_pretrained_model(model_path, args.model_base, model_name,
                                                                           args.model_type)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    num_experts = 3  # 专家数量
    visual_expert_stats = [torch.zeros(num_experts, dtype=torch.long,device='cuda') for _ in range(16)]  # 16 层视觉专家统计
    language_expert_stats = [torch.zeros(num_experts, dtype=torch.long,device='cuda') for _ in range(16)]
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs

        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)[0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).to(dtype=model.dtype, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)
    #         input = model.prepare_inputs_for_generation(input_ids,images=image_tensor.unsqueeze(0).to(dtype=model.dtype, device='cuda', non_blocking=True))
          
    #         outputs = model(**input)
           

    #         # print(outputs.vis_router_logits)
    #         for layer_idx, logits in enumerate(outputs.vis_router_logits):
    #             if logits.shape[0] == 0:
    #                 continue  # 跳过没有视觉 token 的层
    #             top2_indices = torch.topk(logits, k=2, dim=1).indices
    #             flat_indices = top2_indices.view(-1)  # 展平成一维张量
    #             counts = torch.bincount(flat_indices, minlength=num_experts)
    #             visual_expert_stats[layer_idx] += counts.to(visual_expert_stats[layer_idx].device)  # 累积到全局统计

    #     # 累积语言专家统计
    #         for layer_idx, logits in enumerate(outputs.lan_router_logits):
    #             if logits.shape[0] == 0:
    #                 continue  # 跳过没有语言 token 的层
    #             top2_indices = torch.topk(logits, k=2, dim=1).indices
    #             flat_indices = top2_indices.view(-1)  # 展平成一维张量
    #             counts = torch.bincount(flat_indices, minlength=num_experts)
    #             language_expert_stats[layer_idx] += counts.to(language_expert_stats[layer_idx].device)  # 累积到全局统计
            
    # for layer_idx in range(16):
    #     vis_top_experts = torch.topk(visual_expert_stats[layer_idx], k=2)
    #     lan_top_experts = torch.topk(language_expert_stats[layer_idx], k=2)
    #     print(f"Layer {layer_idx + 1}:")
    #     print(f"  Visual Top 2 Experts: {vis_top_experts.indices.tolist()} with counts {vis_top_experts.values.tolist()}")
    #     print(f"  Language Top 2 Experts: {((lan_top_experts.indices)+1).tolist()} with counts {lan_top_experts.values.tolist()}")

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-type", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default=None)
    parser.add_argument("--question-file", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
