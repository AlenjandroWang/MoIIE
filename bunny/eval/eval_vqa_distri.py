import argparse
import itertools
import json
import os
import random
import time
from typing import Optional
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader , Dataset
import re
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


from bunny.model.builder import load_pretrained_model
from bunny.util.utils import disable_torch_init
from bunny.conversation import conv_templates, SeparatorStyle
from bunny.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from bunny.util.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria
)


ds_collections = {
    'docvqa_val': {
        'train': './perception_eval/docvqa/train.jsonl',
        'test': './perception_eval/docvqa/val.jsonl',
        'annotation': './perception_eval/docvqa/val_v1.0_withQT.json',
        # 'annotation': '/remote-home/share/multimodal-datasets/DocVQA/qas/demo_val.json',
        'img_dir': './perception_eval/docvqa_image',
        'metric': 'anls',
        'max_new_tokens': 100,
    },
    'ocrvqa_test': {
        'train': './perception_eval/ocrvqa/ocrvqa_train.jsonl',
        'test': './perception_eval/ocrvqa/ocr_select.jsonl',
        'img_dir': './perception_eval/ocrvqa_image',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    }
}

# https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                              target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def evaluate_relaxed_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            relaxed_correctness(elem['answer'].strip(), ann)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


def evaluate_exact_match_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            (1.0 if
             (elem['answer'].strip().lower() in ann.strip().lower() or ann.strip().lower() in elem['answer'].strip().lower()  ) else 0.0)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)



class VQADataset(Dataset):

    def __init__(self, train, test, prompt, img_dir, few_shot, sample_num):
        self.test = open(test).readlines()[:]
        self.prompt = prompt
        self.img_dir = img_dir

        self.few_shot = few_shot
        if few_shot > 0:
            self.train = open(train).readlines()
        if sample_num == 0:
            return

        assert args.dataset != 'docvqa_val'
        if sample_num > len(self.test):
            print('='*10)
            print(f'the num of dataset is {len(self.test)} , can\'t sample {sample_num}')
            sample_num = len(self.test)
        print(sample_num, f" sample_num")
        indices = random.sample(range(len(self.test)) , sample_num)
        self.test = [self.test[idx] for idx in indices]

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        data = json.loads(self.test[idx].strip())
        image, question, question_id, annotation = data['image'], data[
            'question'], data['question_id'], data.get('answer', None)

        few_shot_prompt = ''
        if self.few_shot > 0:
            few_shot_samples = random.sample(self.train, self.few_shot)
            for sample in few_shot_samples:
                sample = json.loads(sample.strip())
                few_shot_prompt += self.prompt.format(
                    sample['image'],
                    sample['question']) + f" {sample['answer']}"

        return {
            'question': question,
            'question_id': question_id,
            'annotation': annotation,
            'img_path': os.path.join(
                self.img_dir,
                image.split('/')[-1]
            )
        }

def collate_fn(batch):
    # print(batch)
    assert len(batch) == 1 , 'batch_size must be 1 !!!'
    question = batch[0]['question']
    question_id = batch[0]['question_id']
    annotation = batch[0]['annotation']
    img_path = batch[0]['img_path']
    return question , question_id , annotation , img_path


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def get_sentences(
    tokenizer , model , model_name , image_processor , image_processor_aux ,question, 
    img_path , conv_mode
):
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in question:
        if model.config.mm_use_im_start_end:
            question = re.sub(IMAGE_PLACEHOLDER, image_token_se, question)
        else:
            question = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, question)
    else:
        if model.config.mm_use_im_start_end:
            question = image_token_se + "\n" + question
        else:
            question = DEFAULT_IMAGE_TOKEN + "\n" + 'Please answer the question in one words.' + "\n" +question
    
    if "llama-2" in model_name.lower():
        conv_m = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_m = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_m = "mpt"
    elif "llama-3" in model_name.lower():
        conv_m = "llama"
    elif "phi" in model_name.lower():
        conv_m = "phi3"
    elif "qwen" in model_name.lower():
        conv_m = "qwen2_5"
    else:
        conv_m = "vicuna_v1"

    if conv_mode is not None and conv_m != conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_m, conv_mode, conv_mode
            )
        )
    else:
        conv_mode = conv_m

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    images_tensor_aux = None
    try:
        image = Image.open(img_path).convert("RGB")
        images = [image]
        images_tensor = process_images(
        images,
        image_processor,
        model.config
        ).to(dtype=torch.float16).cuda()

        if image_processor_aux is not None:
            images_tensor_aux = process_images(
            images,
            image_processor_aux,
            model.config
            ).to(dtype=torch.float16).cuda()
    except Exception as e:
     print(f'Error processing img: {img_path}')
     print(e)
     raise e

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
    ).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            images_aux = images_tensor_aux if images_tensor_aux is not None else None,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=ds_collections[args.dataset]['max_new_tokens'],
            use_cache=True,
        )

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()

    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )
    output = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    output = output.strip()
    if output.endswith(stop_str):
        output = output[: -len(stop_str)]
    output = output.strip()

    return output


def set_seed(seed=234):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b-lora", 
        help="liuhaotian/llava-v1.5-7b-lora, full-ft:liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default="lmsys/vicuna-7b-v1.5" , 
        help='None , lmsys/vicuna-7b-v1.5')
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--few-shot', type=int, default=0)
    parser.add_argument('--sample_num', type=int, default=0)
    parser.add_argument("--perturbation_params_path" , type=str , default='None' , help='path of perturbated model')
    # add for remove some layers
    parser.add_argument('--remove_layers' , nargs='+')
    args = parser.parse_args()

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    set_seed(seed=234)
    prompt = 'Answer the question using a single word or phrase.' #DocVQA OCRVQA

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path) + '_llava'
    device = torch.device("cuda")
    tokenizer , model , image_processor , image_processor_aux, _ = load_pretrained_model(
        args.model_path,
        args.model_base if args.model_base != 'None' else None, 
        model_name,
        args.model_type,
        device=device
    )
    print(model)    # perturbate model
    if args.perturbation_params_path != "None":
        counter = 0
        model_dict = model.state_dict()
        perturbation_model_dict = torch.load(args.perturbation_params_path)
        for key in perturbation_model_dict:
            new_key = 'model.' + key
            if new_key in model_dict:
                counter += 1
                model_dict[new_key] = perturbation_model_dict[key]
        model.load_state_dict(model_dict)
        print("==========Loading perturbated model==========")
        print(f'{counter} layers are changed')

    # need to remove some layers
    if args.remove_layers:
        new_module_list = nn.ModuleList(
            [module for idx , module in enumerate(model.model.layers) if str(idx) not in args.remove_layers]
        )
        model.model.layers = new_module_list
        print("==========Some layers removed==========")

    dataset = VQADataset(
        train=ds_collections[args.dataset]['train'],
        test=ds_collections[args.dataset]['test'],
        prompt=prompt,
        img_dir=ds_collections[args.dataset]['img_dir'],
        few_shot=args.few_shot,
        sample_num=args.sample_num
    )

    dataloader = DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    outputs = []
    for batch in tqdm(dataloader):
        question , question_id , annotation , img_path = batch
        output = get_sentences(
            tokenizer,
            model,
            model_name,
            image_processor,
            image_processor_aux,
            question,
            img_path,
            args.conv_mode
        )
        # print(question)
        # print(output)

        if args.dataset in ['docvqa_val', 'ocrvqa_test']:
            outputs.append({
                'questionId': question_id,
                'answer': output,
                'annotation': annotation,
            })
        else:
            raise NotImplementedError

    torch.distributed.barrier()
    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

    merged_outputs = [json.loads(_) for _ in merged_outputs]
    merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

    if torch.distributed.get_rank() == 0:
        print(f"Evaluating {args.dataset} ...")
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = f'./perception_eval/outputs/{args.dataset}_{args.save_dir}_fs{args.few_shot}.json'
        os.path.join(results_file)
        json.dump(merged_outputs, open(results_file, 'w', encoding='utf-8'), ensure_ascii=False)

        if ds_collections[args.dataset]['metric'] == 'anls':
            json.dump(merged_outputs,
                        open(results_file, 'w'),
                        ensure_ascii=False)
            print('python infographicsvqa_eval.py -g ' +
                    ds_collections[args.dataset]['annotation'] + ' -s ' +
                    results_file)
            os.system('python infographicsvqa_eval.py -g ' +
                        ds_collections[args.dataset]['annotation'] + ' -s ' +
                        results_file)
        print({'accuracy': evaluate_exact_match_accuracy(merged_outputs)})

    torch.distributed.barrier()