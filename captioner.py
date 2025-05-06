import os
import json
import time
import torch
import prompts
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from torch.utils.data import DataLoader
from loader import SHFDataset
from vlgfm import chat_with_Llama, chat_with_api, LLM_API_Captioner


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='airplane', choices=['airplane', 'tennis', 'WHDLD'], help='choose dataset')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'], help='choose dataset split')
    parser.add_argument('--mode', type=str, default='gallery', choices=['gallery', 'query'], help='choose dataset mode')
    parser.add_argument('--dataset_path', type=str, default='', help='dataset path')
    parser.add_argument('--save_path', type=str, default='', help='save path')
    parser.add_argument('--batch_size', type=int, default=128, help='dataloader batch size')
    parser.add_argument('--llm', type=str, default='Llama', choices=['Llama', 'Qwen', 'Mistral', 'Falcon', 'QwenMax', 'DeepSeekV3', 'Kimi'], help='large language model')
    args, remaining_args = parser.parse_known_args()
    return args, remaining_args


def get_caption_agent(args, remaining_args):
    llm_dict = {'Llama': chat_with_Llama, 'Qwen': chat_with_Llama, 'Mistral': chat_with_Llama, 'Falcon': chat_with_Llama,
                'QwenMax': chat_with_api, 'DeepSeekV3': chat_with_api, 'Kimi': chat_with_api}

    tokenizer, model, image_processor = None, None, None
    caption_agent = None
    
    if args.llm in ['Llama', 'Qwen', 'Mistral', 'Falcon']:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_path_map = {
            'Llama': 'ckpts/Meta-Llama-3.1-8B-Instruct',
            'Qwen': 'ckpts/Qwen2.5-7B-Instruct',
            'Mistral': 'ckpts/Mistral-7B-Instruct-v0.3',
            'Vicuna': 'ckpts/Vicuna-7b-v1.5',
            'Falcon': 'ckpts/Falcon3-7B-Instruct'
        }
        model_id = model_path_map.get(args.llm, None)
        print('model id:', model_id, flush=True)
        dtype = torch.bfloat16
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # tokenizer.add_special_tokens({"pad_token":"<pad>"})
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", torch_dtype=dtype,)
        caption_agent = partial(llm_dict[args.llm], tokenizer=tokenizer, model=model)
    
    elif args.llm in ['QwenMax', 'DeepSeekV3', 'Kimi']:
        model = LLM_API_Captioner(args.llm)
        caption_agent = partial(llm_dict[args.llm], model=model)
    return caption_agent


def get_dataset(args):
    if args.dataset in ['airplane', 'tennis', 'WHDLD']:
        gallery_dataset = SHFDataset(args.dataset_path, split=args.split, mode='gallery', subset=args.dataset, preprocess=None)
        gallery_dataloader = DataLoader(gallery_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        query_dataset = SHFDataset(args.dataset_path, split=args.split, mode='query', subset=args.dataset, preprocess=None) 
        query_dataloader = DataLoader(query_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
        gallery_image_names = []
        for batch in gallery_dataloader:
            gallery_image_names.extend(batch['image_name'])
        
        attribute_dict = {}
        attribute_file = os.path.join(args.dataset_path, f'{args.dataset}_attributes.json')
        if args.dataset in ['airplane', 'tennis']:
            with open(attribute_file, 'r') as f:
                attribute_list = json.load(f) # list of dict
            for attribute in attribute_list:
                for key, value in attribute.items():
                    attribute_dict[key] = value
        else:
            with open(attribute_file, 'r') as f:
                attribute_dict = json.load(f)
    
        query_image_name_prompt_dict = {}
        for batch in query_dataloader:
            for i in range(len(batch['reference_name'])):
                query_image_name_prompt_dict[batch['reference_name'][i]+ '_' + batch['target_name'][i]] = batch['relative_captions'][i]
        return gallery_image_names, query_image_name_prompt_dict, attribute_dict


def generate_caption(args, remaining_args):
    caption_agent = get_caption_agent(args, remaining_args)
    caption_path = os.path.join(args.save_path, 'precomputed', 'pseudo_caption')
    os.makedirs(caption_path, exist_ok=True)
    
    # Prompt for SGI
    convert_prompt_dict = {'airplane': prompts.airplane_convert_prompt, 'tennis': prompts.tennis_convert_prompt, 'WHDLD': prompts.WHDLD_convert_prompt}
    convert_prompt = convert_prompt_dict[args.dataset]
    convert_template_dict = {'airplane': prompts.airplane_convert_template, 'tennis': prompts.tennis_convert_template, 'WHDLD': prompts.WHDLD_convert_template}
    convert_template = convert_template_dict[args.dataset]

    # Prompt for PCG
    llama_prompt_dict = {'airplane': prompts.airplane_llama_prompt, 'tennis': prompts.tennis_llama_prompt, 'WHDLD': prompts.WHDLD_llama_prompt}
    llama_prompt = llama_prompt_dict[args.dataset]
    llama_template_dict = {'airplane': prompts.airplane_llama_template, 'tennis': prompts.tennis_llama_template, 'WHDLD': prompts.WHDLD_llama_template}
    llama_template = llama_template_dict[args.dataset]

    gallery_image_names, query_image_name_prompt_dict, attribute_dict = get_dataset(args)
    start_time = time.time()

    # SGI
    if args.mode == 'gallery':
        for image_name in tqdm(gallery_image_names, desc='Processing images'):
            attribute = json.dumps(attribute_dict[image_name.split('.')[0]])
            query_prompts = [convert_prompt, 
                             convert_template.format(attribute)]
            with torch.no_grad():
                try:
                    answers = caption_agent(cur_prompt=query_prompts)
                except torch.cuda.OutOfMemoryError:
                    print(f'{image_name} out of memory, {query_prompts}')
                    raise 
            captions[image_name] = {
                "convert": answers[-1]
            }

    # PCG
    if args.mode == 'query':
        for query_label in tqdm(query_image_name_prompt_dict.keys(), desc='Processing images', mininterval=100):
            
            query_name, target_name = query_label.split('_')
            if query_label not in captions:
                captions[query_label] = {}
                attribute = json.dumps(attribute_dict[query_name.split('.')[0]])
                query_prompts = [llama_prompt, 
                                 llama_template.format(attribute, query_image_name_prompt_dict[query_label])]
                answers = caption_agent(cur_prompt=query_prompts)
                captions[query_label]['edit'] = answers[-1]

        print(f'Processing time: {time.time() - start_time:.6f} second')
    save_json_path = f'{caption_path}/{args.dataset}_{args.llm}_{args.mode}_{args.split}.json'
    with open(save_json_path, 'w') as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)
    print(f'Captions have been saved in {save_json_path}')


if __name__ == "__main__":
    args, remaining_args = get_arguments()
    # get_dataset(args)
    generate_caption(args, remaining_args)