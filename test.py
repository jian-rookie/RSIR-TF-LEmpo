import os
import cv2
import clip
import json
import time
import torch
import argparse
import numpy as np
import pandas as pd
from utils import *
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from metrics import metrics_calc
from loader import SHFDataset
from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from loralib import layers as lora_layers


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def get_arguments():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--model_name', type=str, default='clip', choices=['remoteclip', 'clip', 'georsclip', 'openai_clip'], help='pre-trained model')
    parser.add_argument('--backbone', type=str, default='ViT-L-14', choices=['ViT-B/32', 'ViT-L/14'], help='pre-trained model type')
    parser.add_argument('--dataset', type=str, default='airplane', choices=['airplane', 'tennis', 'WHDLD'], help='choose dataset')
    parser.add_argument('--dataset_path', type=str, default='', help='dataset path')
    parser.add_argument('--pseudo_caption_path', type=str, default='precomputed/pseudo_caption', help='pseudo caption path')
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both', 'none'], default='both')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'], help='list of attention matrices where putting a LoRA') 
    parser.add_argument('--r', default=8, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=16, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.2, type=float, help='dropout rate applied before the LoRA module')
    parser.add_argument('--ckpt_path', type=str, default='work_dir', help='checkpoint path')
    parser.add_argument('--task_name', type=str, default='clip', help='task name')
    parser.add_argument('--query', type=str, default='triplet_lang_pse', help='query feature type')
    parser.add_argument('--gallery', type=str, default='im_sg', help='gallery feature type')
    parser.add_argument('--llm', type=str, default='Qwen', choices=['Llama', 'Qwen', 'Mistral', 'Falcon', 'QwenMax', 'DeepSeekV3', 'Kimi'], help='large language model')
    parser.add_argument('--csv_filename', type=str, default=None, help='CSV filename to save results')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name for logging')
    parser.add_argument('--save_top5', action='store_true', help='Save top 5 predictions to CSV')
    return parser.parse_args()


def save_top5_predictions_to_csv(similarity, query_image_names, target_image_names, gallery_image_names, 
                                 modifications, ref_scence_graphs, ref_pseudo_captions, gallery_scene_graphs, output_file='predictions.csv'):
    # Ensure similarity is a 2D matrix with dimensions (num_queries, num_gallery_images)
    assert similarity.shape[0] == len(query_image_names), "Mismatch between similarity matrix rows and query image names length"
    assert similarity.shape[1] == len(gallery_image_names), "Mismatch between similarity matrix columns and gallery image names length"

    df = pd.DataFrame(columns=['Query Image Name', 'Target Image Name', 'Top 1 Image Name', 'Top 2 Image Name', 'Top 3 Image Name', 'Top 4 Image Name', 
                               'Top 5 Image Name', 'modification text', 'ref_scene_graph', 'ref_pseudo_caption', 'Top 1 Scene Graph',
                               'Top 2 Scene Graph', 'Top 3 Scene Graph', 'Top 4 Scene Graph', 'Top 5 Scene Graph'])
    # Iterate over each query
    for i, query_name in enumerate(query_image_names):
        # Get the similarity scores for the current query with all gallery images
        sim_scores = similarity[i]

        # Get indices of top 5 most similar gallery images
        top5_indices = np.argsort(sim_scores)[::-1][:5]

        # Get the top 5 gallery image names
        top5_image_names = [gallery_image_names[idx] for idx in top5_indices]
        top5_scence_graphs = [gallery_scene_graphs[idx] for idx in top5_indices]
        element = {
            'Query Image Name': query_name,
            'Target Image Name': target_image_names[i],
            'Top 1 Image Name': top5_image_names[0],
            'Top 2 Image Name': top5_image_names[1],
            'Top 3 Image Name': top5_image_names[2],
            'Top 4 Image Name': top5_image_names[3],
            'Top 5 Image Name': top5_image_names[4],
            'modification text': modifications[i],
            'ref_scene_graph': ref_scence_graphs[i],
            'ref_pseudo_caption': ref_pseudo_captions[i],
            'Top 1 Scene Graph': top5_scence_graphs[0],
            'Top 2 Scene Graph': top5_scence_graphs[1],
            'Top 3 Scene Graph': top5_scence_graphs[2],
            'Top 4 Scene Graph': top5_scence_graphs[3],
            'Top 5 Scene Graph': top5_scence_graphs[4]
        }
        df = df._append(element, ignore_index=True)
        
    # Save the DataFrame to CSV
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def record_result(result, csv_filename, exp_name, dataset):
    fieldnames = ['experiment_name', 'dataset', 'R@1', 'R@5', 'R@10', 'R@20', 'mR', 
                  'P@1', 'P@5', 'P@10', 'P@20', 'AP']
    result['experiment_name'] = exp_name
    result['dataset'] = dataset

    file_exists = os.path.isfile(csv_filename)
    
    with open(csv_filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def test(query_loader, gallery_loader, model, args):
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        gallery_image_features, gallery_sg_features, gallery_image_names, gallery_scene_graph_captions = [], [], [], []
        for i, batch in enumerate(tqdm(gallery_loader)):
            images = batch['image'].cuda()
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            gallery_image_features.append(image_features)

            sg_cap = clip.tokenize(batch['scene_graph_captions'], truncate=True).cuda()
            sg_features = model.encode_text(sg_cap)
            sg_features = sg_features / sg_features.norm(dim=-1, keepdim=True)
            gallery_sg_features.append(sg_features)
            gallery_image_names.extend(batch['image_name'])
            gallery_scene_graph_captions.extend(batch['scene_graph_captions'])
        gallery_image_features = torch.cat(gallery_image_features, dim=0)
        gallery_sg_features = torch.cat(gallery_sg_features, dim=0)
            
        query_image_features, query_mod_features, query_pse_features, query_sg_features, target_image_names, query_image_names = [], [], [], [], [], []
        modification_texts, ref_scene_graph_captions, ref_pseudo_captions = [], [], []
        for i, batch in enumerate(tqdm(query_loader)):
            ref_images = batch['reference_image'].cuda()
            ref_image_features = model.encode_image(ref_images)
            ref_image_features = ref_image_features / ref_image_features.norm(dim=-1, keepdim=True)
            query_image_features.append(ref_image_features)

            ref_caps = clip.tokenize(batch['relative_captions'], truncate=True).cuda()
            ref_text_features = model.encode_text(ref_caps)
            ref_text_features = ref_text_features / ref_text_features.norm(dim=-1, keepdim=True)
            query_mod_features.append(ref_text_features)

            ref_pse_caps = clip.tokenize(batch['pseudo_captions'], truncate=True).cuda()
            ref_pse_text_features = model.encode_text(ref_pse_caps)
            ref_pse_text_features = ref_pse_text_features / ref_pse_text_features.norm(dim=-1, keepdim=True)
            ref_sg_caps = clip.tokenize(batch['ref_scene_graph_captions'], truncate=True).cuda()

            ref_sg_text_features = model.encode_text(ref_sg_caps)
            ref_sg_text_features = ref_sg_text_features / ref_sg_text_features.norm(dim=-1, keepdim=True)
            query_pse_features.append(ref_pse_text_features)
            query_sg_features.append(ref_sg_text_features)
            target_image_names.extend(batch['target_name'])
            query_image_names.extend(batch['reference_name'])
            modification_texts.extend(batch['relative_captions'])
            ref_scene_graph_captions.extend(batch['ref_scene_graph_captions'])
            ref_pseudo_captions.extend(batch['pseudo_captions'])

        query_image_features = torch.cat(query_image_features, dim=0)
        query_mod_features = torch.cat(query_mod_features, dim=0)
        query_pse_features = torch.cat(query_pse_features, dim=0)   
        query_sg_features = torch.cat(query_sg_features, dim=0)

        features = {'ref_image': query_image_features, 'ref_text': query_mod_features,
                    'ref_sg_text': query_sg_features, 'ref_pse_text': query_pse_features,
                    'tar_image': gallery_image_features, 'tar_sg_text': gallery_sg_features}

        sim_matrix = cal_sim(args.query, args.gallery, **features)

        print(f"Time taken: {time.time() - start_time:.6f}s")
        print(f"Query image features shape: {query_image_features.shape}")
        print(f"Similarity matrix shape: {sim_matrix.shape}")

        query2gallery = np.array([[1 if x == y else 0 for y in gallery_image_names] for x in target_image_names])
        
        if args.save_top5:
            save_top5_predictions_to_csv(sim_matrix.cpu().numpy(), query_image_names, target_image_names, gallery_image_names, 
                                         modification_texts, ref_scene_graph_captions, ref_pseudo_captions, gallery_scene_graph_captions,
                                         output_file=os.path.join(args.ckpt_path, args.task_name, 'predictions.csv'))
        rank_k = metrics_calc(sim_matrix, query2gallery, at=[1, 5, 10, 20])
    return rank_k


def main(args):
    model, preprocess_images, tokenizer = load_model(args.model_name, args.backbone)

    val_gallery_dataset = SHFDataset(args.dataset_path, args.pseudo_caption_path, split='val', mode='gallery', 
                                     subset=args.dataset, preprocess=preprocess_images, adding_scene=True, llm=args.llm)
    val_query_dataset = SHFDataset(args.dataset_path, args.pseudo_caption_path, split='val', mode='query', 
                                   subset=args.dataset, preprocess=preprocess_images, adding_scene=True, llm=args.llm)

    val_gallery_loader = torch.utils.data.DataLoader(
        val_gallery_dataset,
        batch_size=512,
        num_workers=8,
        shuffle=False,
        drop_last=False,
        pin_memory=(torch.cuda.is_available()))
    val_query_loader = torch.utils.data.DataLoader(
        val_query_dataset,
        batch_size=512,
        num_workers=8,
        shuffle=False,
        drop_last=False,
        pin_memory=(torch.cuda.is_available()))

    if args.encoder == 'none':
        model = model.float()
        model = model.cuda()
        rank_k = test(val_query_loader, val_gallery_loader, model, args)
        print(', '.join([f"{{{', '.join([f'{key}: {value:.2f}' for key, value in rank_k.items()])}}}"]), flush=True)
    else:
        model.float()
        list_lora_layers = apply_lora(args, model)
        model = model.cuda()
        load_lora(args, list_lora_layers, f'{args.ckpt_path}/{args.task_name}/lora_best.pth')
        rank_k = test(val_query_loader, val_gallery_loader, model, args)
        print(', '.join([f"{{{', '.join([f'{key}: {value:.2f}' for key, value in rank_k.items()])}}}"]), flush=True)

    if args.csv_filename:
        record_result(rank_k, args.csv_filename, args.experiment_name, args.dataset)
        print(f"Results recorded in {args.csv_filename}")

if __name__ == '__main__':
    # Load config file
    args = get_arguments()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    main(args)