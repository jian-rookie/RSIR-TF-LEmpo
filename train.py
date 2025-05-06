import os
import json
import torch
import argparse
import numpy as np
from utils import *
import torch.nn as nn
from tqdm import tqdm
from losses import calcul_loss
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
    parser.add_argument('--pseudo_caption_path', type=str, default='', help='pseudo caption path')
    parser.add_argument('--epoch', type=int, default=30, help='number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both', 'none'], default='both')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'], help='list of attention matrices where putting a LoRA') 
    parser.add_argument('--r', default=8, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=16, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.2, type=float, help='dropout rate applied before the LoRA module')
    parser.add_argument('--ckpt_path', type=str, default='work_dir', help='checkpoint path')
    parser.add_argument('--task_name', type=str, default='clip', help='task name')
    parser.add_argument('--margain', type=float, default=0.2, help='margain of loss')
    parser.add_argument('--query', type=str, default='triplet_lang_pse', help='query feature type')
    parser.add_argument('--gallery', type=str, default='im_sg', help='gallery feature type')
    parser.add_argument('--llm', type=str, default='Qwen', choices=['Llama', 'Qwen', 'Mistral', 'Falcon', 'QwenMax', 'DeepSeekV3', 'Kimi'], help='large language model')
    return parser.parse_args()


def train(args, clip_model, train_loader, val_gallery_loader, val_query_loader):

    model = clip_model
    model_val = clip_model
    list_lora_layers = apply_lora(args, model)
    model = model.cuda() 
    mark_only_lora_as_trainable(model)

    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
            print(name, param.shape)
        all_params += param.numel()
    print(f"Trainable parameters: {trainable_params}")
    print(f"All parameters: {all_params}")
    print(f"Percentage of trainable parameters: {trainable_params/all_params}")
    
    optimizer = torch.optim.AdamW(get_lora_parameters(model), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20 * len(train_loader), gamma=0.7)

    best_mr, best_epoch = 0.0, 0
    
    for train_idx in range(args.epoch):
        model.train()
        # Train
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, args.epoch))
        # torch.autograd.set_detect_anomaly(True)

        for idx, batch in enumerate(tqdm(train_loader)):
            ref_images = batch['reference_image'].cuda()
            tar_images = batch['target_image'].cuda()
            ref_caps = clip.tokenize(batch['relative_captions'], truncate=True).cuda()
            ref_pse_caps = clip.tokenize(batch['pseudo_captions'], truncate=True).cuda()
            ref_sg_caps = clip.tokenize(batch['ref_scene_graph_captions'], truncate=True).cuda()
            tar_sg_caps = clip.tokenize(batch['tar_scene_graph_captions'], truncate=True).cuda()
            
            batch = ref_images.shape[0]
            ref_image_features = model.encode_image(ref_images)
            tar_image_features = model.encode_image(tar_images)
            ref_text_features = model.encode_text(ref_caps)
            ref_pse_text_features = model.encode_text(ref_pse_caps)
            ref_sg_text_features = model.encode_text(ref_sg_caps)
            tar_sg_text_features = model.encode_text(tar_sg_caps)

            ref_image_features = ref_image_features / ref_image_features.norm(dim=-1, keepdim=True)
            tar_image_features = tar_image_features / tar_image_features.norm(dim=-1, keepdim=True)
            ref_text_features = ref_text_features / ref_text_features.norm(dim=-1, keepdim=True)
            ref_pse_text_features = ref_pse_text_features / ref_pse_text_features.norm(dim=-1, keepdim=True)
            ref_sg_text_features = ref_sg_text_features / ref_sg_text_features.norm(dim=-1, keepdim=True)
            tar_sg_text_features = tar_sg_text_features / tar_sg_text_features.norm(dim=-1, keepdim=True)

            features = {'ref_image': ref_image_features, 'ref_text': ref_text_features,
                        'ref_sg_text': ref_sg_text_features, 'ref_pse_text': ref_pse_text_features,
                        'tar_image': tar_image_features, 'tar_sg_text': tar_sg_text_features}

            sim = cal_sim(args.query, args.gallery, **features)
            label = torch.arange(batch).cuda()
            loss = calcul_loss(sim, batch, args.margain)

            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('Loss: {:.4f}'.format(sum(loss_list)/len(loss_list)))

        rank_k = validate_test(val_query_loader, val_gallery_loader, model, args)
        print(', '.join([f"{{{', '.join([f'{key}: {value:.2f}' for key, value in rank_k.items()])}}}"]), flush=True)

        if rank_k['mR'] > best_mr:
            best_mr = rank_k['mR']
            best_rank = rank_k
            best_epoch = train_idx
            os.makedirs(f'{args.ckpt_path}/{args.task_name}', exist_ok=True)
            save_lora(args, list_lora_layers, f'{args.ckpt_path}/{args.task_name}/lora_best.pth')

    print("\n**** Best test recall@k\n")
    print("Best epoch: ", best_epoch)
    print(', '.join([f"{{{', '.join([f'{key}: {value:.2f}' for key, value in best_rank.items()])}}}"]), flush=True)


def validate_test(query_loader, gallery_loader, model, args):
    model.eval()
    with torch.no_grad():
        gallery_image_features, gallery_sg_features, gallery_image_names = [], [], []
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
        gallery_image_features = torch.cat(gallery_image_features, dim=0)
        gallery_sg_features = torch.cat(gallery_sg_features, dim=0)
            
        query_image_features, query_mod_features, query_pse_features, query_sg_features, target_image_names = [], [], [], [], []
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

        query_image_features = torch.cat(query_image_features, dim=0)
        query_mod_features = torch.cat(query_mod_features, dim=0)
        query_pse_features = torch.cat(query_pse_features, dim=0)   
        query_sg_features = torch.cat(query_sg_features, dim=0)

        features = {'ref_image': query_image_features, 'ref_text': query_mod_features,
                    'ref_sg_text': query_sg_features, 'ref_pse_text': query_pse_features,
                    'tar_image': gallery_image_features, 'tar_sg_text': gallery_sg_features}

        sim_matrix = cal_sim(args.query, args.gallery, **features)

        query2gallery = np.array([[1 if x == y else 0 for y in gallery_image_names] for x in target_image_names])
        
        rank_k = metrics_calc(sim_matrix, query2gallery, at=[1, 5, 10, 20])
    return rank_k


def main(args):
    model, preprocess_images, tokenizer = load_model(args.model_name, args.backbone)

    model.float()


    train_dataset = SHFDataset(args.dataset_path, args.pseudo_caption_path, split='train', mode='query', 
                               subset=args.dataset, preprocess=preprocess_images, adding_scene=True, llm=args.llm)
    val_gallery_dataset = SHFDataset(args.dataset_path, args.pseudo_caption_path, split='val', mode='gallery', 
                                     subset=args.dataset, preprocess=preprocess_images, adding_scene=True, llm=args.llm)
    val_query_dataset = SHFDataset(args.dataset_path, args.pseudo_caption_path, split='val', mode='query', 
                                   subset=args.dataset, preprocess=preprocess_images, adding_scene=True, llm=args.llm)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=8, 
        drop_last=True,
        pin_memory=(torch.cuda.is_available()))

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
        rank_k = validate_test(val_query_loader, val_gallery_loader, model, args)
        print(', '.join([f"{{{', '.join([f'{key}: {value:.2f}' for key, value in rank_k.items()])}}}"]), flush=True)
    else:
        # ------------------------------------------ CLIP Training ------------------------------------------
        train(args, model, train_loader, val_gallery_loader, val_query_loader)


if __name__ == '__main__':
    # Load config file
    args = get_arguments()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    main(args)