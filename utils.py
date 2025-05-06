import torch
import pickle
import numpy as np
from PIL import Image
import csv
import open_clip
import clip
import argparse


def load_model(model_name, backbone):
    model_type = backbone.replace('/', '-')
    if model_name == 'openai_clip':
        model, preprocess_images = clip.load(f'ckpts/{model_type}.pt')
        tokenizer = clip.tokenize
    else:
        model, _, preprocess_images = open_clip.create_model_and_transforms(model_type)
        tokenizer = open_clip.get_tokenizer(model_type)

        if model_name == 'remoteclip':
            ckpt = torch.load(f"ckpts/RemoteCLIP-{model_type}.pt", map_location="cpu")
        elif model_name == 'clip':
            ckpt = torch.load(f"ckpts/CLIP-{model_type}.bin", map_location="cpu")
        elif model_name == 'georsclip':
            ckpt = torch.load(f"ckpts/RS5M_{model_type}.pt", map_location="cpu")
        message = model.load_state_dict(ckpt)
        print(message)
    print(f"{model_name} {model_type} has been loaded!")
    model = model.cuda().eval()

    return model, preprocess_images, tokenizer


def cal_sim(query_type, gallery_type, **kwargs):
    sim = None
    gallery_feature = None
    if gallery_type == 'im':
        gallery_feature = kwargs['tar_image']
    elif gallery_type == 'sg':
        gallery_feature = kwargs['tar_sg_text']
    elif gallery_type == 'im_sg':
        gallery_feature = kwargs['tar_image'] + kwargs['tar_sg_text']
    else:
        raise ValueError('Gallery type must in [im, sg, im_sg]')

    if query_type == 'triplet_lang_pse':
        sim_triplet = (kwargs['ref_image'] + kwargs['ref_text']) @ gallery_feature.T
        sim_lang = (kwargs['ref_sg_text'] + kwargs['ref_text']) @ gallery_feature.T
        sim_pse = kwargs['ref_pse_text'] @ gallery_feature.T
        sim = sim_triplet + sim_lang + sim_pse
    elif query_type == 'triplet':
        sim = (kwargs['ref_image'] + kwargs['ref_text']) @ gallery_feature.T
    elif query_type == 'lang':
        sim = (kwargs['ref_sg_text'] + kwargs['ref_text']) @ gallery_feature.T
    elif query_type == 'pse':
        sim = kwargs['ref_pse_text'] @ gallery_feature.T
    elif query_type == 'triplet_lang':
        sim_triplet = (kwargs['ref_image'] + kwargs['ref_text']) @ gallery_feature.T
        sim_lang = (kwargs['ref_sg_text'] + kwargs['ref_text']) @ gallery_feature.T
        sim = sim_triplet + sim_lang
    elif query_type == 'triplet_pse':
        sim_triplet = (kwargs['ref_image'] + kwargs['ref_text']) @ gallery_feature.T
        sim_pse = kwargs['ref_pse_text'] @ gallery_feature.T
        sim = sim_triplet + sim_pse
    elif query_type == 'lang_pse':
        sim_lang = (kwargs['ref_sg_text'] + kwargs['ref_text']) @ gallery_feature.T
        sim_pse = kwargs['ref_pse_text'] @ gallery_feature.T
        sim = sim_lang + sim_pse
    elif query_type == 'im':
        sim = kwargs['ref_image'] @ gallery_feature.T
    elif query_type == 'sg':
        sim = kwargs['ref_sg_text'] @ gallery_feature.T
    elif query_type == 'te':
        sim = kwargs['ref_text'] @ gallery_feature.T
    else:
        raise ValueError('Query type must in [triplet_lang_pse, triplet, lang, pse, triplet_lang, triplet_pse, lang_pse]')
    return sim
