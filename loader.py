import os
import re
import json
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


class SHFDataset(Dataset):
    def __init__(self, dataset_path, pseudo_caption_path=None, split='train', 
                 mode='query', subset='airplane', preprocess=None,
                 adding_scene=True, llm=None):
        dataset_path = Path(dataset_path)
        self.dataset_path = dataset_path
        self.split = split
        self.mode = mode
        self.preprocess = preprocess
        self.adding_scene = adding_scene

        if mode not in ['query', 'gallery']:
            raise ValueError("mode should be in ['query', 'gallery']")
        if split not in ['train', 'val']:
            raise ValueError("split should be in ['train', 'val']")
        if subset not in ['airplane', 'tennis', 'WHDLD']:
            raise ValueError("subset should be in ['airplane', 'tennis', 'WHDLD']")

        if mode == 'query':
            self.triplets = []
            with open(dataset_path / 'captions' / f'cap.{subset}.{split}.json') as f:
                for element in json.load(f):
                    reference_name = self.supplement_suffix(element['candidate'])
                    target_name = self.supplement_suffix(element['target'])
                    relative_caption = self.merge_caption(element['captions']) 
                    self.triplets.append(
                        {
                            'reference_name': reference_name,
                            'target_name': target_name,
                            'relative_captions': relative_caption
                        }
                    )
        if mode == 'gallery':
            self.image_prefix_list = []
            with open(dataset_path / 'image_splits' / f'split.{subset}.{split}.json') as f:
                self.image_prefix_list = json.load(f)
            self.image_prefix_list = [self.supplement_suffix(image_prefix) for image_prefix in self.image_prefix_list]

        if self.adding_scene:
            self.pseudo_captions, self.scene_graph_captions = {}, {}
            with open(f'{pseudo_caption_path}/{subset}_{llm}_query_{split}.json') as f:
                self.pseudo_captions = json.load(f)
                print(f"Load pseudo captions {subset}_{llm}_query_{split}.json from {pseudo_caption_path}")
            with open(f'{pseudo_caption_path}/{subset}_{llm}_gallery_{split}.json') as f:
                self.scene_graph_captions = json.load(f)
                print(f"Load scene graph captions {subset}_{llm}_gallery_{split}.json from {pseudo_caption_path}")

        print(f"SHF {subset} {split} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'query':
                reference_name = self.triplets[index]['reference_name']
                target_name = self.triplets[index]['target_name']
                relative_caption = self.triplets[index]['relative_captions']
                reference_image_path = self.dataset_path / 'images'/ reference_name
                target_image_path = self.dataset_path / 'images'/ target_name
                
                if self.preprocess is not None:
                    reference_image = self.preprocess(Image.open(reference_image_path))
                    target_image = self.preprocess(Image.open(target_image_path))
                    if self.adding_scene:
                        return {
                            'reference_image': reference_image,
                            'reference_name': reference_name,
                            'target_image': target_image,
                            'target_name': target_name,
                            'relative_captions': relative_caption,
                            'pseudo_captions': self.remove_symbols_keep_spaces(self.pseudo_caption_clean(self.pseudo_captions[f'{reference_name}_{target_name}']['edit'])).lower().strip(),
                            'ref_scene_graph_captions': self.remove_symbols_keep_spaces(self.scene_graph_caption_clean(self.scene_graph_captions[reference_name]['convert'])).lower().strip(),
                            'tar_scene_graph_captions': self.remove_symbols_keep_spaces(self.scene_graph_caption_clean(self.scene_graph_captions[target_name]['convert'])).lower().strip()
                        }
                    else:
                        return {
                            'reference_image': reference_image,
                            'reference_name': reference_name,
                            'target_image': target_image,
                            'target_name': target_name,
                            'relative_captions': relative_caption,
                        }
                else:
                    return {
                        'reference_name': reference_name,
                        'target_name': target_name,
                        'relative_captions': relative_caption,
                    }
            elif self.mode == 'gallery':
                image_name = self.image_prefix_list[index]
                image_path = self.dataset_path / 'images'/ image_name
                if self.preprocess is not None:
                    image = self.preprocess(Image.open(image_path))
                    if self.adding_scene:
                        return {
                            'image': image,
                            'image_name': image_name,
                            'scene_graph_captions': self.remove_symbols_keep_spaces(self.scene_graph_caption_clean(self.scene_graph_captions[image_name]['convert'])).lower().strip()
                        }
                    else:
                        return {
                            'image': image,
                            'image_name': image_name
                        }
                else:
                    return {
                        'image_name': image_name
                    }
            else:
                raise ValueError("mode should be in ['query', 'gallery']")
        except Exception as e:
            print(f"Exception: {e}")
        return
    
    def __len__(self):
        if self.mode == 'query':
            return len(self.triplets)
        elif self.mode == 'gallery':
            return len(self.image_prefix_list)

    def supplement_suffix(self, image_prefix):

        path_prefix = self.dataset_path / 'images' / image_prefix 
        if os.path.exists(path_prefix.as_posix() + '.jpg'):
            return image_prefix + '.jpg'
        elif os.path.exists(path_prefix.as_posix() + '.tif'):
            return image_prefix + '.tif'
        else:
            raise ValueError(f"Image {image_prefix} not found")

    def merge_caption(self, captions):
        caps = [cap for cap in captions if cap]
        return " and ".join(caps)

    def pseudo_caption_clean(self, text):
        if 'Edited Description:' not in text:
            return text.split('\n')[-1]
        else:
            text = text.split('Edited Description:')[-1]
            if '\n' in text:
                text = text.split('\n')[0]
            return text

    def scene_graph_caption_clean(self, text):
        if 'Description:' not in text:
            return text.split('\n')[-1]
        else:
            text = text.split('Description:')[-1]
            if '\n' in text:
                text = text.split('\n')[0]
            return text

    def remove_symbols_keep_spaces(self, text):
    # 只保留大小写字母和空格，[^a-zA-Z ] 表示匹配非字母和非空格的字符
        return re.sub(r'[^a-zA-Z ]', '', text)