import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

import json
import pandas as pd

import monai.transforms as mtf
from monai.data import load_decathlon_datalist
from monai.data import set_track_meta

from ..utils.utils import mask2box
from .dataset_info import dataset_info
from .prompt_templates import Caption_templates, Caption_abnormality, Seg_templates, OneOrgan_CT_reasoning_templates
from .term_dictionary import term_dict

import warnings
warnings.filterwarnings('ignore')

class SegDataset(Dataset):
    def __init__(self, args, tokenizer, tag="0000", description=False, mode='train'):
        self.args = args
        self.tokenizer = tokenizer

        self.tag = tag
        self.description = description
        self.mode = mode
        self.dataset_info = dataset_info

        root_path = args.seg_data_path
        if mode == "train":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="train",
            )
        elif mode == "validation":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="test",
            )
        elif mode == "test":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="test",
            )

        train_transform = mtf.Compose(
            [
                mtf.SpatialPadd(keys=["image", "seg"], spatial_size=args.img_size,mode='constant'),
                mtf.RandCropByPosNegLabeld(
                    keys=["image", "seg"],
                    label_key="seg",
                    spatial_size=args.img_size,
                    pos=2,
                    neg=1,
                    num_samples=1,
                    image_key="image",
                    image_threshold=0,
                ),
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.2, spatial_axes=(1, 2)),
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.2, spatial_axes=(1, 2)),
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.2, spatial_axes=(1, 2)),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
                # mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
                # mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.SpatialPadd(keys=["image", "seg"], spatial_size=args.img_size, mode='constant'),
                    mtf.ToTensord(keys=["image"], dtype=torch.float),
                    mtf.ToTensord(keys=["seg"], dtype=torch.int),
                ]
        )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif mode == 'test':
            self.transform = val_transform

        self.cls_questions = Seg_templates["cls_questions"]
        self.des_questions = Seg_templates["des_questions"]
        self.cls_answers = Seg_templates["cls_answers"]
        self.des_answers = Seg_templates["des_answers"]
        self.cls_no_answers = Seg_templates["cls_no_answers"]
        self.des_no_answers = Seg_templates["des_no_answers"]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            data = self.data_list[idx]

            image_path = data['image']
            seg_path = data['label']
            image_path = self.args.data_root + 'M3D-Seg/' + image_path[image_path.find('M3D_Seg_npy'):]
            seg_path = self.args.data_root + 'M3D-Seg/' + seg_path[seg_path.find('M3D_Seg_npy'):]

            image_array = np.load(image_path) #1*32*256*256, normalized
            seg_array = np.load(seg_path)
            if np.sum(seg_array) == 0:
                seg_array = np.zeros(image_array.shape, dtype=np.int8)
            cls_id = int(os.path.basename(seg_path).split('_')[1].split('.')[0])

            try:
                item = {
                    'image': image_array,
                    'seg': seg_array,
                }

                it = self.transform(item)
                if isinstance(it, list):
                    it = it[0]

                image = it['image']
                seg = it['seg']  # 1*D*H*W

                cls_list = self.dataset_info[self.tag]
                if not self.description:
                    question_temple = random.choice(self.cls_questions)
                    question = question_temple.format(cls_list[cls_id])
                    answer = random.choice(self.cls_answers)
                else:
                    question_temple = random.choice(self.des_questions)
                    question = question_temple.format(random.choice(term_dict[cls_list[cls_id]]))
                    answer = random.choice(self.des_answers).format(cls_list[cls_id])

                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length",
                    return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'seg': seg,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'impath': image_path,
                    'question_type': "seg",
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)

class MultiSegDataset(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super(MultiSegDataset, self).__init__()
        self.tokenizer = tokenizer

        self.dataset_info = dataset_info

        self.ds_list = []
        for dataset_code in self.dataset_info.keys():
            self.ds_list.append(SegDataset(args, tokenizer, tag=dataset_code, description=True, mode=mode))
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


