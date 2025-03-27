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


class ITRDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        with open(args.cap_data_path, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

                mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensor(dtype=torch.float),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
            self.data_list = self.data_list[:512]
        elif 'test' in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def truncate_text(self, input_text, max_tokens):
        def count_tokens(text):
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)

        if count_tokens(input_text) <= max_tokens:
            return input_text

        sentences = input_text.split('.')

        selected_sentences = []
        current_tokens = 0

        if sentences:
            selected_sentences.append(sentences.pop(0))

        while current_tokens <= max_tokens and sentences:
            random_sentence = random.choice(sentences)
            new_tokens_len = count_tokens(random_sentence)
            if current_tokens + new_tokens_len <= max_tokens and random_sentence not in selected_sentences:
                selected_sentences.append(random_sentence)
                current_tokens += new_tokens_len
            else:
                sentences.remove(random_sentence)

        truncated_text = '.'.join(selected_sentences)
        return truncated_text

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                image_path = data["image"]
                image_abs_path = os.path.join(self.data_root, image_path)

                image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
                # image = np.load(img_abs_path)[np.newaxis, ...]  # nomalized
                image = self.transform(image)

                text_path = data["text"]
                text_abs_path = os.path.join(self.data_root, text_path)
                with open(text_abs_path, 'r') as text_file:
                    raw_text = text_file.read()
                text = self.truncate_text(raw_text, self.args.max_length)

                text_tensor = self.tokenizer(
                    text, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                ret = {
                    'image': image,
                    'text': text,
                    'input_id': input_id,
                    'attention_mask': attention_mask,
                    'question_type': "Image_text_retrieval",
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)




class CapDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        with open(args.cap_data_path, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        self.caption_prompts = Caption_templates

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

                mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensor(dtype=torch.float),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                image_path = data["image"]
                image_abs_path = os.path.join(self.data_root, image_path)

                image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
                # image = np.load(img_abs_path)[np.newaxis, ...]  # nomalized
                image = self.transform(image)

                text_path = data["text"]
                text_abs_path = os.path.join(self.data_root, text_path)
                with open(text_abs_path, 'r') as text_file:
                    raw_text = text_file.read()
                answer = raw_text

                prompt_question = random.choice(self.caption_prompts)

                question = self.image_tokens + prompt_question

                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
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

                if torch.isnan(torch.max(image)) or torch.isnan(torch.max(input_id)) \
                        or torch.isnan(torch.max(attention_mask)):
                    print('data loader !!!!!!image is NaN >>')
                    # 抛出一个常规的ValueError异常
                    raise ValueError("image has NaN" + str(image_path))

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "Caption",
                }
                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class CaptionSegDataset(Dataset):
    def __init__(self, args, tokenizer, tag="0000", description=False, mode='train'):
        self.args = args
        self.tokenizer = tokenizer

        self.tag = tag
        self.description = description
        self.mode = mode
        self.dataset_info = dataset_info

        self.image_tokens = "<im_patch>" * args.proj_out_num

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

        self.exist_questions = OneOrgan_CT_reasoning_templates["exist_questions"]
        self.des_questions = OneOrgan_CT_reasoning_templates["description_reasoning_questions"]
        self.exist_answers = OneOrgan_CT_reasoning_templates["exist_answers"]
        self.des_answers = OneOrgan_CT_reasoning_templates["description_reasoning_yes_answers"]
        self.exist_no_answers = OneOrgan_CT_reasoning_templates["exist_no_answers"]
        self.des_no_answers = OneOrgan_CT_reasoning_templates["description_reasoning_no_answers"]

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

            try:
                image_array = np.load(image_path) #1*32*256*256, normalized
                seg_array = np.load(seg_path)
                if np.sum(seg_array) == 0:
                    seg_array = np.zeros(image_array.shape, dtype=np.int8)

                cls_id = int(os.path.basename(seg_path).split('_')[1].split('.')[0])

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
                vld_cls = torch.nonzero(torch.sum(seg, dim=(1, 2, 3))).flatten().tolist()
                if vld_cls:
                    if not self.description:
                        question_temple = random.choice(self.exist_questions)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.exist_answers).format(cls_list[cls_id])
                    else:
                        question_temple = random.choice(self.des_questions)
                        question = question_temple.format(random.choice(term_dict[cls_list[cls_id]]))
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.des_answers).format(cls_list[cls_id])
                else:
                    if not self.description:
                        question_temple = random.choice(self.exist_questions)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.exist_no_answers).format(cls_list[cls_id])
                    else:
                        question_temple = random.choice(self.des_questions)
                        question = question_temple.format(random.choice(term_dict[cls_list[cls_id]]))
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.des_no_answers).format(cls_list[cls_id])

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
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "Caption",
                }
                return ret

            except Exception as e:
                print(f"Error CaptionSegDataset in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)
                import traceback
                traceback.print_exc()


class CaptionRefSegDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.caption_abnormality = Caption_abnormality


        self.image_tokens = "<im_patch>" * args.proj_out_num

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
                [   mtf.SpatialPadd(keys=["image", "seg"], spatial_size=args.img_size,mode='constant'),
                    mtf.ToTensord(keys=["image"], dtype=torch.float),
                    mtf.ToTensord(keys=["seg"], dtype=torch.int),
                ]
        )
        set_track_meta(False)

        if mode == 'train':
            self.data_list = pd.read_csv(args.refseg_data_train_path, engine='python')
            self.transform = train_transform
        elif mode == 'validation':
            self.data_list = pd.read_csv(args.refseg_data_test_path, engine='python')
            self.transform = val_transform
        elif mode == 'test':
            self.data_list = pd.read_csv(args.refseg_data_test_path, engine='python')
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]
                image_path = os.path.join(self.args.refseg_data_path, data["Image"].replace("nii.gz","npy"))
                image_path = self.args.data_root + 'M3D-RefSeg/' + image_path[image_path.find('M3D_RefSeg_npy'):]

                seg_path = os.path.join(self.args.refseg_data_path, data["Mask"].replace("nii.gz","npy"))
                seg_path = self.args.data_root + 'M3D-RefSeg/' + seg_path[seg_path.find('M3D_RefSeg_npy'):]

                image_array = np.load(image_path)  # 1*32*256*256, normalized
                seg_array = np.load(seg_path)
                if np.sum(seg_array)==0:
                    # 抛出一个常规的ValueError异常
                    raise ValueError("sum(seg)==0" + str(image_path))
                seg_array = (seg_array == data["Mask_ID"]).astype(np.int8)

                item = {
                    "image": image_array,
                    "seg": seg_array,
                }

                it = self.transform(item)
                if isinstance(it, list):
                    it = it[0]

                image = it['image']
                seg = it['seg']
                vld_cls = torch.nonzero(torch.sum(seg, dim=(1, 2, 3))).flatten().tolist()
                if vld_cls:
                    answer = data["Text"]
                else:
                    answer = "no significant abnormalities."

                prompt_question = random.choice(self.caption_abnormality)
                question = self.image_tokens + prompt_question

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
                    question, max_length=self.args.max_length, truncation=True, padding="max_length",
                    return_tensors="pt"
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

                if torch.isnan(torch.max(image)) or torch.isnan(torch.max(input_id)) \
                        or torch.isnan(torch.max(attention_mask)):
                    print('data loader !!!!!!image is NaN >>')
                    # 抛出一个常规的ValueError异常
                    raise ValueError("image has NaN" + str(image_path))

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "Caption",
                }

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)

class VQADataset(Dataset):
    def __init__(self, args, tokenizer, close_ended=True, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.close_ended = close_ended

        self.image_tokens = "<im_patch>" * args.proj_out_num

        if mode == "train":
            self.data_list = pd.read_csv(args.vqa_data_train_path)
        elif mode == "validation":
            self.data_list = pd.read_csv(args.vqa_data_val_path, nrows=2048)
        elif "test" in mode:
            self.data_list = pd.read_csv(args.vqa_data_test_path)
        else:
            print("The mode is not desired ! ")

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

                mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensor(dtype=torch.float),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]
                image_abs_path = os.path.join(self.args.data_root, data["Image Path"])

                image = np.load(image_abs_path)  # nomalized, 0-1, C,D,H,W
                # image = np.load(img_path)[np.newaxis, ...]  # nomalized

                image = self.transform(image)

                if self.close_ended:
                    question = data["Question"]
                    choices = "Choices: A. {} B. {} C. {} D. {}".format(data["Choice A"], data["Choice B"], data["Choice C"], data["Choice D"])
                    question = question + ' ' + choices
                    answer = "{}. {}".format(data["Answer Choice"], data["Answer"])
                else:
                    question = data["Question"]
                    answer = str(data["Answer"])


                question = self.image_tokens + ' ' + question
                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt",
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

                if torch.isnan(torch.max(image)) or torch.isnan(torch.max(input_id)) \
                        or torch.isnan(torch.max(attention_mask)):
                    print('data loader !!!!!!image is NaN >>')
                    # 抛出一个常规的ValueError异常
                    raise ValueError("image has NaN" + str(image_abs_path))

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'answer_choice': data["Answer Choice"],
                    'question_type': data["Question Type"],
                }

                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class SegDataset(Dataset):
    def __init__(self, args, tokenizer, tag="0000", description=False, mode='train'):
        self.args = args
        self.tokenizer = tokenizer

        self.tag = tag
        self.description = description
        self.mode = mode
        self.dataset_info = dataset_info

        self.image_tokens = "<im_patch>" * args.proj_out_num

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
                vld_cls = torch.nonzero(torch.sum(seg, dim=(1, 2, 3))).flatten().tolist()
                if vld_cls:
                    if not self.description:
                        question_temple = random.choice(self.cls_questions)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.cls_answers)
                    else:
                        question_temple = random.choice(self.des_questions)
                        question = question_temple.format(random.choice(term_dict[cls_list[cls_id]]))
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.des_answers).format(cls_list[cls_id])
                else:
                    if not self.description:
                        question_temple = random.choice(self.cls_questions)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.cls_no_answers).format(cls_list[cls_id])
                    else:
                        question_temple = random.choice(self.des_questions)
                        question = question_temple.format(random.choice(term_dict[cls_list[cls_id]]))
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.des_no_answers).format(cls_list[cls_id])

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



class RefSegDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.image_tokens = "<im_patch>" * args.proj_out_num
        self.cls_no_answers = Seg_templates["cls_no_answers"]

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
                [   mtf.SpatialPadd(keys=["image", "seg"], spatial_size=args.img_size,mode='constant'),
                    mtf.ToTensord(keys=["image"], dtype=torch.float),
                    mtf.ToTensord(keys=["seg"], dtype=torch.int),
                ]
        )
        set_track_meta(False)

        if mode == 'train':
            self.data_list = pd.read_csv(args.refseg_data_train_path, engine='python')
            self.transform = train_transform
        elif mode == 'validation':
            self.data_list = pd.read_csv(args.refseg_data_test_path, engine='python')
            self.transform = val_transform
        elif mode == 'test':
            self.data_list = pd.read_csv(args.refseg_data_test_path, engine='python')
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]
                image_path = os.path.join(self.args.refseg_data_path, data["Image"].replace("nii.gz","npy"))
                seg_path = os.path.join(self.args.refseg_data_path, data["Mask"].replace("nii.gz","npy"))

                image_path = self.args.data_root + 'M3D-RefSeg/' + image_path[image_path.find('M3D_RefSeg_npy'):]
                seg_path = self.args.data_root + 'M3D-RefSeg/' + seg_path[seg_path.find('M3D_RefSeg_npy'):]

                image_array = np.load(image_path)  # 1*32*256*256, normalized
                seg_array = np.load(seg_path)
                if np.sum(seg_array)==0:
                    seg_array = np.zeros(image_array.shape, dtype=np.int8)
                    print("++image_path: ", image_path)
                seg_array = (seg_array == data["Mask_ID"]).astype(np.int8)

                item = {
                    "image": image_array,
                    "seg": seg_array,
                }

                it = self.transform(item)
                if isinstance(it, list):
                    it = it[0]

                image = it['image']
                seg = it['seg']  # C*D*H*W

                question = data["Question"]
                question = self.image_tokens + ' ' + question

                vld_cls = torch.nonzero(torch.sum(seg, dim=(1, 2, 3))).flatten().tolist()
                if vld_cls:
                    answer = data["Answer"]
                else:
                    answer = random.choice(self.cls_no_answers).format("Irregular target described")

                self.tokenizer.padding_side = "right"
                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
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
                    'question_type': "refseg",
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
        self.ds_list.append(RefSegDataset(args, tokenizer, mode=mode))
        for dataset_code in self.dataset_info.keys():
            self.ds_list.append(SegDataset(args, tokenizer, tag=dataset_code, description=False, mode=mode))
            self.ds_list.append(SegDataset(args, tokenizer, tag=dataset_code, description=True, mode=mode))
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class TextDatasets(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super(TextDatasets, self).__init__()
        self.dataset_info = dataset_info
        self.ds_list = [
            # CapDataset(args, tokenizer, mode),
            # VQADataset(args, tokenizer, close_ended=True, mode=mode),
            # VQADataset(args, tokenizer, close_ended=False, mode=mode),
            CaptionRefSegDataset(args, tokenizer, mode),
        ]
        for dataset_code in self.dataset_info.keys():
            self.ds_list.append(CaptionSegDataset(args, tokenizer, tag=dataset_code, description=False, mode=mode))
            self.ds_list.append(CaptionSegDataset(args, tokenizer, tag=dataset_code, description=True, mode=mode))
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class UniDatasets(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super(UniDatasets, self).__init__()
        self.ds_list = [
            # CapDataset(args, tokenizer, mode),
            # VQADataset(args, tokenizer, close_ended=True, mode=mode),
            # VQADataset(args, tokenizer, close_ended=False, mode=mode),
            # VQAYNDataset(args, tokenizer, mode=mode),
            # MultiPosDataset(args, tokenizer, mode),
            MultiSegDataset(args, tokenizer, mode),
            # MultiSegDataset(args, tokenizer, mode),
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]



