import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import pandas as pd
from monai import transforms
import monai.transforms as mtf
from monai.data import load_decathlon_datalist
from monai.data import set_track_meta
from .dataset_info import dataset_info
# from .prompt_templates import SAM_textPrompt
import warnings
warnings.filterwarnings('ignore')

class CaptionRefSegDataset(Dataset):
    def __init__(self, args, mode="train"):
        self.args = args
        self.mode = mode

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
                # mtf.OneOf(transforms=[
                #     mtf.Resized(keys=["image", "seg"], spatial_size=args.img_size),
                #     mtf.RandCropByPosNegLabeld(
                #         keys=["image", "seg"],
                #         label_key="seg",
                #         spatial_size=args.img_size,
                #         pos=2,
                #         neg=1,
                #         num_samples=1,
                #         image_key="image",
                #         image_threshold=0,
                #     ),
                # ],
                #     weights=[1, 1]
                # ),
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
                    mtf.SpatialPadd(keys=["image", "seg"], spatial_size=args.img_size,mode='constant'),
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
                text = data["Text"]

                ret = {
                    'image': image,
                    'promptarget': text,
                    'impath': image_path,
                    'seg': seg,
                    # 'input_id': input_id,
                    # 'attention_mask': attention_mask,
                    'question_type': "refseg",
                }
                return ret
            except Exception as e:
                print(f"Error in __getitem__ at refSegDataset index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)

class SegDataset(Dataset):
    def __init__(self, args, tag="0000", mode='train'):
        self.args = args
        # self.tokenizer = tokenizer
        self.ct_type = 'abdomen'
        self.tag = tag
        self.mode = mode
        self.dataset_info = dataset_info
        # self.sam_textPrompt = SAM_textPrompt

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
                # mtf.OneOf(transforms=[
                #     mtf.Resized(keys=["image", "seg"], spatial_size=args.img_size),
                #     mtf.RandCropByPosNegLabeld(
                #         keys=["image", "seg"],
                #         label_key="seg",
                #         spatial_size=args.img_size,
                #         pos=2,
                #         neg=1,
                #         num_samples=1,
                #         image_key="image",
                #         image_threshold=0,
                #     ),
                # ],
                #     weights=[1, 1]
                # ),
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

        val_transform = transforms.Compose(
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
                if np.sum(seg_array)==0:
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

                # target = str(cls_list)
                # if len(cls_list)>15:
                #     text = ','.join(random.sample(cls_list, 15))
                # elif len(cls_list)>1 and len(cls_list)<=15:
                #     text = ','.join(cls_list)
                # else:
                #     text = cls_list[0]
                # text = 'The  {} computed tomography scan revealed details of various organs, such as {}'.format(self.ct_type, text)
                # if "tumor" in target or "cyst" in target or "cancer" in target or "lesion" in target or "tumour" in target:
                #     text += ", and suggested the potential presence of organ lesions."
                #
                # text_tensor = self.tokenizer(
                #     text, max_length=self.args.max_length, truncation=True, padding="max_length",
                #     return_tensors="pt"
                # )

                # input_id = text_tensor["input_ids"][0]
                # attention_mask = text_tensor["attention_mask"][0]
                #
                # valid_len = torch.sum(attention_mask)
                # if valid_len < len(input_id):
                #     input_id[valid_len] = self.tokenizer.eos_token_id
                target = cls_list[cls_id]
                # promptarget = random.choice(self.sam_textPrompt).format(target)
                # promptarget = 'A {} in the computerized tomography.'.format(target)
                promptarget = 'A computerized tomography of a {}.'.format(target)
                ret = {
                    'image': image,
                    'promptarget': promptarget,
                    'impath': image_path,
                    'seg': seg,
                    'question_type': "seg",
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at segdata cls_id {cls_id} image_path {image_path} index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)
                import traceback
                traceback.print_exc()

class SegDatasets(Dataset):
    def __init__(self, args, mode='train'):
        super(SegDatasets, self).__init__()
        # self.tokenizer = tokenizer
        self.dataset_info = dataset_info

        self.ds_list = []
        # self.ds_list.append(CaptionRefSegDataset(args, mode=mode))
        for dataset_code in self.dataset_info.keys():
            self.ds_list.append(SegDataset(args, tag=dataset_code, mode=mode))
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
