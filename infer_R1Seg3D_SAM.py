import os
import random
from typing import Optional
import transformers
from dataclasses import dataclass, field
import torch
from monai.metrics import DiceMetric
from LaMed.src.utils.nii_utils import saveToNii
from LaMed.src.utils.slidingWindowInference_SAM import sliding_window_inference
import numpy as np
import monai.transforms as mtf
from LaMed.src.dataset.dataset_info import dataset_info
from LaMed.src.model.r1Seg3DSAM_Config import R1Seg3DSAM_Config
from LaMed.src.model.build_vit3dseg import model_registry
from monai import transforms
from safetensors.torch import load_file

local_rank = 0
def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    sam_bert_path: Optional[str] = field(default="./LaMed/pretrained_model/bert_base_uncased/")
    pretrained_model: Optional[str] = field(default="/",
                                              metadata={"help": "Path to the model weight."})
    need_text_en: bool = field(default=True)  # of build TextEncoder
    test_mode: bool = field(default=True)
    in_channels: int = field(default=1)
    img_size: tuple = field(default=(32, 256, 256))
    patch_size: tuple = field(default=(4, 16, 16))
    hidden_size: int = field(default=768)
    mlp_dim: int = field(default=3072)
    num_layers: int = field(default=12)
    num_heads: int = field(default=12)
    pos_embed: str = field(default="perceptron")
    dropout_rate: float = field(default=0.0)
    spatial_dims: int = field(default=3)
    num_clicks: int = field(default=0)

@dataclass
class InferringArguments(transformers.TrainingArguments):
    output_nii: str = field(default="image.nii.gz")
    image_path: str = field(default="/defaultShare/M3D_Data/M3D-Seg/M3D_Seg_npy/0011/s0733/image.npy")
    seg_path: str = field(default="/defaultShare/M3D_Data/M3D-Seg/M3D_Seg_npy/0011/s0733/masks/mask_45.npy")
    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    output_dir: str = "./LaMed/output/LaMed-pretrain-test"
    infer_overlap: float = 0.5

def main():
    # gpu = 0
    # torch.cuda.set_device(gpu)
    # device = torch.device('cuda') # 'cpu', 'cuda'
    # dtype = torch.bfloat16 # or bfloat16, float16, float32
    print("device_count：", torch.cuda.device_count())
    # # for single GPU
    # import torch.distributed as dist
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12345'
    # dist.init_process_group(backend='nccl', rank=0, world_size=1)
    parser = transformers.HfArgumentParser((ModelArguments, InferringArguments))
    model_args, inferring_args = parser.parse_args_into_dataclasses()


    # model_args.max_length = data_args.max_length
    config = R1Seg3DSAM_Config.from_dict(vars(model_args))
    config.mm_hidden_size = model_args.hidden_size
    model = model_registry['vit'](config=config, checkpoint=None) # checkpoint for pretrained ViT
    # rank0_print(model)

    if model_args.pretrained_model is not None:
        if model_args.pretrained_model.find(".safetensors") > 0:
            ckpt = load_file(model_args.pretrained_model)
        else:
            ckpt = torch.load(model_args.pretrained_model, map_location='cpu')
        model.load_state_dict(ckpt, strict=True)
        # rank0_print("pretrained_model >>>>>>>>>>>>>", ckpt.keys())
        rank0_print(">>>>>>>>>>>>>load pretrained model>>>>>>>>>>>>>")


    val_transform = transforms.Compose(
        [
            mtf.SpatialPadd(keys=["image", "seg"], spatial_size=model_args.img_size, mode='constant'),
            mtf.ToTensord(keys=["image"], dtype=torch.float),
            mtf.ToTensord(keys=["seg"], dtype=torch.int),
        ]
    )
    image_array = np.load(inferring_args.image_path)  # 1*32*256*256, normalized
    seg_array = np.load(inferring_args.seg_path)
    if np.sum(seg_array) == 0:
        seg_array = np.zeros(image_array.shape, dtype=np.int8)
    cls_id = int(os.path.basename(inferring_args.seg_path).split('_')[1].split('.')[0])
    item = {
        'image': image_array,
        'seg': seg_array,
    }
    it = val_transform(item)
    if isinstance(it, list):
        it = it[0]
    image = it['image']
    seg = it['seg']  # 1*D*H*W

    cls_list = dataset_info["0011"]
    promptarget = 'A computerized tomography of a {}.'.format(cls_list[cls_id])
    print("promptarget", promptarget)

    #inferring
    model.eval()
    model = model.cuda()

    with torch.no_grad():
        # Hausdorff_metric = HausdorffDistance(percentile=95.0)
        acc_func = DiceMetric(include_background=False, reduction="mean", get_not_nans=True, num_classes=1, ignore_empty=False)
        val_input, val_label = (image.cuda(), seg.cuda())
        # attention_mask = question_tensor["attention_mask"].cuda()
        print("Inference on case {}".format(inferring_args.image_path))
        # print("question", question)
        print("val_input shape", val_input.shape)
        print("val_label shape", val_label.shape)
        # print("input_id shape", input_id.shape)
        # print("attention_mask shape", attention_mask.shape)

        # start = time.time()
        # save Nii
        # saveToNii("data_{}".format(inferring_args.output_nii), val_input.cpu(), rot180=True, out_dir=inferring_args.output_dir)
        saveToNii("target_{}".format(inferring_args.output_nii), val_label.cpu(), rot180=True, sitkcast=True,
                  out_dir=inferring_args.output_dir)

        # inference
        val_outputs = sliding_window_inference(inputs=val_input.unsqueeze(0),
                                               promptargets=promptarget,
                                               roi_size=model_args.img_size,
                                               sw_batch_size=1,
                                               predictor=model,
                                               overlap=inferring_args.infer_overlap,
                                               )
        # print("val_input.shape", val_input.shape)#torch.Size([1, 1, 238, 190, 246])
        # print("val_outputs.shape", val_outputs.shape)#torch.Size([1, 1, 238, 190, 246])
        # end = time.time()
        # res.append(end-start)
        # shapes.append((val_outputs.shape[2],val_outputs.shape[3],val_outputs.shape[4]))
        # print("val_outputs.shape", val_outputs.shape)
        # print("torch.sum(val_outputs)", torch.sum(val_outputs))

        val_outputs_onehot = torch.where(torch.sigmoid(val_outputs) >= 0.5, 1.0, 0.0)
        # val_labels_onehot = val_labels.cpu()

        # print("torch.sum(val_outputs_onehot)", torch.sum(val_outputs_onehot))
        # print("torch.sum(val_label)", torch.sum(val_label))
        print("val_outputs_onehot.shape", val_outputs_onehot.shape)
        print("val_label.shape", val_label.shape)

        # save Nii
        saveToNii("predict_{}".format(inferring_args.output_nii), val_outputs_onehot[0, :].cpu(), rot180=True, sitkcast=False,
                  out_dir=inferring_args.output_dir)

        organ_Dice = acc_func(val_outputs_onehot, val_label.unsqueeze(0)).cpu().numpy()[0][0]
        print("organ_Dice", organ_Dice)

        with open(os.path.join(inferring_args.output_dir, inferring_args.output_nii[:-4]+'_'+str(organ_Dice) + 'SAM_' + '.txt'), "w", encoding='utf-8') as f:
            f.write(inferring_args.image_path)
            f.write('Dice')
            f.write(str(organ_Dice))
            f.close()
        print('organ_Dice', organ_Dice)

if __name__ == "__main__":
    main()
