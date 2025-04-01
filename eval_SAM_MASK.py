import os
from typing import Optional
import transformers
from dataclasses import dataclass, field
from LaMed.src.dataset.seg_dataset import SegDatasets
import torch
from LaMed.src.model.r1Seg3DSAM_Config import R1Seg3DSAM_Config
from LaMed.src.model.build_vit3dseg import model_registry
from safetensors.torch import load_file
from torch.utils.data import DataLoader
# from mindspore.nn.metrics import HausdorffDistance
from LaMed.src.utils.coefficient import iou_score, sensitivity, ppv
from monai.metrics import DiceMetric
from LaMed.src.utils.nii_utils import saveToNii
from LaMed.src.utils.slidingWindowInference_SAM import sliding_window_inference
import numpy as np
import pandas as pd

local_rank = 0
def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    sam_bert_path: str = field(default="./LaMed/pretrained_model/SegVol/")
    need_text_en: bool = field(default=True)# of build TextEncoder
    test_mode: bool = field(default=False)
    train_clip: bool = field(default=False)
    resume_ckpt: str = field(default=None)

    pretrained_model: str = field(default=None)

    in_channels: int = field(default=1)
    out_channels: int = field(default=1)
    img_size: tuple = field(default=(32, 256, 256))
    patch_size: tuple = field(default=(4, 16, 16))

    hidden_size: int = field(default=768)
    mlp_dim: int = field(default=3072)
    num_layers: int = field(default=12)
    num_heads: int = field(default=12)
    pos_embed: str = field(default="perceptron")
    dropout_rate: float = field(default=0.0)
    spatial_dims: int = field(default=3)
    num_clicks: int = field(default=2)


@dataclass
class DataArguments:
    dataset_code: str = "0000" # "RSeg" for CaptionRefSegDataset; "0000" for SegDataset
    max_length: int = field(default=512)
    data_root: str = field(default="/yulong/mllmDataset/", metadata={"help": "Root directory for all data."})

    # caption data
    cap_data_path: str = field(default="/yulong/mllmDataset/M3D-Cap/M3D_Cap.json",
                               metadata={"help": "Path to caption data."})

    # positioning & segmentation data
    seg_data_path: str = field(default="/yulong/mllmDataset/M3D-Seg/M3D_Seg_npy/",
                               metadata={"help": "Path to segmentation data."})
    refseg_data_path: str = field(default="/yulong/mllmDataset/M3D-RefSeg/M3D_RefSeg_npy/", metadata={"help": "Root directory for all data."})
    refseg_data_train_path: str = field(default="/yulong/mllmDataset/M3D-RefSeg/M3D_RefSeg_train.csv", metadata={"help": "Path to refering segmentation data."})
    refseg_data_test_path: str = field(default="/yulong/mllmDataset/M3D-RefSeg/M3D_RefSeg_test.csv", metadata={"help": "Path to refering segmentation data."})



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)

    ddp_backend: str = "nccl"
    ddp_find_unused_parameters: bool = False

    # config in bash file
    bf16: bool = True
    output_dir: str = "./LaMed/output/CLIP"
    num_train_epochs: int = 100
    per_device_train_batch_size: int = 32 #32
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    evaluation_strategy: str = "steps"
    eval_accumulation_steps: int = 1
    eval_steps: float = 0.04 # 0.04
    save_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit: int = 1
    learning_rate: float = 1e-4 #1e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: float = 0.001 # 0.001
    gradient_checkpointing: bool = False # train fast
    dataloader_pin_memory: bool = True # fast
    dataloader_num_workers: int = 8
    report_to: str = "tensorboard"
    infer_overlap: float = 0.5

    # gradient_clipping: float = 1.0 # 梯度截断阈值 only for step 2

def compute_metrics(eval_pred):
    predict = eval_pred.predictions
    target = eval_pred.label_ids

    predict = torch.sigmoid(predict)
    target_ = target.clone().float()
    target_[target == -1] = 0
    assert predict.shape[0] == target.shape[0], "predict & target batch size don't match\n" + str(
        predict.shape) + '\n' + str(target.shape[0])
    predict = predict.contiguous().view(predict.shape[0], -1)
    target_ = target_.contiguous().view(target_.shape[0], -1)

    num = torch.sum(torch.mul(predict, target_), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target_, dim=1) + 1

    dice_score = 2 * num / den

    dice_score = dice_score.sum() / dice_score.shape[0]
    return {"dice_score": dice_score}

def preprocess_logits_for_metrics(logits, labels):
    preds = torch.argmax(logits, dim=-1)
    return preds

@dataclass
class DataCollator:
    def __init__(self):
        super().__init__()

    def __call__(self, batch: list) -> dict:
        images, impaths, promptargets, segs = tuple(
                [b[key] for b in batch] for key in ('image', 'impath', 'promptarget', 'seg'))
        images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)

        batch_size = images.shape[0]
        labels = torch.arange(batch_size, device=images.device, dtype=torch.long)

        for i, seg in enumerate(segs):
            if seg is None or seg.sum() == 0:
                segs[i] = torch.zeros((1, 1, 32, 256, 256))
            else:
                segs[i] = seg.unsqueeze(0)
        segs = torch.cat(segs, dim=0)

        return_dict = dict(
            images=images,
            impaths=impaths,
            promptargets=promptargets,
            segs=segs,
            # input_ids=input_ids,
            # attention_mask=attention_mask,
            labels=labels,
        )

        return return_dict


def main():
    gpu = 0
    torch.cuda.set_device(gpu)
    # # for single GPU
    # import torch.distributed as dist
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12345'
    # dist.init_process_group(backend='nccl', rank=0, world_size=1)
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # model_args.max_length = data_args.max_length
    config = R1Seg3DSAM_Config.from_dict(vars(model_args))
    config.mm_hidden_size = config.hidden_size
    model = model_registry['vit'](config=config, checkpoint=None)   # checkpoint for pretrained ViT

    if model_args.pretrained_model is not None:
        ckpt = load_file(model_args.pretrained_model)
        model.load_state_dict(ckpt, strict=False)
        # rank0_print("pretrained_model >>>>>>>>>>>>>", ckpt.keys())
        rank0_print(">>>>>>>>>>>>>load pretrained model>>>>>>>>>>>>>")

    data_args.img_size = model_args.img_size
    # eval_dataset = SegDatasets(data_args, mode='test')

    eval_dataset = SegDatasets(data_args, tag=data_args.dataset_code, mode='test')# one dataset for eval

    data_collator = DataCollator()
    dataloader_params = {
        "batch_size": training_args.eval_batch_size,
        "collate_fn": data_collator,
        "num_workers": training_args.dataloader_num_workers,
        "pin_memory": training_args.dataloader_pin_memory,
        "persistent_workers": True,
    }

    val_loader = DataLoader(eval_dataset, **dataloader_params)

    model = torch.nn.DataParallel(model, device_ids=[gpu])
    model.eval()

    # csv for report
    coefficient = ["dice", "HD95", "IOU", "precision", "recall"]
    csv_dict = {}
    col1 = []
    col2 = []
    col3 = []
    col4 = []
    col5 = []
    col6 = []
    col7 = []  # dataset
    col8 = []
    col9 = []
    col10 = []
    col11 = []
    with torch.no_grad():
        # Hausdorff_metric = HausdorffDistance(percentile=95.0)
        acc_func = DiceMetric(include_background=False, reduction="mean", get_not_nans=True, num_classes=1)
        # res = []
        # shapes = []
        for j, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["images"].cuda(), batch["segs"].cuda())
            img_name = batch["impaths"]
            promptarget = batch["promptargets"]
            print("Inference on case {}".format(img_name))
            print("promptarget", promptarget)
            print("val_inputs shape", val_inputs[0].shape)

            # start = time.time()
            # save Nii
            # saveToNii("data_{}".format(img_name), val_inputs[0, :].cpu(), rot180=True, out_dir=args.niidir)
            # saveToNii("target_{}".format(img_name), val_labels[0, :].cpu(), rot180=True, sitkcast=True, out_dir=args.niidir)
            #inference
            val_outputs = sliding_window_inference(inputs=val_inputs,#[val_inputs, promptarget],
                                                   promptargets=promptarget,
                                                   roi_size=data_args.img_size,
                                                   sw_batch_size=1,
                                                   predictor=model,
                                                   overlap=training_args.infer_overlap,
                                                   )
            # print("val_inputs.shape", val_inputs.shape)#torch.Size([1, 1, 238, 190, 246])
            # print("val_outputs.shape", val_outputs.shape)#torch.Size([1, 1, 238, 190, 246])
            # end = time.time()
            # res.append(end-start)
            # shapes.append((val_outputs.shape[2],val_outputs.shape[3],val_outputs.shape[4]))
            print("val_outputs.shape", val_outputs.shape)

            val_outputs_onehot = torch.where(torch.sigmoid(val_outputs) >= 0.5, 1.0, 0.0)
            # val_labels_onehot = val_labels.cpu()

            # save Nii
            # saveToNii("predict_{}".format(img_name), val_outputs_onehot[0, :], rot180=True, sitkcast=True, out_dir=training_args.output_dir)

            organ_Dice = acc_func(val_outputs_onehot, val_labels).cpu().numpy()[0][0]
            print('organ_Dice',organ_Dice)
            val_o = val_outputs_onehot.cpu().numpy()
            val_l = val_labels.cpu().numpy()

            val_o_sum = np.sum(val_o)
            val_l_sum = np.sum(val_l)

            pn = 1
            if val_l_sum > 0:
                # Hausdorff_metric.clear()
                # Hausdorff_metric.update(val_o, val_l, 1)
                # organ_HD = Hausdorff_metric.eval()
                organ_HD = 1
                organ_IOU = iou_score(val_o, val_l)
                organ_precision = ppv(val_o, val_l)
                organ_recall = sensitivity(val_o, val_l)
            else:
                pn = 0
                organ_Dice = np.sum(val_o)
                organ_HD = 500
                organ_IOU = 0.0
                organ_precision = 0.0
                organ_recall = 0.0

            # coefficient = ["dice", "HD95", "IOU", "precision", "recall"]
            for i in range(5):  # five coefficient
                col1.append(model_args.pretrained_model)  # method
                col2.append(img_name)
                col3.append(coefficient[i])
                col4.append(val_outputs.shape)
                col5.append(promptarget)
                if i==0:
                    col6.append(organ_Dice)
                elif i==1:
                    col6.append(organ_HD)
                elif i==2:
                    col6.append(organ_IOU)
                elif i==3:
                    col6.append(organ_precision)
                elif i==4:
                    col6.append(organ_recall)
                col7.append(data_args.dataset_code)
                col8.append(val_o_sum)
                col9.append(val_l_sum)
                col10.append(model_args.num_clicks)
                col11.append(pn)
        csv_dict["method"] = col1
        csv_dict["num_clicks"] = col10
        csv_dict["dataset"] = col7
        csv_dict["NP"] = col11
        csv_dict["label_sum"] = col9
        csv_dict["out_sum"] = col8
        csv_dict["sampleName"] = col2
        csv_dict["coefficient"] = col3
        csv_dict["shape"] = col4
        csv_dict["organName"] = col5
        csv_dict["val"] = col6
        df = pd.DataFrame(csv_dict)
        # 保存 dataframe
        df.to_csv(os.path.join(training_args.output_dir, "num_clicks"+str(model_args.num_clicks)+"_"+data_args.dataset_code+"_report.csv"))


if __name__ == "__main__":
    main()
