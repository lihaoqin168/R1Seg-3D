import os
from typing import Optional
import transformers
from transformers import Trainer
from dataclasses import dataclass, field
from LaMed.src.dataset.seg_dataset import SegDatasets
import torch
from LaMed.src.model.r1Seg3DSAM_Config import R1Seg3DSAM_Config
from LaMed.src.model.build_vit3dseg import model_registry

local_rank = 0
def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    language_model_name_or_path: str = field(default="./LaMed/pretrained_model/bert_base_uncased/")
    sam_bert_path: str = field(default="./LaMed/pretrained_model/SegVol/")
    need_text_en: bool = field(default=True)# of build TextEncoder
    gather_loss: bool = field(default=True, metadata={"help": "Gather all distributed batch data of multiple GPUs and calculate contrastive loss together."})
    local_loss: bool = field(default=False)
    test_mode: bool = field(default=False)
    resume_ckpt: str = field(default=None)
    pretrained_model: str = field(default=None)
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
    num_clicks: int = field(default=2)

@dataclass
class DataArguments:
    max_length: int = field(default=512)
    data_root: str = field(default="./mllmDataset/", metadata={"help": "Root directory for all data."})
    # caption data
    cap_data_path: str = field(default="./mllmDataset/M3D-Cap/M3D_Cap.json",
                               metadata={"help": "Path to caption data."})
    # positioning & segmentation data
    seg_data_path: str = field(default="./mllmDataset/M3D-Seg/M3D_Seg_npy/",
                               metadata={"help": "Path to segmentation data."})
    refseg_data_path: str = field(default="./mllmDataset/M3D-RefSeg/M3D_RefSeg_npy/", metadata={"help": "Root directory for all data."})
    refseg_data_train_path: str = field(default="./mllmDataset/M3D-RefSeg/M3D_RefSeg_train.csv", metadata={"help": "Path to refering segmentation data."})
    refseg_data_test_path: str = field(default="./mllmDataset/M3D-RefSeg/M3D_RefSeg_test.csv", metadata={"help": "Path to refering segmentation data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    ddp_backend: str = "nccl"
    ddp_find_unused_parameters: bool = False
    # config in bash file
    bf16: bool = True
    output_dir: str = "./LaMed/output/R1Seg-3DSAM-step1"
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
    def __init__(self, gather_all):
        self.gather_all = gather_all

    def __call__(self, batch: list) -> dict:
        images, impaths, promptargets, segs = tuple(
                [b[key] for b in batch] for key in ('image', 'impath', 'promptarget', 'seg'))
        images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
        batch_size = images.shape[0]
        if self.gather_all:
            world_size = torch.distributed.get_world_size()
            batch_size *= world_size

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
            labels=labels,
        )

        return return_dict


def main():

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    config = R1Seg3DSAM_Config.from_dict(vars(model_args))
    config.mm_hidden_size = model_args.hidden_size
    model = model_registry['vit'](config=config, checkpoint=model_args.pretrained_model)   # checkpoint for pretrained ViT

    # freeze prompt text_encoder!!
    for param in model.text_encoder.clip_text_model.parameters():
        param.requires_grad = False

    for n, p in model.named_parameters():
        if not p.requires_grad:
            rank0_print(n, p.requires_grad)
    rank0_print(model)

    data_args.img_size = model_args.img_size
    train_dataset = SegDatasets(data_args, mode='train')
    eval_dataset = SegDatasets(data_args, mode='validation')

    if model_args.gather_loss and not model_args.local_loss:
        gather_all = True
    else:
        gather_all = False
    data_collator = DataCollator(gather_all)

    trainer = Trainer(
                        model=model,
                        args=training_args,
                        data_collator=data_collator,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        compute_metrics=compute_metrics,
                        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                      )

    # if you want to resume your training, pls set the checkpoint in trainer.train(resume_from_checkpoint="")
    if model_args.resume_ckpt is not None:
        trainer.train(resume_from_checkpoint=model_args.resume_ckpt)
    trainer.train()

    trainer.save_state()
    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(training_args.output_dir, 'r1seg-3dsam.bin'))

    state_dict = model.image_encoder.state_dict()
    torch.save(state_dict, os.path.join(training_args.output_dir, 'seg_vit.bin'))


if __name__ == "__main__":
    main()
