import sys
import os
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import logging
from typing import Optional
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer
from dataclasses import dataclass, field
from LaMed.src.dataset.llm_seg_dataset import MultiSegDataset
from LaMed.src.model.language_model.phi3_Seg3D import Phi3_Seg3DForCausalLM
from LaMed.src.train.lamed_trainer import LaMedTrainer
local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    resume_ckpt: str = field(default=None)
    model_name_or_path: Optional[str] = field(default="../pretrain/Phi-3-mini-4k-instruct", metadata={"help": "Path to the LLM or MLLM."})
    model_type: Optional[str] = field(default="phi3", metadata={"help": "llama3, phi3, Qwen2.5, llavaMed1.5"})

    need_text_en: bool = field(default=False)# if build TextEncoder
    test_mode: bool = field(default=False)

    freeze_backbone: bool = field(default=False)
    pretrain_mllm: Optional[str] = field(default=None)
    # tune_mm_mlp_adapter: bool = field(default=False, metadata={"help": "Used in pretrain: tune mm_projector and embed_tokens"})

    # image
    image_channel: int = field(default=1)
    img_size: tuple = field(default=(32, 256, 256))
    patch_size: tuple = field(default=(4, 16, 16))

    # vision
    vision_module: Optional[str] = field(default="vit3dseg")
    vision_select_layer: Optional[int] = field(default=-1)
    vision_select_feature: Optional[str] = field(default="patch")
    pretrain_vision_model: str = field(default="/yulong/pretrain/segvolClip/model.bin", metadata={"help": "Path to pretrained model for vision_model."})
    freeze_vision_tower: bool = field(default=True)

    # cross_seg
    make_module: str = field(default="cross_seg", metadata={"help": "cross_seg"})
    pretrain_seg_module: str = field(default=None, metadata={"help": "Pretrained seg model."})
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
    seg_enable: bool = field(default=True)
    data_root: str = field(default="/yulong/mllmDataset/", metadata={"help": "Root directory for all data."})

    # positioning & segmentation data
    seg_data_path: str = field(default="/yulong/mllmDataset/M3D-Seg/M3D_Seg_npy/", metadata={"help": "Path to segmentation data."})
    refseg_data_path: str = field(default="/yulong/mllmDataset/M3D-RefSeg/M3D_RefSeg_npy/", metadata={"help": "Root directory for all data."})
    refseg_data_train_path: str = field(default="/yulong/mllmDataset/M3D-RefSeg/M3D_RefSeg_train.csv", metadata={"help": "Path to refering segmentation data."})
    refseg_data_test_path: str = field(default="/yulong/mllmDataset/M3D-RefSeg/M3D_RefSeg_test.csv", metadata={"help": "Path to refering segmentation data."})



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # tune_vision_module
    tune_vision_module: bool = False
    # lora
    lora_enable: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=512, #512
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    seed: int = 42
    ddp_backend: str = "nccl"
    ddp_timeout: int = 1280000
    ddp_find_unused_parameters: bool = False
    optim: str = field(default="adamw_torch")

    # This is set up to facilitate debugging, pls config these in bash file in training.
    bf16: bool = True
    output_dir: str = "./LaMed/output/LaMed-pretrain-test"
    num_train_epochs: float = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    evaluation_strategy: str = "steps"
    eval_accumulation_steps: int = 1
    eval_steps: float = 0.04
    save_strategy: str = "steps"
    save_steps: int = 2000
    save_total_limit: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: float = 10 # 0.001
    gradient_checkpointing: bool = False # train fast
    dataloader_pin_memory: bool = True # fast
    dataloader_num_workers: int = 0
    report_to: str = "tensorboard"


def compute_metrics(eval_preds):
    labels_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions

    labels = labels_ids[:, 1:]
    preds = pred_ids[:, :-1]

    labels_flatten = labels.reshape(-1)
    preds_flatten = preds.reshape(-1)
    valid_indices = np.where(labels_flatten != -100)
    filtered_preds = preds_flatten[valid_indices]
    filtered_labels = labels_flatten[valid_indices]
    acc_score = sum(filtered_preds==filtered_labels) / len(filtered_labels)
    return {"accuracy": acc_score}

# def compute_metrics(pred):
#     # 解包标签和额外的数据
#     labels, additional_data = pred.label_ids
#     # 获取模型预测
#     preds = pred.predictions.argmax(-1)
#     # 计算精确度、召回率、F1分数和准确率
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
#     acc = accuracy_score(labels, preds)
#     # 返回计算的评价指标
#     return {
#     'accuracy': acc,
#     'f1': f1,
#     'precision': precision,
#     'recall': recall
#     }

def preprocess_logits_for_metrics(logits, labels):

    print("++logits", logits.shape)
    print("++labels", labels.shape)
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_mm_projector_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    # Process of elimination: LoRA only targets on LLM backbone
    ignore_keywords = ['vision_module', 'embed_tokens', 'lm_head', 'seg_projector']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in ignore_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    return list(lora_module_names)

@dataclass
class DataCollator:
    def __init__(self, seg_enable):
        self.seg_enable = seg_enable
    def __call__(self, batch: list) -> dict:
        if self.seg_enable:
            images, input_ids, labels, attention_mask, segs = tuple(
                [b[key] for b in batch] for key in ('image', 'input_id', 'label', 'attention_mask', 'seg'))

            images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
            input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
            labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
            attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)

            for i, seg in enumerate(segs):
                if seg.sum() == 0:
                    segs[i] = torch.zeros((1, 1, 32, 256, 256))
                else:
                    segs[i] = seg.unsqueeze(0)
            segs = torch.cat(segs, dim=0)

            return_dict = dict(
                images=images,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                segs=segs,
            )
        else:
            images, input_ids, labels, attention_mask = tuple(
                [b[key] for b in batch] for key in ('image', 'input_id', 'label', 'attention_mask'))

            images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
            input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
            labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
            attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)

            return_dict = dict(
                images=images,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )

        return return_dict


def main():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank

    rank0_print("="*20 + " Tokenizer preparation " + "="*20)
    # Load tokenizer from the given path with specified configurations
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Define and add special tokens
    special_token = {"additional_special_tokens": ["<im_patch>", "<bx_start>", "<bx_end>"]}
    tokenizer.add_special_tokens(
        special_token
    )
    tokenizer.add_tokens("[SEG]")

    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    # Convert special tokens to token IDs and set related arguments
    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    model_args.seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")
    model_args.vocab_size = len(tokenizer)
    rank0_print("seg_token_id: ", model_args.seg_token_id)
    rank0_print("vocab_size: ", model_args.vocab_size)

    rank0_print("="*20 + " Model preparation " + "="*20)
    if model_args.vision_module is not None and model_args.model_name_or_path is not None:
        if 'phi3' in model_args.model_type:
            model = Phi3_Seg3DForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir
            )
        else:
            raise ValueError(f"Unknown Model Type {model_args.model_type}")

    model.config.seg_token_id = model_args.seg_token_id
    model.config.use_cache = False
    model.config.vision_module = model_args.vision_module
    model.config.need_text_en = model_args.need_text_en
    model.test_mode = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    model.enable_input_require_grads()
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # initialize vision module
    if model_args.vision_module is not None:
        model.get_model().initialize_vision_modules(model_args=model_args)

    model_args.num_new_tokens = 4
    model.initialize_vision_tokenizer(model_args, tokenizer)


    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        rank0_print("Adding LoRA adapters only on LLM.")
        model = get_peft_model(model, lora_config)

        for n, p in model.named_parameters():

            if any(
                    [x in n for x in ['embed_tokens', 'lm_head', 'seg_projector']]
            ):
                p.requires_grad = True

            if any(
                    [x in n for x in ['vision_module']]# finetune mask
            ):
                if training_args.tune_vision_module:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

            if any(
                    [x in n for x in ['vision_module.image_encoder']]
            ):
                p.requires_grad = False

            rank0_print(n, p.requires_grad)

    for n, p in model.named_parameters():
        if p.requires_grad:
            rank0_print(n, p.requires_grad)

    rank0_print(model)

    if model_args.pretrain_mllm:
        # ckpt = load_file(model_args.pretrain_mllm)
        ckpt = torch.load(model_args.pretrain_mllm, map_location='cpu')
        model.load_state_dict(ckpt, strict=False)
        rank0_print(">>>>>>>>> load pretrained pretrain_mllm weights.")

    rank0_print("="*20 + " Dataset preparation " + "="*20)
    data_args.max_length = training_args.model_max_length
    data_args.img_size = model_args.img_size
    train_dataset = MultiSegDataset(data_args, tokenizer, mode='train')

    eval_dataset = MultiSegDataset(data_args, tokenizer, mode='validation')
    data_collator = DataCollator(data_args.seg_enable)

    rank0_print("="*20 + " Training " + "="*20)
    trainer = LaMedTrainer(
                            model=model,
                            args=training_args,
                            data_collator=data_collator,
                            train_dataset=train_dataset,
                            eval_dataset=eval_dataset,
                            compute_metrics=compute_metrics,
                            # preprocess_logits_for_metrics=preprocess_logits_for_metrics
                      )

    # if you want to resume your training, pls set the checkpoint in trainer.train(resume_from_checkpoint="")
    if model_args.resume_ckpt is not None:
        trainer.train(resume_from_checkpoint=model_args.resume_ckpt)
    trainer.train()
    trainer.save_state()
    model.config.use_cache = True

    rank0_print("="*20 + " Save model " + "="*20)
    if training_args.lora_enable:
        state_dict_with_lora = model.state_dict()
        torch.save(state_dict_with_lora, os.path.join(training_args.output_dir, 'model_with_lora.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    main()
