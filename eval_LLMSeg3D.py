import os
from typing import Optional
import transformers
from dataclasses import dataclass, field
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
# from mindspore.nn.metrics import HausdorffDistance
from LaMed.src.utils.coefficient import iou_score, sensitivity, ppv
from monai.metrics import DiceMetric
from LaMed.src.utils.nii_utils import saveToNii
from LaMed.src.utils.slidingWindowInference import sliding_window_inference
import numpy as np
import pandas as pd
from LaMed.src.dataset.multi_dataset import SegDataset, RefSegDataset
from LaMed.src.model.language_model.phi3_Seg3D import Phi3_Seg3DForCausalLM

# from LaMed.src.dataset.dataset_info import dataset_info
# from torch.utils.data import ConcatDataset

local_rank = 0


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    resume_ckpt: str = field(default=None)
    model_name_or_path: Optional[str] = field(default="/yulong/pretrain/Phi-3-mini-4k-instruct",
                                              metadata={"help": "Path to the LLM or MLLM."})
    model_type: Optional[str] = field(default="phi3", metadata={"help": "llama3, phi3, Qwen2.5, llavaMed"})

    need_text_en: bool = field(default=False)  # if build TextEncoder
    test_mode: bool = field(default=False)

    freeze_backbone: bool = field(default=False)
    pretrain_mllm: Optional[str] = field(default=None)

    tune_mm_mlp_adapter: bool = field(default=False,
                                      metadata={"help": "Used in pretrain: tune mm_projector and embed_tokens"})
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None, metadata={
        "help": "Path to pretrained mm_projector and embed_tokens."})

    # image
    image_channel: int = field(default=1)
    img_size: tuple = field(default=(32, 256, 256))
    patch_size: tuple = field(default=(4, 16, 16))

    # projector
    mm_projector_type: Optional[str] = field(default='spp', metadata={"help": "spp"})
    proj_layer_type: str = field(default="mlp",
                                 metadata={"help": "Type of layer in projector. options: [linear, mlp]."})
    proj_layer_num: int = field(default=2, metadata={"help": "Number of layers in projector."})
    proj_pooling_type: str = field(default="spatial",
                                   metadata={"help": "Type of pooling in projector. options: [spatial, sequence]."})
    proj_pooling_size: int = field(default=2, metadata={"help": "Size of pooling in projector."})

    # vision
    vision_module: Optional[str] = field(default="vit3dseg")
    vision_select_layer: Optional[int] = field(default=-1)
    vision_select_feature: Optional[str] = field(default="patch")
    pretrain_vision_model: str = field(default="/yulong/pretrain/segvolClip/model.bin",
                                       metadata={"help": "Path to pretrained model for vision_model."})
    freeze_vision_tower: bool = field(default=True)

    # cross_seg
    # make_module: str = field(default="cross_seg", metadata={"help": "cross_seg"})
    # pretrain_seg_module: str = field(default=None, metadata={"help": "Pretrained seg model."})
    hidden_size: int = field(default=768)
    mlp_dim: int = field(default=3072)
    num_layers: int = field(default=12)
    num_heads: int = field(default=12)
    pos_embed: str = field(default="perceptron")
    dropout_rate: float = field(default=0.0)
    spatial_dims: int = field(default=3)
    num_clicks: int = field(default=0)


@dataclass
class DataArguments:
    description: bool = field(default=False)
    dataset_code: str = "0000"  # "RSeg" for CaptionRefSegDataset; "0000" for SegDataset
    max_length: int = field(default=512)
    data_root: str = field(default="/yulong/mllmDataset/", metadata={"help": "Root directory for all data."})

    # positioning & segmentation data
    seg_data_path: str = field(default="/yulong/mllmDataset/M3D-Seg/M3D_Seg_npy/",
                               metadata={"help": "Path to segmentation data."})
    refseg_data_path: str = field(default="/yulong/mllmDataset/M3D-RefSeg/M3D_RefSeg_npy/",
                                  metadata={"help": "Root directory for all data."})
    refseg_data_train_path: str = field(default="/yulong/mllmDataset/M3D-RefSeg/M3D_RefSeg_train.csv",
                                        metadata={"help": "Path to refering segmentation data."})
    refseg_data_test_path: str = field(default="/yulong/mllmDataset/M3D-RefSeg/M3D_RefSeg_test.csv",
                                       metadata={"help": "Path to refering segmentation data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # lora
    lora_enable: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=512,  # 512
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
    logging_steps: float = 10  # 0.001
    gradient_checkpointing: bool = False  # train fast
    dataloader_pin_memory: bool = True  # fast
    dataloader_num_workers: int = 0
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
        images, segs, impaths, questions, answers = tuple(
            [b[key] for b in batch] for key in ('image', 'seg', 'impath', 'question', 'answer'))

        images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)

        for i, seg in enumerate(segs):
            if seg.sum() == 0:
                segs[i] = torch.zeros((1, 1, 32, 256, 256))
            else:
                segs[i] = seg.unsqueeze(0)
        segs = torch.cat(segs, dim=0)

        return_dict = dict(
            images=images,
            segs=segs,
            impaths=impaths,
            questions=questions,
            answers=answers
        )
        return return_dict


def main():
    # gpu = 0
    # torch.cuda.set_device(gpu)
    # device = torch.device('cuda') # 'cpu', 'cuda'
    dtype = torch.bfloat16 # or bfloat16, float16, float32
    print("device_count：", torch.cuda.device_count())
    # # for single GPU
    # import torch.distributed as dist
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12345'
    # dist.init_process_group(backend='nccl', rank=0, world_size=1)
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    rank0_print("=" * 20 + " Tokenizer preparation " + "=" * 20)
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
    if 'llama3' in model_args.model_type:
        tokenizer.eos_token_id = 128001
        tokenizer.pad_token = tokenizer.eos_token

    # Convert special tokens to token IDs and set related arguments
    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    model_args.seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")
    model_args.vocab_size = len(tokenizer)
    rank0_print("seg_token_id: ", model_args.seg_token_id)
    rank0_print("vocab_size: ", model_args.vocab_size)

    if 'llm_phi3' in model_args.model_type:
        model = Phi3_Seg3DForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir
        )
    else:
        raise ValueError(f"Unknown Model Type {model_args.model_type}")

    # model.config.seg_token_id = model_args.seg_token_id
    # model.config.use_cache = False
    #
    # # initialize vision and seg modules on LLM
    # if model_args.vision_module is not None:
    #     model.get_model().initialize_vision_modules(model_args=model_args)
    #
    # model_args.num_new_tokens = 4
    # model.initialize_vision_tokenizer(model_args, tokenizer)

    # if model_args.pretrain_mllm:
    #     # ckpt = load_file(model_args.pretrain_mllm)
    #     ckpt = torch.load(model_args.pretrain_mllm, map_location='cpu')
    #     model.load_state_dict(ckpt, strict=True)
    #     rank0_print("load pretrained MLLM weights.")

    rank0_print(model)
    model.config.num_clicks = model_args.num_clicks
    model.config.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    rank0_print("=" * 20 + " Dataset preparation " + "=" * 20)
    data_args.max_length = training_args.model_max_length
    data_args.proj_out_num = model.get_model().mm_projector.proj_out_num
    rank0_print("vision tokens output from projector: ", data_args.proj_out_num)

    data_args.img_size = model_args.img_size
    # eval_dataset = MultiSegDataset(data_args, tokenizer, mode='test')
    # ds_list = []
    # for dataset_code in dataset_info.keys():
    #     ds_list.append(SegDataset(data_args, tokenizer, tag=dataset_code, description=data_args.description, mode='test'))
    # eval_dataset = ConcatDataset(ds_list)

    if "RSeg" in data_args.dataset_code:
        eval_dataset = RefSegDataset(args=data_args, tokenizer=tokenizer, mode='test')
    else:
        eval_dataset = SegDataset(args=data_args, tokenizer=tokenizer, tag=data_args.dataset_code,
                                  description=data_args.description, mode='test')

    data_collator = DataCollator()

    dataloader_params = {
        "batch_size": training_args.eval_batch_size,
        "collate_fn": data_collator,
        "num_workers": training_args.dataloader_num_workers,
        "pin_memory": training_args.dataloader_pin_memory,
        "persistent_workers": True,
    }

    val_loader = DataLoader(eval_dataset, **dataloader_params)

    model.test_mode = True
    model.eval()

    if torch.cuda.device_count() > 1:
        from accelerate import dispatch_model
        from accelerate.utils import infer_auto_device_map, get_balanced_memory
        device_map = infer_auto_device_map(model, max_memory=get_balanced_memory(model))
        print('multi GPU predict => {}'.format(device_map))
        model = dispatch_model(model, device_map)
    else:
        model = model.cuda()
        print("single GPU predict")

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
    col12 = []
    col13 = []
    col14 = []
    with torch.no_grad():
        # Hausdorff_metric = HausdorffDistance(percentile=95.0)
        acc_func = DiceMetric(include_background=False, reduction="mean", get_not_nans=True, num_classes=1, ignore_empty=False)
        # res = []
        # shapes = []
        # {
        #     'image': image,
        #     'input_id': input_id,
        #     'label': label,
        #     'seg': seg,
        #     'attention_mask': attention_mask,
        #     'question': question,
        #     'answer': answer,
        #     'question_type': "seg",
        # }
        for j, batch in enumerate(val_loader):
            val_input, val_label = (batch["images"].cuda(), batch["segs"].cuda())
            img_name = batch["impaths"]
            question = batch["questions"]
            answer = batch["answers"]
            question_tensor = tokenizer(question,
                                        return_tensors="pt")
            input_id = question_tensor["input_ids"].cuda()
            # attention_mask = question_tensor["attention_mask"].cuda()
            print("Inference on case {}".format(img_name))
            # print("question", question)
            print("answer", answer)
            # print("val_input shape", val_input.shape)
            # print("val_label shape", val_label.shape)
            # print("input_id shape", input_id.shape)
            # print("attention_mask shape", attention_mask.shape)

            # start = time.time()
            # save Nii
            # saveToNii("data_{}".format(img_name), val_input[0, :].cpu(), rot180=True, out_dir=args.niidir)
            # saveToNii("target_{}".format(img_name), val_label[0, :].cpu(), rot180=True, sitkcast=True, out_dir=args.niidir)
            # inference
            output_generations, val_outputs = sliding_window_inference(inputs=val_input,
                                                                       input_ids=input_id,
                                                                       # attention_mask=attention_mask,
                                                                       max_new_tokens=100,
                                                                       do_sample=True,
                                                                       top_p=0.9,
                                                                       temperature=1.0,
                                                                       roi_size=data_args.img_size,
                                                                       sw_batch_size=1,
                                                                       predictor=model,
                                                                       overlap=training_args.infer_overlap
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
            print("output_generations", len(output_generations))

            generated_texts = []
            # output_generation
            for op_id in output_generations:
                # print("output_generation op_id", op_id.shape)
                generated_text = tokenizer.batch_decode(op_id, skip_special_tokens=True)
                # print('++！！ generated_texts:', generated_text)
                generated_texts.append(generated_text[0])
            # save Nii
            # saveToNii("predict_{}".format(img_name), val_outputs_onehot[0, :], rot180=True, sitkcast=True, out_dir=training_args.output_dir)

            organ_Dice = acc_func(val_outputs_onehot, val_label).cpu().numpy()[0][0]
            print('organ_Dice', organ_Dice)

            # print('break!!!!!!!!!!!!!!!!!!!!!!!!!!!',organ_Dice)
            # break

            print('val_labels unique', torch.unique(val_label))
            val_o = val_outputs_onehot.cpu().numpy()
            val_l = val_label.cpu().numpy()

            val_o_sum = np.sum(val_o)
            val_l_sum = np.sum(val_l)
            print('val_l_sum', val_l_sum)

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
                col1.append(model_args.model_name_or_path)  # method
                col2.append(img_name)
                col3.append(coefficient[i])
                col4.append(val_outputs.shape)
                col5.append(question)
                if i == 0:
                    col6.append(organ_Dice)
                elif i == 1:
                    col6.append(organ_HD)
                elif i == 2:
                    col6.append(organ_IOU)
                elif i == 3:
                    col6.append(organ_precision)
                elif i == 4:
                    col6.append(organ_recall)
                col7.append(data_args.dataset_code)
                col8.append(val_o_sum)
                col9.append(val_l_sum)
                col10.append(model_args.num_clicks)
                col11.append(pn)
                col12.append(str(data_args.description))
                col13.append(answer)
                col14.append(str(generated_texts))
        csv_dict["method"] = col1
        csv_dict["num_clicks"] = col10
        csv_dict["dataset"] = col7
        csv_dict["NP"] = col11
        csv_dict["description"] = col12
        csv_dict["label_sum"] = col9
        csv_dict["out_sum"] = col8
        csv_dict["sampleName"] = col2
        csv_dict["coefficient"] = col3
        csv_dict["shape"] = col4
        csv_dict["question"] = col5
        csv_dict["answer"] = col13
        csv_dict["generated_texts"] = col14
        csv_dict["val"] = col6
        df = pd.DataFrame(csv_dict)
        # 保存 dataframe
        if not os.path.exists(training_args.output_dir):
            os.mkdir(training_args.output_dir)
        df.to_csv(os.path.join(training_args.output_dir, "Llamed_" + model_args.model_type + "_num_clicks" + str(
            model_args.num_clicks) + "_" + data_args.dataset_code + "_report.csv"))


if __name__ == "__main__":
    main()
