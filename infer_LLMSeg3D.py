import os
import random
from typing import Optional
import transformers
from dataclasses import dataclass, field
import torch
from transformers import AutoTokenizer
from monai.metrics import DiceMetric
from LaMed.src.utils.nii_utils import saveToNii
from LaMed.src.utils.slidingWindowInference import sliding_window_inference
import numpy as np
from LaMed.src.model.language_model.phi3_Seg3D import Phi3_Seg3DForCausalLM
import monai.transforms as mtf
from LaMed.src.dataset.dataset_info import dataset_info
from LaMed.src.dataset.prompt_templates import Seg_templates
from LaMed.src.dataset.term_dictionary import term_dict

local_rank = 0
def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    model_name_or_path: Optional[str] = field(default="/",
                                              metadata={"help": "Path to the model weight."})
    model_type: Optional[str] = field(default="phi3", metadata={"help": "llama3, phi3, Qwen2.5, llavaMed"})
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
    description: bool = field(default=True)
    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=512,  # 512
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
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

    rank0_print("=" * 20 + " Tokenizer preparation " + "=" * 20)
    # Load tokenizer from the given path with specified configurations
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=inferring_args.cache_dir,
        model_max_length=inferring_args.model_max_length,
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

    if 'phi3' in model_args.model_type:
        model = Phi3_Seg3DForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=inferring_args.cache_dir
        )
    else:
        raise ValueError(f"Unknown Model Type {model_args.model_type}")

    # rank0_print(model)
    model.config.num_clicks = model_args.num_clicks
    model.config.tune_mm_mlp_adapter = False
    rank0_print("=" * 20 + " Dataset preparation " + "=" * 20)

    val_transform = mtf.Compose(
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
    cls_questions = Seg_templates["cls_questions"]
    des_questions = Seg_templates["des_questions"]
    cls_answers = Seg_templates["cls_answers"]
    des_answers = Seg_templates["des_answers"]
    if not inferring_args.description:
        question_temple = random.choice(cls_questions)
        question = question_temple.format(cls_list[cls_id])
        answer = random.choice(cls_answers)
    else:
        question_temple = random.choice(des_questions)
        question = question_temple.format(random.choice(term_dict[cls_list[cls_id]]))
        answer = random.choice(des_answers).format(cls_list[cls_id])

    text_tensor = tokenizer(
        question + ' ' + answer, max_length=inferring_args.model_max_length, truncation=True, padding="max_length",
        return_tensors="pt"
    )

    input_id = text_tensor["input_ids"][0]
    attention_mask = text_tensor["attention_mask"][0]

    valid_len = torch.sum(attention_mask)
    if valid_len < len(input_id):
        input_id[valid_len] = tokenizer.eos_token_id

    question_tensor = tokenizer(
        question, max_length=inferring_args.model_max_length, truncation=True, padding="max_length", return_tensors="pt"
    )
    question_len = torch.sum(question_tensor["attention_mask"][0])

    label = input_id.clone()
    label[:question_len] = -100
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        label[label == tokenizer.pad_token_id] = -100
        if valid_len < len(label):
            label[valid_len] = tokenizer.eos_token_id
    else:
        label[label == tokenizer.pad_token_id] = -100

    #inferring
    model.eval()
    model = model.cuda()

    with torch.no_grad():
        # Hausdorff_metric = HausdorffDistance(percentile=95.0)
        acc_func = DiceMetric(include_background=False, reduction="mean", get_not_nans=True, num_classes=1, ignore_empty=False)
        val_input, val_label = (image.cuda(), seg.cuda())
        question_tensor = tokenizer(question, return_tensors="pt")
        input_id = question_tensor["input_ids"].cuda()
        # attention_mask = question_tensor["attention_mask"].cuda()
        print("Inference on case {}".format(inferring_args.image_path))
        # print("question", question)
        print("answer", answer)
        print("val_input shape", val_input.shape)
        print("val_label shape", val_label.shape)
        # print("input_id shape", input_id.shape)
        # print("attention_mask shape", attention_mask.shape)

        # start = time.time()
        # save Nii
        # saveToNii("data_{}".format(inferring_args.output_nii), val_input.cpu(), rot180=True, out_dir=inferring_args.output_dir)
        # saveToNii("target_{}".format(inferring_args.output_nii), val_label.cpu(), rot180=True, sitkcast=True, out_dir=inferring_args.output_dir)
        # inference
        output_generations, val_outputs = sliding_window_inference(inputs=val_input.unsqueeze(0),
                                                                   input_ids=input_id,
                                                                   max_new_tokens=100,
                                                                   do_sample=True,
                                                                   top_p=0.9,
                                                                   temperature=1.0,
                                                                   roi_size=model_args.img_size,
                                                                   sw_batch_size=1,
                                                                   predictor=model,
                                                                   overlap=inferring_args.infer_overlap
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
        print("val_outputs_onehot.shape", val_outputs_onehot.shape)
        saveToNii("predict_{}".format(inferring_args.output_nii), val_outputs_onehot[0, :].cpu(), rot180=True, sitkcast=False,
                  out_dir=inferring_args.output_dir)

        organ_Dice = acc_func(val_outputs_onehot, val_label.unsqueeze(0)).cpu().numpy()[0][0]
        print("organ_Dice", organ_Dice)

        with open(os.path.join(inferring_args.output_dir, inferring_args.output_nii[:-4]+'_'+str(organ_Dice) + 'rSeg_' + '.txt'), "w", encoding='utf-8') as f:
            f.write(inferring_args.image_path)
            f.write('Dice')
            f.write(str(organ_Dice))
            f.write('question /t/r')
            f.write(question)
            f.write('answer /t/r')
            f.write(answer)
            f.write('generated_texts /t/r')
            f.write(str(generated_texts))
            f.close()
        print('organ_Dice', organ_Dice)

if __name__ == "__main__":
    main()
