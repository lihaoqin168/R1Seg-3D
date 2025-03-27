import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np
import torch
from dataclasses import dataclass, field
import simple_slice_viewer as ssv
import SimpleITK as sikt
from LaMed.src.model.language_model import LamedLlamaForCausalLM, LamedPhi3ForCausalLM

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

@dataclass
class AllArguments:
    model_name_or_path: str = field(default="/107556523204/output/M3D-ViT-RSeg/LaMed/output_model/LaMed-Phi3-4B-X256")

    proj_out_num: int = field(default=256, metadata={"help": "Number of output tokens in Projector."})
    image_path: str = field(default="/107556523204/haoqin/code/M3D-ViT-RSeg/Data/data/examples/example_04.npy")


def main():
    seed_everything(42)
    device = torch.device('cuda') # 'cpu', 'cuda'
    dtype = torch.float16 # or bfloat16, float16, float32

    parser = transformers.HfArgumentParser(AllArguments)
    args = parser.parse_args_into_dataclasses()[0]

    model = LamedPhi3ForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map='auto',
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    print(model)
    model = model.to(device=device)

    # question = "Can you provide a caption consists of findings for this medical image?"
    question = "What are liver and kidney in this image? Please output the segmentation mask."
    # question = "What is liver in this image? Please output the box."

    image_tokens = "<im_patch>" * args.proj_out_num
    input_txt = image_tokens + question
    input_id = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device=device)

    image_np = np.load(args.image_path)
    image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=dtype, device=device)

    # generation = model.generate(image_pt, input_id, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)
    generation, seg_logit = model.generate(image_pt, input_id, seg_enable=True, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)

    generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
    seg_mask = (torch.sigmoid(seg_logit) > 0.5) * 1.0

    print('question:', question)
    print('generated_texts:', generated_texts[0])

    image = image_np[0]
    slice = image.shape[0]
    print('seg_logit:', seg_mask.shape)
    for i in range(slice):
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(image[i], cmap='gray')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(seg_mask[0][0][i].cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.show()

    # image = sikt.GetImageFromArray(image_np)
    # ssv.display(image)
    #
    # seg = sikt.GetImageFromArray(seg_mask.cpu().numpy()[0])
    # ssv.display(seg)

if __name__ == "__main__":
    main()
