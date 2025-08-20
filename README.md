# R1Seg-3D: Rethinking Reasoning Segmentation for Medical 3D CTs

[Paper]:https://
[Model]:https://huggingface.co/lihao0011/R1Seg-3D-Phi-3-4B
<font size=3><div align='center' > <a href=https://arxiv.org/abs/2404.00578>**Paper**</a> | [**Data**][data] | [**Model**][model] | [**Training**](#training) | [**Benchmark**](#benchmark) | [**Online Demo**][demo]</div></font>
The explosive development of large-scale model technology has provided strong support for achieving more intelligent, robust, 
and precise segmentation techniques. However, owing to the unique challenges posed by medical domain data, 
the typical 3D medical image-text alignment model, 3D CLIP, struggles to match the performance of its natural scene counterpart. 
This limitation hinders the application of CLIP-based text-image reasoning in medical segmentation tasks. 
Furthermore, CLIP has been shown to rely on high-level semantic alignment between vision and text, 
lacking effective support for local visual features that are crucial for dense prediction tasks. 
Existing reasoning segmentation methods often adopt a redundant design with two visual encoders—one from CLIP and the other from large vision models for downstream dense tasks. 
This adversely affects model efficiency and complicates the training process. 
To address these challenges, we propose a novel framework, R1Seg-3D, which unifies a visual encoder. 
Our approach achieves a three-way alignment of dense visual, text reasoning, and mask decoding features within a shared latent space. 
Compared with previous methods, R1Seg-3D implicitly incorporates more detailed spatial features into the reasoning path. 
Therefore, it can strengthen the reasoning ability by incorporating additional visual spatial details and directly enhances the 
mask decoding process. The R1Seg-3D architecture is more concise and easier to be trained.

## Quickstart
Here, we can easily use our model based on Hugging Face.

```python
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import simple_slice_viewer as ssv
import SimpleITK as sikt

device = torch.device('cuda') # 'cpu', 'cuda'
dtype = torch.bfloat16 # or bfloat16, float16, float32

model_name_or_path = 'lihao0011/R1Seg-3D-Phi-3-4B'
proj_out_num = 256

# Prepare your 3D medical image:
# 1. The image shape needs to be processed as 1*32*256*256, consider resize and other methods.
# 2. The image needs to be normalized to 0-1, consider Min-Max Normalization.
# 3. The image format needs to be converted to .npy 
# 4. Although we did not train on 2D images, in theory, the 2D image can be interpolated to the shape of 1*32*256*256 for input.
image_path = "./Data/data/examples/example_03.npy"

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=dtype,
    device_map='auto',
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    model_max_length=512,
    padding_side="right",
    use_fast=False,
    trust_remote_code=True
)

model = model.to(device=device)

# question = "Can you provide a caption consists of findings for this medical image?"
question = "What is liver in this image? Please output the segmentation mask."
# question = "What is liver in this image? Please output the box."

image_tokens = "<im_patch>" * proj_out_num
input_txt = image_tokens + question
input_id = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device=device)

image_np = np.load(image_path)
image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=dtype, device=device)

# generation = model.generate(image_pt, input_id, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)
generation, seg_logit = model.generate(image_pt, input_id, seg_enable=True, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)

generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
seg_mask = (torch.sigmoid(seg_logit) > 0.5) * 1.0

print('question', question)
print('generated_texts', generated_texts[0])

image = sikt.GetImageFromArray(image_np)
ssv.display(image)
seg = sikt.GetImageFromArray(seg_mask.cpu().numpy()[0])
ssv.display(seg)
```

## Model
| Model    | Download Link                                                                                                                                 |
|----------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| R1Seg-3D-SAM | [HuggingFace](https://huggingface.co/lihao0011/R1Seg-3D-SAM)   |
| R1Seg-3D-Phi-3-4B  | [HuggingFace](https://huggingface.co/lihao0011/R1Seg-3D-Phi-3-4B)|
## Data

| Dataset  | Type | Images | Texts | Download Link |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| M3D-Seg | 3D images, category text, and segmentation masks | 5,772 | 149,196 | [HuggingFace](https://huggingface.co/datasets/GoodBaiBai88/M3D-Seg), [ModelScope](https://www.modelscope.cn/datasets/GoodBaiBai88/M3D-Seg) |

#### LLM
Phi-3-4B: Download and follow [here](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct).
Llama-2-7B: Download and follow [here](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).
Llama-2-7B: Download and follow [here](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).
Llama-2-7B: Download and follow [here](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).

#### Configuration
We suggest using `accelerate` to train. It was developed by Hugging Face 
and conveniently supports common training strategies such as distributed training, mixed precision, DeepSpeed, etc.
It should be configured on first use:
```bash
accelerate config
```
Please follow the configuration guide and we can choose the appropriate training strategy. 
We recommend using bf16 and Deepspeed for acceleration, and the ZeRO type depends on your own situation.

If you don't know how to configure it, we provide a simple configuration `default_config.yaml` for your reference.
<details>
<summary>default_config.yaml</summary>

```bash
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 1
  zero3_init_flag: false
  zero_stage: 0
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 6
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```
</details>


## Citation
If our project are helpful to you, please consider citing:

```BibTeX
@misc{R1Seg-3D,
      title={R1Seg-3D: Rethinking Reasoning Segmentation for Medical 3D CTs}, 
      author={Qinhao and Long Yu and Shengwei Tian and Xujiong Ye and Lei Zhang},
      year={2025},
      eprint={2025.00578},
      archivePrefix={},
      primaryClass={cs.CV}
}
```

## Acknowledgement
We appreciate open source projects including: 
[LLaVA](https://github.com/haotian-liu/LLaVA),
[M3D](https://github.com/BAAI-DCAI/M3D), 
[SegVol](https://github.com/BAAI-DCAI/SegVol).
