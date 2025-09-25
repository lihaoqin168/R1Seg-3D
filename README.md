# R1Seg-3D: Rethinking Reasoning Segmentation for Medical 3D CTs

[Paper]:https://ore.exeter.ac.uk/articles/conference_contribution/R1Seg-3D_Rethinking_Reasoning_Segmentation_for_Medical_3D_CTs/30024691
[Model]:https://huggingface.co/lihao0011/R1Seg-3D-Phi-3-4B
<font size=3><div align='center' > <a href=https://ore.exeter.ac.uk/articles/conference_contribution/R1Seg-3D_Rethinking_Reasoning_Segmentation_for_Medical_3D_CTs/30024691>**Paper**</a> | [**Model**][model] </div></font>
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

## Model
| Model    | Download Link                                                                                                                                 |
|----------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| R1Seg-3D-SAM | [HuggingFace](https://huggingface.co/lihao0011/R1Seg-3D-SAM)   |
| R1Seg-3D-Phi-3-4B  | [HuggingFace](https://huggingface.co/lihao0011/R1Seg-3D-Phi-3-4B)|
| R1Seg-3D-llama-3-8B  | [HuggingFace](https://huggingface.co/lihao0011/R1Seg-3D-llama-3-8B)|


#### LLM
Phi-3-4B: Download and follow [here](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct).
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
