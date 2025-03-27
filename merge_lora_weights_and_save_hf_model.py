import os
import torch
from typing import Optional
import transformers
from transformers import AutoTokenizer
from dataclasses import dataclass, field
from LaMed.src.model.language_model.lamed_phi3_crossA import LamedPhi3ForCausalLMCrossA
from LaMed.src.model.language_model.lamed_llama import LamedLlamaForCausalLM
from LaMed.src.model.language_model.lamed_llava_med import LlavaMistralForCausalLM
from LaMed.src.model.language_model.lamed_Qwen2 import LamedQwenForCausalLM

@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    model_with_lora: str = field(default=None)
    model_name_or_path: Optional[str] = field(default="/yulong/pretrain/Phi-3-mini-4k-instruct", metadata={"help": "Path to the LLM or MLLM."})
    model_type: Optional[str] = field(default="phi3", metadata={"help": "llama3, phi3, Qwen2.5, llavaMed"})

    need_text_en: bool = field(default=False)# if build TextEncoder
    test_mode: bool = field(default=False)

    freeze_backbone: bool = field(default=False)

    tune_mm_mlp_adapter: bool = field(default=False, metadata={"help": "Used in pretrain: tune mm_projector and embed_tokens"})
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None, metadata={"help": "Path to pretrained mm_projector and embed_tokens."})

    # image
    image_channel: int = field(default=1)
    img_size: tuple = field(default=(32, 256, 256))
    patch_size: tuple = field(default=(4, 16, 16))

    # projector
    mm_projector_type: Optional[str] = field(default='spp', metadata={"help": "spp"})
    proj_layer_type: str = field(default="mlp", metadata={"help": "Type of layer in projector. options: [linear, mlp]."})
    proj_layer_num: int = field(default=2, metadata={"help": "Number of layers in projector."})
    proj_pooling_type: str = field(default="spatial", metadata={"help": "Type of pooling in projector. options: [spatial, sequence]."})
    proj_pooling_size: int = field(default=2, metadata={"help": "Size of pooling in projector."})

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
class TrainingArguments(transformers.TrainingArguments):
    lora_enable: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    cache_dir: Optional[str] = field(default=None)
    output_dir: str = "./LaMed/output/LaMed-Phi3-4B-finetune-0000/hf/"



def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    # Process of elimination: LoRA only targets on LLM backbone
    ignore_keywords = ['vision_module', 'mm_projector', 'embed_tokens', 'lm_head', 'seg_projector']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in ignore_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    return list(lora_module_names)


def main():
    gpu = 0
    torch.cuda.set_device(gpu)
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    print("Tokenizer preparation")
    # Load tokenizer from the given path with specified configurations
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
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
    print("seg_token_id: ", model_args.seg_token_id)
    print("vocab_size: ", model_args.vocab_size)

    print("Model preparation")
    if 'phi3' in model_args.model_type:
        model = LamedPhi3ForCausalLMCrossA.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir
        )
    elif 'llama3' in model_args.model_type:
        model = LamedLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir
        )
    elif 'Qwen2.5' in model_args.model_type:
        model = LamedQwenForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir
        )
    elif 'llavaMed' in model_args.model_type:
        model = LlavaMistralForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir
        )

    else:
        raise ValueError(f"Unknown Model Type {model_args.model_type}")

    model.config.seg_token_id = model_args.seg_token_id
    model.config.use_cache = False
    model.config.vision_module = model_args.vision_module
    model.config.need_text_en = model_args.need_text_en
    model.config.mm_hidden_size = model.config.hidden_size
    model.config.test_mode = model_args.test_mode

    # initialize vision and seg modules on LLM
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
        print("Adding LoRA adapters only on LLM.")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    print("Load weights with LoRA")
    state_dict = torch.load(model_args.model_with_lora, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    print("Merge weights with LoRA")
    model = model.merge_and_unload()
    state_dict = model.state_dict()

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    torch.save(state_dict, os.path.join(training_args.output_dir, 'merged_model.bin'))

    model.model.config.architectures = model.__class__.__name__
    model._name_or_path = training_args.output_dir

    print("Save pretrained")
    model.config.save_pretrained(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    print("Finish")


if __name__ == "__main__":
    main()
