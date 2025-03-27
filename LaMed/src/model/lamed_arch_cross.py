from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from .multimodal_projector.builder import build_mm_projector
from LaMed.src.model.r1Seg3DSAM_Config import R1Seg3DSAM_Config
from LaMed.src.model.build_vit3dseg import model_registry
from safetensors.torch import load_file

class LamedMetaModel:
    def __init__(self, config):
        super(LamedMetaModel, self).__init__(config)
        self.config = config
        self.seg_enable = True
        self.vision_module = None
        # mm_projector
        if hasattr(config, "mm_projector_type") and hasattr(config, "patch_size") and hasattr(config, "mm_hidden_size"):
            self.mm_projector = build_mm_projector(config)

        if hasattr(config, "vision_module"):
            # vision module
            config = R1Seg3DSAM_Config.from_dict(vars(config))
            self.vision_module =  model_registry['vit'](config=config, checkpoint=None)
            self.seg_projector = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(config.hidden_size, config.mm_hidden_size),
                nn.Dropout(0.1),
            )

    def initialize_vision_modules(self, model_args):
        self.config.image_channel = model_args.image_channel
        self.config.img_size = model_args.img_size
        self.config.patch_size = model_args.patch_size
        self.config.num_clicks = model_args.num_clicks
        self.config.vision_select_layer = model_args.vision_select_layer
        self.config.vision_select_feature = model_args.vision_select_feature
        self.config.mm_projector_type = model_args.mm_projector_type
        self.config.proj_layer_type = model_args.proj_layer_type
        self.config.proj_layer_num = model_args.proj_layer_num
        self.config.proj_pooling_type = model_args.proj_pooling_type
        self.config.proj_pooling_size = model_args.proj_pooling_size

        # vision module
        config = R1Seg3DSAM_Config.from_dict(vars(model_args))
        self.config.mm_hidden_size = config.mm_hidden_size = model_args.hidden_size
        self.vision_module =  model_registry['vit'](config=config, checkpoint=None)

        if model_args.pretrain_vision_model is not None:
            if model_args.pretrain_vision_model.endswith('safetensors'):
                vision_model_weights = load_file(model_args.pretrain_vision_model)
            else:
                vision_model_weights = torch.load(model_args.pretrain_vision_model, map_location='cpu')
            self.vision_module.load_state_dict(vision_model_weights, strict=False)

        # mm_projector
        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_mm_projector(self.config)

        if model_args.pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=True)

        # seg_projector for [SEG] seg_token
        self.seg_projector = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.hidden_size, int(self.config.mm_hidden_size)),
            nn.Dropout(0.1),
        )

class LamedMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def encode_images(self, images):
        image_features_vit, hidden_states_vit = self.get_model().vision_module.image_encoder(images)
        image_features = self.get_model().mm_projector(image_features_vit) # SpatialPoolingProjector
        return image_features, image_features_vit

    def prepare_inputs_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images,
    ):
        if images is None or input_ids.shape[1] == 1:
            print("images is None", images is None)
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None
        else:
            image_features, image_features_vit = self.encode_images(images)
            inputs_embeds = self.get_model().embed_tokens(input_ids)
            inputs_embeds = torch.cat(
                (inputs_embeds[:, :1, :], image_features, inputs_embeds[:, (image_features.shape[1] + 1):, :]),
                dim=1)
            return None, position_ids, attention_mask, past_key_values, inputs_embeds, labels, image_features_vit

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        num_new_tokens = model_args.num_new_tokens
        self.resize_token_embeddings(len(tokenizer))
        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data
            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
            else:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = True

        if model_args.pretrain_mm_mlp_adapter:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']

            if input_embeddings.shape == embed_tokens_weight.shape:
                input_embeddings = embed_tokens_weight
            elif embed_tokens_weight.shape[0] == num_new_tokens:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight
            else:
                raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")