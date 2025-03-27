from typing import List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import numpy as np

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Phi3Config, Phi3Model, Phi3ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..lamed_arch_cross import LamedMetaModel, LamedMetaForCausalLM

class LamedPhi3Config(Phi3Config):
    model_type = "lamed_phi3_cross"

class LamedPhi3ModelCrossA(LamedMetaModel, Phi3Model):
    config_class = LamedPhi3Config
    def __init__(self, config: Phi3Config):
        super(LamedPhi3ModelCrossA, self).__init__(config)


class LamedPhi3ForCausalLMCrossA(LamedMetaForCausalLM, Phi3ForCausalLM):
    config_class = LamedPhi3Config

    def __init__(self, config):
        super(LamedPhi3ForCausalLMCrossA, self).__init__(config)
        self.model = LamedPhi3ModelCrossA(config)
        self.vocab_size = config.vocab_size
        self.test_mode = False
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            images: Optional[torch.FloatTensor] = None,
            input_ids: torch.LongTensor = None,
            labels: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            segs: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            num_logits_to_keep: torch.LongTensor = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        input_ids_pre = input_ids

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                image_features_vit,
            ) = self.prepare_inputs_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
            )


        outputs = super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            return_dict=return_dict
        )
        print(" LLM outputs.loss", outputs.loss)

        if self.config.tune_mm_mlp_adapter:
            return outputs

        try:
            seg_ids = torch.nonzero(torch.sum(segs, dim=(1, 2, 3, 4))).flatten().tolist()
        except:
            seg_ids = []

        if seg_ids:
            output_hidden_states = outputs.hidden_states
            last_hidden_state = output_hidden_states[-1]

            seg_token_mask = input_ids_pre[:, 1:] == self.config.seg_token_id
            seg_token_mask = torch.cat(
                [
                    seg_token_mask,
                    torch.zeros((seg_token_mask.shape[0], 1), dtype=seg_token_mask.dtype).cuda(),
                ],
                dim=1,
            )

            seg_prompts = []
            for i in seg_ids:
                if torch.sum(seg_token_mask[i]) == 1:
                    seg_token = last_hidden_state[i][seg_token_mask[i]]
                    seg_prompt = self.get_model().seg_projector(seg_token)
                elif torch.sum(seg_token_mask[i]) > 1:
                    seg_token = last_hidden_state[i][seg_token_mask[i]]
                    seg_token = torch.mean(seg_token, dim=0, keepdim=True)
                    seg_prompt = self.get_model().seg_projector(seg_token)
                else:
                    seg_prompt = torch.zeros([1, int(self.config.mm_hidden_size)], dtype=last_hidden_state.dtype,
                                             device=last_hidden_state.device)
                seg_prompts.append(seg_prompt)

            text_embedding = torch.cat(seg_prompts, dim=0)

            img_shape = (images.shape[2], images.shape[3], images.shape[4])
            # vision_module
            bs, _, _ = image_features_vit.shape
            img_emb_size = np.divide(self.config.img_size, self.config.patch_size).astype(int)
            image_features_vit = image_features_vit.permute(0, 2, 1).reshape(bs, -1, img_emb_size[0],
                                                                             img_emb_size[1],
                                                                             img_emb_size[2])
            low_res_masks = self.forward_decoder(image_features_vit[seg_ids], text_embeddings=text_embedding,
                                                 masks=None)
            for num_click in range(self.config.num_clicks):
                low_res_masks = self.forward_decoder(image_features_vit[seg_ids], text_embeddings=text_embedding,
                                                     masks=low_res_masks)

            logits = self.get_model().vision_module.postprocess_masks(
                low_res_masks,
                input_size=img_shape,
                original_size=img_shape,
            )

            loss_dice = self.get_model().vision_module.dice_loss(logits, segs[seg_ids])
            loss_bce = self.get_model().vision_module.bce_loss(logits, segs[seg_ids])
            print("loss_dice", loss_dice)
            print("loss_bce", loss_bce)
            seg_loss = loss_dice + loss_bce
            print("seg_loss", seg_loss)
            outputs.loss = outputs.loss + seg_loss
            outputs.logits = logits

        return outputs

    def forward_decoder(self, image_embeddings=None, text_embeddings=None, masks=None):
        sparse_embeddings, dense_embeddings = self.get_model().vision_module.prompt_encoder(
            text_embeddings=text_embeddings,
            masks=masks
        )
        dense_pe = self.get_model().vision_module.prompt_encoder.get_dense_pe()
        low_res_masks, _ = self.get_model().vision_module.mask_decoder(
            image_embeddings=image_embeddings,
            text_embedding = text_embeddings,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return low_res_masks

    @torch.no_grad()
    def generate(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor, Any]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            (
                input_ids,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                image_features_vit,
            ) = self.prepare_inputs_for_multimodal(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                labels=None,
                images=images,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)

        outputs = super().generate(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict_in_generate=True,
            **kwargs
        )

        output_hidden_states = outputs.hidden_states  # 8,33,(1,272,3072) (1,1,3072) (1,1,3072)...
        output_ids = outputs.sequences  # [1,max_new_tokens]
        seg_token_mask = output_ids[:, 1:] == self.config.seg_token_id
        if torch.sum(seg_token_mask).item()==0:
            logits = torch.zeros(images.shape).cuda()-1000
        else:
            last_tensors = [tuple[-1] for tuple in output_hidden_states]  # 8,(1,272,3072) (1,1,3072) (1,1,3072)...
            last_hidden_state = torch.cat(last_tensors[1:], dim=1)  # (1,7,3072)

            seg_prompts = []
            noseg_ids = []
            for i in range(len(seg_token_mask)):
                if torch.sum(seg_token_mask[i]) == 1:
                    seg_token = last_hidden_state[i][seg_token_mask[i]]
                    seg_prompt = self.get_model().seg_projector(seg_token)
                elif torch.sum(seg_token_mask[i]) > 1:
                    seg_tokens = last_hidden_state[i][seg_token_mask[i]]
                    seg_token = torch.mean(seg_tokens, dim=0, keepdim=True)
                    seg_prompt = self.get_model().seg_projector(seg_token)
                else:
                    noseg_ids.append(i)
                    seg_prompt = torch.zeros([1, self.config.mm_hidden_size], dtype=last_hidden_state.dtype,
                                             device=last_hidden_state.device)
                seg_prompts.append(seg_prompt)

            text_embedding = torch.cat(seg_prompts, dim=0)

            img_shape = (images.shape[2], images.shape[3], images.shape[4])
            # Mask
            bs, _, _ = image_features_vit.shape
            img_emb_size = np.divide(self.config.img_size, self.config.patch_size).astype(int)
            image_features_vit = image_features_vit.permute(0, 2, 1).reshape(bs, -1, img_emb_size[0], img_emb_size[1],
                                                                             img_emb_size[2])

            low_res_masks = self.forward_decoder(image_features_vit, text_embeddings=text_embedding,
                                                 masks=None)
            for num_click in range(self.config.num_clicks):
                low_res_masks = self.forward_decoder(image_features_vit, text_embeddings=text_embedding,
                                                     masks=low_res_masks)

            logits = self.get_model().vision_module.postprocess_masks(
                low_res_masks,
                input_size=img_shape,
                original_size=img_shape,
            )
        return {"output_ids":output_ids, "logits":logits}


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        return inputs


AutoConfig.register("lamed_phi3_cross", LamedPhi3Config)
AutoModelForCausalLM.register(LamedPhi3Config, LamedPhi3ForCausalLMCrossA)