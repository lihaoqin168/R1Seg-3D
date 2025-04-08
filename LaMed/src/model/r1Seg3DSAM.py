import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, CLIPTextModel
from LaMed.src.model.loss import BCELoss, BinaryDiceLoss
from typing import Tuple

class R1Seg3DSAM(nn.Module):
    mask_threshold: float = 0.0
    def __init__(self,
                image_encoder,
                mask_decoder,
                prompt_encoder,
                config,
                ):
        """
        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        if config.need_text_en:
            self.text_encoder = TextEncoder(config)
        self.feat_shape = np.array(config.img_size)/np.array(config.patch_size)
        self.decoder_iter = 6
        self.img_size = config.img_size
        self.num_clicks = config.num_clicks
        self.test_mode = config.test_mode
        self.dice_loss = BinaryDiceLoss()
        self.bce_loss = BCELoss()

        # #align image and text
        self.mm_vision_proj = nn.Linear(config.mm_hidden_size, config.mm_hidden_size)
        self.mm_language_proj = nn.Linear(config.mm_hidden_size, config.mm_hidden_size)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.local_loss = config.local_loss
        self.gather_loss = config.gather_loss

    def forward(self, images, promptargets=None, impaths=None, labels=None, segs=None):
        bs = images.shape[0]
        img_shape = (images.shape[2], images.shape[3], images.shape[4])
        #SAM
        image_embedding, _ = self.image_encoder(images)
        image_embedding = image_embedding.transpose(1, 2).view(bs, -1,
            int(self.feat_shape[0]), int(self.feat_shape[1]), int(self.feat_shape[2]))

        #SAM test mode
        if self.test_mode:
            with torch.no_grad():
                logits = self.supervised_forward(image_embeddings=image_embedding, img_shape=img_shape, train_segs=None, texts=promptargets)
            ret = {
                "loss": None,
                "logits": logits,
            }
        else:
            sl_loss, logits = self.supervised_forward(image_embeddings=image_embedding,
                                                      img_shape=img_shape, train_segs=segs, texts=promptargets)
            print('seg total loss :', sl_loss.item())
            ret = {
                "loss": sl_loss,
                "logits": logits,
            }

        return ret


    def generate(self, images, promptargets=None,):
        bs = images.shape[0]
        img_shape = (images.shape[2], images.shape[3], images.shape[4])
        #SAM
        image_embedding, _ = self.image_encoder(images)
        image_embedding = image_embedding.transpose(1, 2).view(bs, -1,
            int(self.feat_shape[0]), int(self.feat_shape[1]), int(self.feat_shape[2]))

        with torch.no_grad():
            logits = self.supervised_forward(image_embeddings=image_embedding, img_shape=img_shape, train_segs=None,
                                             texts=promptargets)
            return logits

    def forward_decoder(self, image_embeddings=None, text_embeddings=None, masks=None):
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            text_embeddings=text_embeddings,
            masks=masks
        )

        dense_pe = self.prompt_encoder.get_dense_pe()
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            text_embedding = text_embeddings,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return low_res_masks

    def supervised_forward(self, image_embeddings, img_shape, train_segs, texts):
        sl_loss = 0
        text_embeddings = self.text_encoder(texts=texts)
        low_res_masks = self.forward_decoder(image_embeddings, text_embeddings=text_embeddings, masks=None)

        for num_click in range(self.num_clicks):
                low_res_masks = self.forward_decoder(image_embeddings, text_embeddings=text_embeddings, masks=low_res_masks)

        logits = self.postprocess_masks(masks=low_res_masks, input_size=img_shape, original_size=img_shape)
        if train_segs is None:
            return logits
        else:
            sl_loss_dice = self.dice_loss(logits, train_segs)
            sl_loss_bce = self.bce_loss(logits, train_segs)
            print('dice loss:',sl_loss_dice.item())
            sl_loss += sl_loss_dice + sl_loss_bce
        return sl_loss, logits

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (input_size[0], input_size[1], input_size[2]),
            mode="trilinear",
            align_corners=False,
        )
        if original_size is not None and input_size!=original_size:
            print('++ remove padding and resize to original_size !!',input_size, original_size)
            masks = masks[..., : input_size[0], : input_size[1], : input_size[2]]
            masks = F.interpolate(masks, original_size, mode="trilinear", align_corners=False)
        return masks

class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.sam_bert_path)
        self.clip_text_model = CLIPTextModel.from_pretrained(config.sam_bert_path)
        self.dim_align = nn.Linear(512, config.hidden_size)
        # freeze text encoder
        for param in self.clip_text_model.parameters():
            param.requires_grad = False

    def organ2tokens(self, texts):
        text_list = [text for text in texts]
        tokens = self.tokenizer(text_list, padding=True, return_tensors="pt")
        input_ids = tokens['input_ids'].cuda()
        attention_mask = tokens['attention_mask'].cuda()
        return input_ids, attention_mask

    def forward(self, texts):
        if texts is None:
            return None
        if type(texts) is str:
            texts = [texts]
        input_ids, attention_mask = self.organ2tokens(texts)
        clip_outputs = self.clip_text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embedding = clip_outputs.pooler_output# pooled (EOS token) states, last_hidden_state[:, 0, :];->nn.Linear(config.hidden_size, config.hidden_size);->nn.Tanh()

        text_embedding = self.dim_align(text_embedding)
        return text_embedding