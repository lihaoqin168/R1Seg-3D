# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
from pathlib import Path
import urllib.request
import torch
from .r1Seg3DSAM import R1Seg3DSAM

from .segment_anything_volumetric.modeling import (
    MaskDecoder,
    PromptEncoder,
    TwoWayTransformer,
)
import numpy as np
from .multimodal_encoder.vit import ViT

def build_vit3dseg(config, checkpoint=None):
    print('build_vit3dseg...')
    return _build_vit3dseg(
        image_encoder_type='vit',
        checkpoint=checkpoint,
        config=config,
    )

model_registry = {
    "vit": build_vit3dseg,
}


def _build_vit3dseg(
    image_encoder_type,
    checkpoint,
    config,
):
    
    image_encoder=ViT(
        in_channels=1,
        img_size=config.img_size,
        patch_size=config.patch_size,
        hidden_size=config.mm_hidden_size,
        mlp_dim=config.mlp_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        pos_embed=config.pos_embed,
        classification=False,
        dropout_rate=config.dropout_rate,
    )
    image_embedding_size = [int(item) for item in (np.array(config.img_size) / np.array(config.patch_size))]

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location='cpu')['state_dict']
            encoder_dict = {k.replace('model.encoder.', ''): v for k, v in state_dict.items() if 'model.encoder.' in k}
        image_encoder.load_state_dict(encoder_dict)
        print(f'===> image_encoder.load_param: {checkpoint}')
    return R1Seg3DSAM(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=config.mm_hidden_size,
            image_embedding_size=image_embedding_size,
            input_image_size=config.img_size,
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            image_encoder_type=image_encoder_type,
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=config.mm_hidden_size,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=config.mm_hidden_size,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            image_size=np.array(config.img_size),
            patch_size=np.array(config.patch_size),
        ),
        config=config,
    )
