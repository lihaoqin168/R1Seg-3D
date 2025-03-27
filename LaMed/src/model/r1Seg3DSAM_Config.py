from transformers import PretrainedConfig, AutoConfig

class R1Seg3DSAM_Config(PretrainedConfig):
    model_type = "r1seg-3dsam"

    def __init__(
        self,
        language_model_name_or_path: str = "",
        sam_bert_path: str = "",
        local_loss: bool = False,
        gather_loss: bool = True,
        in_channels: int = 1,
        img_size: tuple = (32, 256, 256),
        patch_size: tuple = (4, 16, 16),
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        dropout_rate: float = 0,
        spatial_dims: int = 3,
        max_text_len: int = 128,
        vocab_size: int = 30522,
        num_clicks: int = 0,
        **kwargs,
    ):
        self.language_model_name_or_path = language_model_name_or_path
        self.sam_bert_path = sam_bert_path
        self.in_channels = in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pos_embed = pos_embed
        self.dropout_rate = dropout_rate
        self.spatial_dims = spatial_dims
        self.local_loss = local_loss
        self.gather_loss = gather_loss
        self.max_text_len = max_text_len
        self.vocab_size = vocab_size
        self.num_clicks = num_clicks
        super().__init__(**kwargs)

AutoConfig.register("r1seg-3dsam", R1Seg3DSAM_Config)