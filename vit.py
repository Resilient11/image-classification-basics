import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, 
                 in_channels : int = 3,
                 patch_size : int = 16,
                 embedding_dim : int = 768) -> None :
        super().__init__()
        self.patch_size = patch_size
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)
        self.flatten  = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x) -> torch.Tensor:
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0,\
            f"Input image size must be divisible by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"
        x = self.patcher(x)
        x = self.flatten(x)
        return x.permute(0, 2, 1)


class MutilHeadSelfAttentionBlock(nn.Module):
    def __init__(self,
                 embedding_dim : int = 768, 
                 num_heads : int =  12,
                 attn_dropout : float = 0) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim, 
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True)
    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query = x,
                                             key = x,
                                             value = x,
                                             need_weights = False)
        return attn_output 
    

class MLPBlock(nn.Module):
    def __init__(self,
                 embedimg_dim : int = 768,
                 mlp_size : int = 3072,
                 dropout : float = 0.1) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedimg_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedimg_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, out_features=embedimg_dim),
            nn.Dropout()
        )
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, 
                 embeding_dim : int = 768,
                 num_head : int = 12,
                 mlp_size : int = 3072,
                 mlp_dropout : float = 0.1,
                 attn_dropout : float = 0
                 ) -> None:
        super().__init__()
        self.msa_block = MutilHeadSelfAttentionBlock(embedding_dim=embeding_dim, 
                                                     num_heads=num_head,
                                                     attn_dropout=attn_dropout)
        self.mlp_block = MLPBlock(embedimg_dim=embeding_dim, 
                                  mlp_size=mlp_size,
                                  dropout=mlp_dropout)
    def forward(self, x):
        x = x + self.msa_block(x)
        x = x + self.mlp_block(x)
        return x

class ViT(nn.Module):
    def __init__(self, 
                 img_size : int = 224,
                 in_channel : int = 3,
                 patch_size : int = 16,
                 embedding_dim : int = 768,
                 mlp_size : int = 3072,
                 mlp_dropout : float = 0.1,
                 attn_dropout : float = 0,
                 num_transformer_layers : int = 12,
                 num_head : int = 12,
                 embedding_dropout : float = 0.1,
                 num_classes : int = 1000) -> None:
        super().__init__()

        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."
        self.num_patches = (img_size*img_size) // patch_size**2
        self.class_embedding = nn.Parameter(data=torch.randn((1, 1, embedding_dim)),
                                            requires_grad=True)
        self.position_embedding = nn.Parameter(data=torch.randn((1, self.num_patches+1, embedding_dim)),
                                            requires_grad=True)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.patch_embedding = PatchEmbedding(in_channels=in_channel,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)
        self.transformer_emcoder = nn.Sequential( *[TransformerEncoderBlock(embeding_dim=embedding_dim,
                                                           num_head=num_head,
                                                           mlp_size=mlp_size,
                                                           mlp_dropout=mlp_dropout,
                                                           attn_dropout=attn_dropout) for _ in range(num_transformer_layers)])
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        x =  self.patch_embedding(x)
        x = torch.cat((class_token, x), dim=1)
        x = x + self.position_embedding
        x = self.embedding_dropout(x)
        x = self.transformer_emcoder(x)
        x = self.classifier(x[:, 0])
        return x
    


if __name__ == "__main__":

    # patch embedding
    # input = torch.rand((1, 3, 224, 224))    # input : [1, 3, 224, 224]
    # print(input.shape)
    # model = PatchEmbedding()
    # output = model(input)       # output : [1, 196, 768]
    # print(output.shape)


    from torchinfo import summary
    vit = ViT()
    # # Print a summary of our custom ViT model using torchinfo (uncomment for actual output)
    summary(model=vit,
            input_size=(32, 3, 224, 224), # (batch_size, color_channels, height, width)
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
)