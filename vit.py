import re
from collections.abc import Mapping

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
        if x.ndim != 4:
            raise ValueError("输入应为四维张量：[批大小, 通道数, 高度, 宽度]。")
        height, width = x.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"图片高宽都必须能被 patch 大小整除，实际为 {height}×{width}，"
                f"patch 大小为 {self.patch_size}。"
            )
        x = self.patcher(x)
        x = self.flatten(x)
        return x.permute(0, 2, 1)


class MutilHeadSelfAttentionBlock(nn.Module):
    def __init__(self,
                 embedding_dim : int = 768,
                 num_heads : int =  12,
                 attn_dropout : float = 0) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim, eps=1e-6)
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
        self.layer_norm = nn.LayerNorm(normalized_shape=embedimg_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedimg_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, out_features=embedimg_dim),
            nn.Dropout(p=dropout)
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

        if img_size <= 0 or patch_size <= 0 or img_size % patch_size:
            raise ValueError("图片边长和 patch 大小必须为正整数，且边长必须能被 patch 大小整除。")
        if num_transformer_layers < 1:
            raise ValueError("Transformer 编码器层数必须大于零。")
        self.img_size = img_size
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
            nn.LayerNorm(normalized_shape=embedding_dim, eps=1e-6),
            nn.Linear(in_features=embedding_dim, out_features=num_classes)
        )

    def load_pretrained_weights(self, state_dict: Mapping, load_classifier: bool = False):
        """将 torchvision ViT 的权重映射到手写模型。

        默认只加载主干，保留当前任务的随机分类头。仅当类别数与来源模型
        一致时，才使用 load_classifier=True 加载分类头。层数、参数形状及
        所有主干参数必须完整匹配；不支持卷积前端或带额外投影层的模型。
        同时兼容官方早期权重的 mlp.linear_1 / mlp.linear_2 参数名。
        注意力头数不保存在 state_dict 中，调用者仍需采用来源架构的头数。
        """
        # 官方部分权重仍使用旧 MLP 命名，复制后转换，避免修改调用者的数据。
        state_dict = dict(state_dict)
        for key in list(state_dict):
            legacy = re.fullmatch(
                r"(encoder\.layers\.encoder_layer_\d+\.mlp\.)linear_([12])\.(weight|bias)",
                key,
            )
            if legacy:
                layer = "0" if legacy.group(2) == "1" else "3"
                normalized_key = f"{legacy.group(1)}{layer}.{legacy.group(3)}"
                if normalized_key in state_dict:
                    raise ValueError(f"预训练权重同时包含新旧两套参数：{key} 与 {normalized_key}。")
                state_dict[normalized_key] = state_dict.pop(key)

        layer_indices = {
            int(match.group(1))
            for key in state_dict
            if (match := re.match(r"encoder\.layers\.encoder_layer_(\d+)\.", key))
        }
        expected_layers = set(range(len(self.transformer_emcoder)))
        if layer_indices != expected_layers:
            raise ValueError(
                f"预训练编码器层数不匹配：期望层编号 {sorted(expected_layers)}，"
                f"实际为 {sorted(layer_indices)}。"
            )

        # 显式列出每个参数的来源，避免宽松加载时遗漏主干参数。
        key_map = {
            "class_token": "class_embedding",
            "encoder.pos_embedding": "position_embedding",
            "conv_proj.weight": "patch_embedding.patcher.weight",
            "conv_proj.bias": "patch_embedding.patcher.bias",
            "encoder.ln.weight": "classifier.0.weight",
            "encoder.ln.bias": "classifier.0.bias",
        }
        block_parts = {
            "ln_1.weight": "msa_block.layer_norm.weight",
            "ln_1.bias": "msa_block.layer_norm.bias",
            "self_attention.in_proj_weight": "msa_block.multihead_attn.in_proj_weight",
            "self_attention.in_proj_bias": "msa_block.multihead_attn.in_proj_bias",
            "self_attention.out_proj.weight": "msa_block.multihead_attn.out_proj.weight",
            "self_attention.out_proj.bias": "msa_block.multihead_attn.out_proj.bias",
            "ln_2.weight": "mlp_block.layer_norm.weight",
            "ln_2.bias": "mlp_block.layer_norm.bias",
            "mlp.0.weight": "mlp_block.mlp.0.weight",
            "mlp.0.bias": "mlp_block.mlp.0.bias",
            "mlp.3.weight": "mlp_block.mlp.3.weight",
            "mlp.3.bias": "mlp_block.mlp.3.bias",
        }
        for index in sorted(expected_layers):
            for source, target in block_parts.items():
                key_map[f"encoder.layers.encoder_layer_{index}.{source}"] = (
                    f"transformer_emcoder.{index}.{target}"
                )

        classifier_map = {
            "heads.head.weight": "classifier.1.weight",
            "heads.head.bias": "classifier.1.bias",
        }
        if load_classifier:
            key_map.update(classifier_map)
        missing = sorted(set(key_map) - set(state_dict))
        unexpected = sorted(set(state_dict) - set(key_map) - set(classifier_map))
        if missing or unexpected:
            raise ValueError(f"预训练权重参数不完整或架构不受支持；缺少：{missing}；多出：{unexpected}。")

        target_state = self.state_dict()
        mapped_state = {}
        for source, target in key_map.items():
            value = state_dict[source]
            if not isinstance(value, torch.Tensor) or value.shape != target_state[target].shape:
                source_shape = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
                raise ValueError(
                    f"预训练参数 {source} 形状不匹配：来源为 {source_shape}，"
                    f"手写模型需要 {tuple(target_state[target].shape)}。"
                )
            mapped_state[target] = value

        # 所有检查都在写入参数前完成，加载失败时保留原有模型参数。
        ignored = set() if load_classifier else set(classifier_map.values())
        if set(target_state) - set(mapped_state) != ignored:
            raise ValueError("手写模型存在未映射的参数，需同步更新预训练权重映射。")
        return self.load_state_dict(mapped_state, strict=load_classifier)

    def forward(self, x):
        if x.ndim != 4 or x.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f"ViT 需要 [批大小, 通道数, {self.img_size}, {self.img_size}] 的固定分辨率输入，"
                f"实际形状为 {tuple(x.shape)}。"
            )
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

    # 将图片划分为 patch 并生成 embedding。
    # input = torch.rand((1, 3, 224, 224))    # 输入：[1, 3, 224, 224]
    # print(input.shape)
    # model = PatchEmbedding()
    # output = model(input)       # 输出：[1, 196, 768]
    # print(output.shape)


    from torchinfo import summary
    vit = ViT()
    # 使用 torchinfo 查看手写 ViT 的结构摘要。
    summary(model=vit,
            input_size=(32, 3, 224, 224), # （批大小，颜色通道数，高度，宽度）
            # col_names=["input_size"], # 取消注释可精简摘要输出。
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
)
