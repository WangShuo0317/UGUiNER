
import torch
import torch.nn as nn
from transformers import Qwen2Model


class CrossDomainAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.2):
        super(CrossDomainAttention, self).__init__()

        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

        # 残差连接和归一化
        self.norm1_a = nn.LayerNorm(embed_dim)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

        # 残差连接和归一化
        self.norm2_a = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, A, B):
        """
        Args:
            A (Tensor): 特征矩阵 A，shape = (seq_len_a, batch_size, embed_dim)
            B (Tensor): 特征矩阵 B，shape = (seq_len_b, batch_size, embed_dim)
            mask_A (Tensor): 文本 A 的 attention mask，shape = (batch_size, seq_len_a)
            mask_B (Tensor): 文本 B 的 attention mask，shape = (batch_size, seq_len_b)

        Returns:
            Tensor: 特征 A2，shape = (seq_len_a, batch_size, embed_dim)
        """
        C, _ = self.multihead_attn(A, B, B)
        # Step 2: 残差连接 + 层归一化
        A_prime = self.norm1_a(A + self.dropout(C))

        # Step 3: 前馈网络
        A_ffn = self.ffn(A_prime)

        # Step 4: 残差连接 + 层归一化（再次处理 A）
        A2 = self.norm2_a(A_prime + self.dropout(A_ffn))
        return A2


class StackedCrossDomainAttention(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout=0.2):
        super(StackedCrossDomainAttention, self).__init__()
        self.layers = nn.ModuleList(
            [CrossDomainAttention(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )

    def forward(self, A, B):
        for layer in self.layers:
            A = layer(A, B)
        return A


class CrossDomainModel(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, num_classes, dropout=0.3):
        super(CrossDomainModel, self).__init__()

        # 特征提取层（使用预训练Qwen）
        self.qwen = Qwen2Model.from_pretrained("/root/autodl-tmp/Models/Qwen2.5-1.5B/")
        self.embedding_dim = self.qwen.config.hidden_size  # Qwen嵌入维度

        # 跨领域注意力模块
        self.attention = StackedCrossDomainAttention(num_layers, embed_dim, num_heads, ff_dim, dropout)

        # 分类层
        self.classifier = nn.Linear(embed_dim, num_classes)  # 因为是双向LSTM，所以输入大小翻倍

    def forward(self, input_A, attention_mask_A, input_B, attention_mask_B, tags=None):
        """
        Args:
            input_A (Tensor): 文本 A 的输入 ID
            attention_mask_A (Tensor): 文本 A 的 attention mask
            input_B (Tensor): 文本 B 的输入 ID
            attention_mask_B (Tensor): 文本 B 的 attention mask
            tags (Tensor, optional): 真实标签，用于训练阶段的CRF层

        Returns:
            emissions (Tensor): 发射分数，shape = (batch_size, seq_len_a, num_classes)
            loss (Tensor, optional): 如果提供了真实标签，则返回损失值
        """
        # Step 1: 使用 Qwen 提取特征，避免每次迭代中计算不必要的梯度
        with torch.no_grad():
            outputs_A = self.qwen(input_ids=input_A, attention_mask=attention_mask_A)
            outputs_B = self.qwen(input_ids=input_B, attention_mask=attention_mask_B)

        # 提取最后一层隐藏状态
        A = outputs_A.last_hidden_state.permute(1, 0, 2)  # 转为 (seq_len_a, batch_size, embed_dim)
        B = outputs_B.last_hidden_state.permute(1, 0, 2)  # 转为 (seq_len_b, batch_size, embed_dim)

        # Step 2: 通过跨领域注意力模块
        A_out = self.attention(A, B)  # 输出 A7
        A_out = A_out.permute(1, 0, 2)  # 转为 (batch_size, seq_len_a, embed_dim)
        B_out = B.permute(1,0,2)
        # Step 5: 分类层
        emissions = self.classifier(A_out)  # 输出 (batch_size, seq_len_a, num_classes)

        return emissions,A_out,B_out
