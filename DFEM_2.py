import torch
import torch.nn as nn


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


class EntityAttentionDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers=9, num_classes=2, dropout=0.3, device="cuda"):
        super(EntityAttentionDecoder, self).__init__()


        self.embed_dim = embed_dim
        self.device = device
        # 跨领域注意力模块
        self.attention = StackedCrossDomainAttention(num_layers, embed_dim, num_heads, ff_dim, dropout)

        # 分类层
        self.classifier = nn.Linear(embed_dim, num_classes)



    def forward(self, encoded_matrix, describe_matrix, attention_mask_A,candidate_entities, labels=None, weight=5.0):

        encoded_matrix = encoded_matrix.to(self.device)
        describe_matrix = describe_matrix.to(self.device)
        candidate_entities = candidate_entities.to(self.device)

        if labels is not None:
            labels = labels.to(self.device)

        criterion = nn.CrossEntropyLoss(reduction='none')

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        predicted_labels = torch.zeros_like(candidate_entities, dtype=torch.long, device=self.device)

        encoded_matrix = encoded_matrix.permute(1, 0, 2)
        describe_matrix = describe_matrix.permute(1, 0, 2)

        A_out = self.attention(encoded_matrix, describe_matrix)  # 输出 A7
        A_out = A_out.permute(1, 0, 2)

        emissions = self.classifier(A_out)  # 输出 (batch_size, seq_len_a, num_classes)

        if labels is not None:
            logits_flattened = emissions.view(-1, emissions.size(-1))
            labels_flattened = labels.view(-1)
            attention_mask_flattened = attention_mask_A.view(-1)

            # 创建掩码，标记哪些位置是有效的
            valid_mask = (attention_mask_flattened == 1)

            # 只选择有效的位置
            valid_logits = logits_flattened[valid_mask]
            valid_labels = labels_flattened[valid_mask]

            weights = torch.ones_like(valid_labels, dtype=torch.float, device=self.device)
            weights[valid_labels == 1] = weight  # 标签为 1 的损失权重设置为 25
            # 计算损失
            if valid_logits.numel() > 0 and valid_labels.numel() > 0:
                losses = criterion(valid_logits, valid_labels) * weights  # 应用权重
                loss = losses.mean()  # 对加权损失取平均
            else:
                loss = torch.tensor(0.0, device=self.device)  # 如果没有有效位置，则损失为 0
            total_loss = total_loss + loss

        _, predicted = torch.max(emissions, dim=-1)

        #DPCV
        for i in range(candidate_entities.shape[0]):  # 遍历行
            for j in range(candidate_entities.shape[1]):  # 遍历列
                if candidate_entities[i, j] == 1 and predicted[i, j] == 1:  # 当两个张量的值均为1时
                    predicted_labels[i, j] = 1  # 新张量对应位置置为1

        return total_loss, predicted_labels