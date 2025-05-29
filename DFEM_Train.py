import json
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm
from transformers import Qwen2TokenizerFast
from DFEM import CrossDomainModel

# 定义数据集类
class EntityTextDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        entity_type_describe = item['entity_type_describe']
        input_text = item['input_text']
        labels = item['labels']

        # 对输入文本和描述文本进行tokenization
        encoding_A = self.tokenizer(input_text, truncation=True, padding=False, return_tensors="pt")
        encoding_B = self.tokenizer(entity_type_describe, truncation=True, padding=False, return_tensors="pt")

        # 转换labels为Tensor
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            'text': input_text,
            'input_A': encoding_A,
            'input_B': encoding_B,
            'labels': labels
        }



def collate_fn(batch):
    # 从每个样本中提取数据
    input_A_batch = [item['input_A'] for item in batch]
    input_B_batch = [item['input_B'] for item in batch]
    labels_batch = [item['labels'] for item in batch]
    texts = [item['text'] for item in batch]
    # 获取每个序列的长度
    input_A_lengths = [seq['input_ids'].size(1) for seq in input_A_batch]
    input_B_lengths = [seq['input_ids'].size(1) for seq in input_B_batch]

    # 计算批次中最长的序列长度
    max_input_A_len = max(input_A_lengths)
    max_input_B_len = max(input_B_lengths)

    # 使用 pad_sequence 对 input_A, input_B, labels 进行填充
    padded_input_A = pad_sequence([torch.tensor(seq['input_ids'].squeeze(0)) for seq in input_A_batch],
                                  batch_first=True, padding_value=0)
    padded_input_B = pad_sequence([torch.tensor(seq['input_ids'].squeeze(0)) for seq in input_B_batch],
                                  batch_first=True, padding_value=0)

    # 标签的长度应该与input_A的长度一致
    padded_labels = pad_sequence(labels_batch, batch_first=True, padding_value=-1)

    # 生成 attention_mask，填充部分为0，其他部分为1
    attention_mask_A = (padded_input_A != 0).long()
    attention_mask_B = (padded_input_B != 0).long()

    return {
        'input_A': padded_input_A,
        'input_B': padded_input_B,
        'labels': padded_labels,
        'attention_mask_A': attention_mask_A,
        'attention_mask_B': attention_mask_B,
        'max_input_A_len': max_input_A_len,
        'max_input_B_len': max_input_B_len,
        'texts': texts
    }

# 设置训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 模型和优化器
model = CrossDomainModel(embed_dim=1536, num_heads=16, ff_dim=2048, num_layers=9, num_classes=2).to(device)

# 加载保存的参数文件
checkpoint = torch.load("D:\Models_parm\model_layers_9_weight_25.pth")
# 将 attention 层的参数加载到模型中
model.attention.load_state_dict(checkpoint['attention'])
# 将 classifier 层的参数加载到模型中
model.classifier.load_state_dict(checkpoint['classifier'])

# 加载数据并创建数据集实例
with open("train_data.json", "r", encoding="utf-8") as file:
    train_datas = json.load(file)


model_path = "/root/autodl-tmp/Models/Qwen2.5-1.5B/"
tokenizer = Qwen2TokenizerFast.from_pretrained(model_path)

train_dataset = EntityTextDataset(train_datas, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)


# 冻结 Qwen2Model 的参数
for param in model.qwen.parameters():
    param.requires_grad = False

# 创建优化器，只优化 attention 和 classifier 的参数
optimizer = optim.Adam([
    {'params': model.attention.parameters()},
    {'params': model.classifier.parameters()}
], lr=1e-6)  # 设置初始学习率

# 初始化自定义调度器
num_epochs = 10

# 初始化优化器（假设已经定义了 model 和 optimizer）
num_batches_per_epoch = len(train_loader)

# 损失函数
criterion = nn.CrossEntropyLoss(reduction='none')


# 训练循环
for epoch in range(num_epochs):
    model.train()  # 设定模型为训练模式
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    true_positives = 0  # 真实标签和预测标签都为1的数量
    actual_positives = 0  # 真实标签为1的总数量
    w = 25
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
        for batch_idx, batch in enumerate(pbar):
            is_valid = True
            # 数据转移到设备
            input_A = batch['input_A'].to(device)
            input_B = batch['input_B'].to(device)
            attention_mask_A = batch['attention_mask_A'].to(device)
            attention_mask_B = batch['attention_mask_B'].to(device)
            labels = batch['labels'].to(device)

            # 检查注意力掩码的有效性
            for attention in attention_mask_A:
                if attention[0] == 0:
                    is_valid = False
            if not is_valid:
                continue
            # 前向传播
            logits, feature_matrix, describe_matrix= model(input_A, attention_mask_A, input_B, attention_mask_B)
            # 展开 logits 和 labels
            logits_flattened = logits.view(-1, logits.size(-1))
            labels_flattened = labels.view(-1)
            attention_mask_flattened = attention_mask_A.view(-1)

            # 创建掩码，标记哪些位置是有效的
            valid_mask = (attention_mask_flattened == 1)

            # 只选择有效的位置
            valid_logits = logits_flattened[valid_mask]
            valid_labels = labels_flattened[valid_mask]
            # 定义损失权重

            weights = torch.ones_like(valid_labels, dtype=torch.float, device=device)
            weights[valid_labels == 1] = max(10,w) # 标签为 1 的损失权重设置为 15
            # 计算损失
            if valid_logits.numel() > 0 and valid_labels.numel() > 0:
                losses = criterion(valid_logits, valid_labels) * weights  # 应用权重
                loss = losses.mean()  # 对加权损失取平均
            else:
                loss = torch.tensor(0.0, device=device)  # 如果没有有效位置，则损失为 0
            total_loss += loss.item()

            # 计算准确度
            _, predicted = torch.max(logits, dim=-1)
            valid_predicted = predicted[attention_mask_A.bool()]
            valid_labels = labels[attention_mask_A.bool()]
            correct_predictions = (valid_predicted == valid_labels).sum().item()
            total_predictions = valid_labels.numel()

            # 标签为1的准确度（即召回率）
            positive_labels_mask = valid_labels == 1
            actual_positives = positive_labels_mask.sum().item()
            true_positives = ((valid_predicted == 1) & positive_labels_mask).sum().item()
            num_predicted_ones = (valid_predicted == 1).sum().item()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新进度条的描述，显示当前的损失、总体准确度和标签为1的准确度
            avg_loss = total_loss / (pbar.n + 1)
            accuracy = correct_predictions / total_predictions * 100 if total_predictions > 0 else 0
            pos_accuracy = true_positives / actual_positives * 100 if actual_positives > 0 else 0
            pbar.set_postfix(loss=avg_loss, accuracy=f"{accuracy:.2f}%", pos_accuracy=f"{pos_accuracy:.2f}%",
                             num_1=num_predicted_ones, total_num=total_predictions)

    save_model_path = f"/root/autodl-tmp/Models/cross_model_param/model_layers_9_weight_25.pth"
    # 训练完成后保存指定层的参数
    torch.save({
        'attention': model.attention.state_dict(),
        'classifier': model.classifier.state_dict()
    }, save_model_path)

    print("Model layers saved successfully.")