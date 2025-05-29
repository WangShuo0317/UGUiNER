import json
import torch
import re
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Qwen2TokenizerFast
from DFEM import CrossDomainModel
from DFEM_2 import EntityAttentionDecoder


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



def find_word_indices(text):
    # 正则表达式匹配单词或符号
    pattern = r'\w+|[^\w\s]'

    # 找到所有匹配的部分
    matches = list(re.finditer(pattern, text))

    # 创建索引对的列表
    result = [(match.start(), match.end()) for match in matches]

    return result

def find_specific_symbol_indices(text, specific_symbols={',', '、', ':'}):
    # 将特定符号转换为正则表达式模式
    pattern = '|'.join(re.escape(symbol) for symbol in specific_symbols)
    # 查找所有匹配的符号并记录其索引
    return [(match.start(), match.end()) for match in re.finditer(pattern, text)]


def adjust_labels(word_indices,symbol_indices,token_offsets, labels):
    """
    调整标签列表，确保一个单词对应的所有 token 的标签一致。

    参数:
        word_indices (list): 单词的索引列表，例如 [(0, 3), (4, 10), ...]。
        token_offsets (list): 分词后的 offsets 列表，例如 [(0, 3), (3, 8), ...]。
        labels (list): 分词后的标签列表，例如 [0, 1, 0, ...]。

    返回:
        list: 调整后的标签列表。
    """
    adjusted_labels = labels.copy()  # 复制原始标签列表以避免修改原列表
    for word_start, word_end in word_indices:
        # 找到当前单词对应的所有 token
        token_indices = [
            i for i, (token_start, token_end) in enumerate(token_offsets)
            if token_start < word_end and token_end > word_start
        ]
        # 如果当前单词对应的 token 中有任何一个标签为 1，则将所有 token 的标签设置为 1
        if any(labels[i] == 1 for i in token_indices):
            for i in token_indices:
                adjusted_labels[i] = 1

    for symbol_start, symbol_end in symbol_indices:
        # 找到当前单词对应的所有 token
        token_indices = [
            i for i, (token_start, token_end) in enumerate(token_offsets)
            if token_start == symbol_start and token_end == symbol_end
        ]
        # 如果当前单词对应的 token 中有任何一个标签为 1，则将所有 token 的标签设置为 1
        for i in token_indices:
            adjusted_labels[i] = 0

    return adjusted_labels


def adjust_labels_based_on_words_batch(texts, batch_labels_tensor, device, tokenizer):
    adjusted_batch_labels = []
    for text, labels_tensor in zip(texts, batch_labels_tensor):
        encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoding.offset_mapping  # 获取每个token在原始文本中的偏移量
        word_indices = find_word_indices(text)
        symbol_indices = find_specific_symbol_indices(text)
        labels = labels_tensor.tolist()
        adjusted_label = adjust_labels(word_indices,symbol_indices ,offsets, labels)
        adjusted_batch_labels.append(adjusted_label)

    adjusted_labels_tensor = torch.tensor(adjusted_batch_labels, dtype=torch.long).to(device)
    return adjusted_labels_tensor

def find_entity_spans(labels):
    """
    Given a list of labels, returns a list of entity spans (start, end).

    :param labels: List[int] - sequence of labels (1 for inside an entity, 0 otherwise)
    :return: List[Tuple[int, int]] - list of entity spans (start, end)
    """
    entities = []
    start = None
    for i, label in enumerate(labels + [0]):  # Add extra 0 to handle the last entity
        if label == 1:
            if start is None:
                start = i
        elif start is not None:
            entities.append((start, i))
            start = None
    return entities


def evaluate_entities_only(model, entity_model, dataloader, device, tokenizer):
    model.eval()  # 设定模型为评估模式
    entity_model.eval()
    total_entities = 0  # 总实体数量
    predicted_entities = 0  # 预测实体数量
    correct_entities = 0  # 正确预测的实体数量

    with tqdm(dataloader, desc="评估", unit="batch") as pbar:
        for batch in pbar:
            is_vaild = True
            # 数据转移到设备
            input_A = batch['input_A'].to(device)
            input_B = batch['input_B'].to(device)
            attention_mask_A = batch['attention_mask_A'].to(device)
            attention_mask_B = batch['attention_mask_B'].to(device)
            labels = batch['labels'].to(device)
            texts = batch['texts']

            for attention in attention_mask_A:
                if attention[0] == 0:
                    is_vaild = False
            if not is_vaild:
                continue

            # 前向传播
            logits, feature_matrix, describe_matrix= model(input_A, attention_mask_A, input_B, attention_mask_B)
            predicted_labels = torch.argmax(logits, dim=-1)
            predicted_labels = adjust_labels_based_on_words_batch(
                texts=texts, batch_labels_tensor=predicted_labels, device=device, tokenizer=tokenizer
            )
            _, rejust_labels = entity_model(feature_matrix, describe_matrix,predicted_labels, labels)

            adjust_labels = adjust_labels_based_on_words_batch(
                texts=texts, batch_labels_tensor=rejust_labels, device=device, tokenizer=tokenizer
            )

            valid_predicted = adjust_labels[attention_mask_A.bool()]
            valid_labels = labels[attention_mask_A.bool()]

            # 将张量转换为列表以便处理
            pred_list = valid_predicted.tolist()
            label_list = valid_labels.tolist()


            # 找到参考标签中的实体
            ref_entities = find_entity_spans(label_list)
            pred_entities = find_entity_spans(pred_list)
            tokens = []
            for l in range(len(texts)):
                tokens = tokens + tokenizer.tokenize(texts[l])


            ref_tokens = []
            pred_tokens = []
            for start,end in ref_entities:
                ref_token = []
                for j in range(start, end):
                    ref_token.append(tokens[j])
                ref_tokens.append(ref_token)
            for start,end in pred_entities:
                pred_token = []
                for j in range(start, end):
                    pred_token.append(tokens[j])
                pred_tokens.append(pred_token)

            # 更新实体统计
            total_entities += len(ref_entities)
            predicted_entities += len(pred_entities)
            correct_entities += sum(
                1 for ref_start, ref_end in ref_entities
                if any(
                    (abs(pred_start - ref_start) <= 0 and pred_end == ref_end) or
                    (pred_start == ref_start and abs(pred_end - ref_end) <= 0)
                    for pred_start, pred_end in pred_entities
                )
            )
            # 计算并更新进度条描述
            precision = correct_entities / predicted_entities if predicted_entities > 0 else 0
            recall = correct_entities / total_entities if total_entities > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            pbar.set_postfix(Precision=f"{precision:.3f}", Recall=f"{recall:.3f}", F1_Score=f"{f1_score:.3f}")

    # 最终打印结果
    print(f"Final Precision for Entities: {precision * 100:.2f}%")
    print(f"Final Recall for Entities: {recall * 100:.2f}%")
    print(f"Final F1 Score for Entities: {f1_score * 100:.2f}%")

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }





# 设置训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 模型和优化器
model = CrossDomainModel(embed_dim=1536, num_heads=16, ff_dim=2048, num_layers=9, num_classes=2).to(device)

entity_model = EntityAttentionDecoder(embed_dim=1536,num_heads=16,ff_dim=2048).to(device)

# 加载保存的参数文件
checkpoint = torch.load('model_layers_9_weight_25.pth')

entity_checkpoint = torch.load("entity_attention_model_layer_9_weight_25.pth")
# 将 attention 层的参数加载到模型中
model.attention.load_state_dict(checkpoint['attention'])
# 将 classifier 层的参数加载到模型中
model.classifier.load_state_dict(checkpoint['classifier'])

entity_model.load_state_dict(entity_checkpoint)


# 加载数据并创建数据集实例
with open("test_data.json", "r", encoding="utf-8") as file:
    test_datas = json.load(file)

model_path = "/root/autodl-tmp/Models/Qwen2.5-1.5B/"
tokenizer = Qwen2TokenizerFast.from_pretrained(model_path)

test_dataset = EntityTextDataset(test_datas, tokenizer)

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

entity_model.eval()
model.eval()

evaluate_entities_only(model,entity_model,test_loader,device,tokenizer)
