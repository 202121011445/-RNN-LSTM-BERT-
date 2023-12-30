import torch
import numpy as np


class Preprocessing:
    def __init__(self, hidden_dim, word2idx, max_len):
        # 参数预设
        self.hidden_dim = hidden_dim
        self.word2idx = word2idx
        self.max_len = max_len + 2
        # 特殊字符
        self.pad_index = 0
        self.unk_index = 1
        self.cls_index = 2
        self.sep_index = 3
        self.mask_index = 4
        self.num_index = 5
        # 初始化位置编码
        self.position_encoding = self.init_position_embedding()

    # 初始化位置编码
    def init_position_embedding(self):
        position_embedding = np.array([
            [pos / np.power(10000, 2 * i / self.hidden_dim) for i in range(self.hidden_dim)]
            if pos != 0 else np.zeros(self.hidden_dim) for pos in range(self.max_len)])
        position_embedding[1:, 0::2] = np.sin(position_embedding[1:, 0::2])
        position_embedding[1:, 1::2] = np.cos(position_embedding[1:, 1::2])
        position_embedding = position_embedding / (np.sqrt(np.sum(position_embedding ** 2, axis=1, keepdims=True)) + 1e-8)
        return position_embedding

    # 重载括号运算符
    def __call__(self, texts, seq_len):
        # 词向量化并添加<CSL>和<SEP>标记
        tokens = [torch.tensor([self.cls_index] + [self.word2idx.get(char, self.unk_index) for char in text] + [self.sep_index]) for text in texts]
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        position = torch.from_numpy(self.position_encoding[:seq_len + 2]).type(torch.FloatTensor)
        return tokens, position
