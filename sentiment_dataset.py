from torch.utils.data import Dataset
import tqdm
import json
import torch
import random
import numpy as np
from sklearn.utils import shuffle
import re


class SentimentDataset(Dataset):
    def __init__(self, corpus_path, word2idx, seq_len, regular=False):
        # 预设参数
        self.corpus_path = corpus_path
        self.word2idx = word2idx
        self.seq_len = seq_len
        self.regular = regular
        # 特殊字符
        self.pad_index = 0
        self.unk_index = 1
        self.cls_index = 2
        self.sep_index = 3
        self.mask_index = 4
        self.num_index = 5
        # 加载语料并获取长度
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.lines = [eval(line) for line in f.readlines()]
            self.corpus_lines = len(self.lines)

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, index):
        text = self.lines[index]["text"]
        label = self.lines[index]["label"]
        # 缓解过拟合：每个文本10%的几率将再分句
        if self.regular:
            if random.random() < 0.1:
                site = [i.span() for i in re.finditer("；|。|？|！", text)]    # 通过i.span()获取匹配的位置
                if len(site) != 0:
                    position = site[np.random.randint(len(site))][1]
                    if position > 2:
                        text = text[:position]
                    else:
                        text = text[position:]
        # 句子向量化
        token_embeddings = [self.word2idx.get(i, self.unk_index) for i in text]
        token_embeddings = ([self.cls_index] + token_embeddings + [self.sep_index])[:self.seq_len]

        return {"token_embeddings": torch.tensor(token_embeddings),"label": torch.tensor([label])}