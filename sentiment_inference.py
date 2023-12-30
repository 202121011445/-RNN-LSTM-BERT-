from dataset.preprocessing import Preprocessing
from models.sentiment import *
import numpy as np
import configparser
import os
import json
import warnings
import matplotlib.pyplot as plt

from models.sentiment import SentimentModel


class Sentiment_Analysis:
    def __init__(self, seq_len, batch_size):
        config = configparser.ConfigParser()
        config.read("./sentiment_config.ini")
        self.config = config["DEFAULT"]
        self.vocab_size = int(self.config["vocab_size"])
        self.batch_size = batch_size
        self.seq_len = seq_len
        # 加载字典
        with open(self.config["word2idx_path"], "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)
        # 初始化BERT情感分析模型
        bertconfig = BertConfig(vocab_size=self.vocab_size)
        self.bert_model = SentimentModel(config=bertconfig)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bert_model.to(self.device)
        # 开去evaluation模型, 关闭模型内部的dropout层
        self.bert_model.eval()
        # 初始化位置编码
        self.hidden_dim = bertconfig.hidden_size
        self.positional_enc = self.init_positional_encoding()
        self.positional_enc = torch.unsqueeze(self.positional_enc, dim=0)
        # 初始化预处理器
        self.process_batch = Preprocessing(hidden_dim=bertconfig.hidden_size, word2idx=self.word2idx,
                                           max_len=seq_len)
        # 加载BERT预训练模型
        self.load_model(self.bert_model, dir_path=self.config["state_dict_dir"])

    def init_positional_encoding(self):
        position_enc = np.array([
            [pos / np.power(10000, 2 * i / self.hidden_dim) for i in range(self.hidden_dim)]
            if pos != 0 else np.zeros(self.hidden_dim) for pos in range(self.seq_len)])
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        denominator = np.sqrt(np.sum(position_enc ** 2, axis=1, keepdims=True))
        position_enc = position_enc / (denominator + 1e-8)
        position_enc = torch.from_numpy(position_enc).type(torch.FloatTensor)
        return position_enc

    def load_model(self, model, dir_path="../output"):
        checkpoint_dir = self.find_most_recent_state_dict(dir_path)
        checkpoint = torch.load(checkpoint_dir)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        torch.cuda.empty_cache()
        model.to(self.device)
        print("{} 已被加载!".format(checkpoint_dir))

    def __call__(self, text_list, batch_size=1, threshold=.5):
        # 异常判断
        if isinstance(text_list, str):
            text_list = [text_list, ]
        text_list = [i for i in text_list if len(i) != 0]
        # seq_len=self.seq_len+2 因为要留出cls和sep的位置
        max_seq_len = max([len(i) for i in text_list])
        # 预处理, 获取batch
        texts_tokens, positional_enc = self.process_batch(text_list, seq_len=max_seq_len)
        positional_enc = torch.unsqueeze(positional_enc, dim=0).to(self.device)
        # 正向
        mini_batch = math.ceil(len(texts_tokens) / batch_size)
        # 数据按mini batch切片过正向, 这里为了可视化所以吧batch size设为1
        for i in range(mini_batch):
            start = i * batch_size
            end = start + batch_size
            texts_tokens_ = texts_tokens[start: end].to(self.device)
            predictions = self.bert_model.forward(text_input=texts_tokens_,
                                                  positional_enc=positional_enc,
                                                  )
            predictions = np.ravel(predictions.detach().cpu().numpy()).tolist()
            for text, pred in zip(text_list[start: end], predictions):
                self.sentiment_print_func(text, pred, threshold)

    def sentiment_print_func(self, text, pred, threshold):
        print(text)
        if pred >= threshold:
            print("正样本, 输出值{:.2f}".format(pred))
        else:
            print("负样本, 输出值{:.2f}".format(pred))
        print("----------")

    def find_most_recent_state_dict(self, dir_path):
        """
        :param dir_path: 存储所有模型文件的目录
        :return: 返回最新的模型文件路径, 按模型名称最后一位数进行排序
        """
        dic_lis = [i for i in os.listdir(dir_path)]
        dic_lis = [i for i in dic_lis if "sentiment" in i]
        dic_lis = sorted(dic_lis, key=lambda k: int(k.split(".")[-1]))
        return dir_path + "/" + dic_lis[-1]


if __name__ == '__main__':
    model = Sentiment_Analysis(seq_len=300, batch_size=1)
    test_list =['这个手机拍照还可以，不过摄像模块太大了，看起来特别特别夸张。其他没什么惊艳的地方，米粉节性价比超高！',
                '感觉手机非常好，特别喜欢，要是发热再好点就更好了',
                '拍照清晰系统流畅 变焦超级给力 不愧是拍照手机 \n美中不足的是电池虽大但掉电有些快',
                '性价比之王，辨识度很高，屏幕分辨率很清晰，拍照很清楚，无线充电是最方便的，还送了一个手机壳',
                '旗舰手机，不玩游戏，只拍照，性能溢出，不多说了，会买的都懂',
                '拍照效果还没我之前的10s好',
                '充电和玩游戏，手机发热发烫',
                '才用了几天 屏幕就出了问题 心态崩了',
                '手机没有三包凭证，也不知道是不是三无产品，这样的东西我只能给一星，并且赶紧退货',
                '骁龙8gen1是真的烫。摄像头是真的突出，不太行',
                '千万别买，伤眼睛不说是真的不好用',
                '手机确实是太好用了，系统特别好用，流畅的不得了'
                ]
    model(test_list,threshold=0.2)
