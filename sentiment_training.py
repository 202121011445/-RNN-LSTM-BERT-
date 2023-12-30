import time

from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset.sentiment_dataset import SentimentDataset
from models.sentiment import *
from sklearn import metrics

import tqdm
import pandas as pd
import numpy as np
import configparser
import os
import json


class SentimentTrainer:
    def __init__(self, seq_len, batch_size, lr):
        # 参数设置
        config = configparser.ConfigParser()
        config.read("sentiment_config.ini")
        self.config = config["DEFAULT"]
        self.vocab_size = int(self.config["vocab_size"])
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.lr = lr
        # 字典加载
        with open(self.config["word2idx_path"], "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)
        # 初始化BERT情感分析模型
        bertconfig = BertConfig(vocab_size=self.vocab_size)
        self.model = SentimentModel(config=bertconfig)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # 声明训练数据集
        train_dataset = SentimentDataset(corpus_path=self.config["train_corpus_path"], word2idx=self.word2idx,
                                         seq_len=self.seq_len, regular=True)
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=0,
                                           collate_fn=lambda x: x  # 这里为了动态padding
                                           )
        # 声明测试数据集
        test_dataset = SentimentDataset(corpus_path=self.config["test_corpus_path"], word2idx=self.word2idx,
                                        seq_len=self.seq_len, regular=False)
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=self.batch_size,
                                          num_workers=0,
                                          collate_fn=lambda x: x)
        # 初始化位置编码
        self.hidden_dim = bertconfig.hidden_size
        self.position_embed = self.init_position_embed()
        self.position_embed = torch.unsqueeze(self.position_embed, dim=0)
        # 声明需要优化的参数, 并传入Adam优化器
        self.optim_parameters = list(self.model.parameters())
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=1e-3)

    def init_position_embed(self):
        position_enc = np.array([
            [pos / np.power(10000, 2 * i / self.hidden_dim) for i in range(self.hidden_dim)]
            if pos != 0 else np.zeros(self.hidden_dim) for pos in range(self.seq_len)])
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        denominator = np.sqrt(np.sum(position_enc ** 2, axis=1, keepdims=True))
        position_enc = position_enc / (denominator + 1e-8)
        position_enc = torch.from_numpy(position_enc).type(torch.FloatTensor)
        return position_enc

    def load_model(self, model, dir_path, load_bert=False):
        if load_bert:
            dic_list = [i for i in os.listdir(dir_path) if 'bert' in i]
        else:
            dic_list = [i for i in os.listdir(dir_path) if 'sentiment' in i]
        dic_list = sorted(dic_list, key=lambda k: int(k.split(".")[-1]))
        print(dir_path + "/" + dic_list[-1])
        checkpoint = torch.load(dir_path + "/" + dic_list[-1])
        if load_bert:
            checkpoint["model_state_dict"] = {k[5:]: v for k, v in checkpoint["model_state_dict"].items()
                                              if k[:4] == "bert" and "pooler" not in k}
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        torch.cuda.empty_cache()
        model.to(self.device)
        print("{} 已被加载!".format(dir_path + "/" + dic_list[-1]))

    def save_state_dict(self, model, epoch, state_dict_dir, file_path):
        save_path = state_dict_dir + "/" + file_path + ".model.epoch.{}".format(str(epoch))
        model.to("cpu")
        torch.save({"model_state_dict": model.state_dict()}, save_path)
        print("{} 已被保存!".format(save_path))
        model.to(self.device)

    def train(self, epoch):
        # 一个epoch的训练
        self.model.train()
        self.iteration(epoch, self.train_dataloader, train=True)

    def test(self, epoch):
        # 一个epoch的测试, 并返回测试集的auc
        self.model.eval()  # 将模型重置为评估模式
        with torch.no_grad():
            return self.iteration(epoch, self.test_dataloader, train=False)

    def init_optimizer(self, lr):
        # 用指定的学习率初始化优化器
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=1e-3)

    def iteration(self, epoch, data_loader, train=True):
        # 使用pandas DataFrame进行日志存储
        log_path = self.config["state_dict_dir"] + "/" + "log.pickle"
        if not os.path.isfile(log_path):
            pd.DataFrame(columns=["epoch", "train_loss", "train_auc",
                                  "test_loss", "test_auc", "time"
                                  ]).to_pickle(log_path)
        # 进度条显示
        str_train = "train" if train else "test"
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="epoch_%s:%d" % (str_train, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        total_loss = 0
        all_predictions, all_labels = [], []

        for i, data in data_iter:
            # 补齐长度
            text_input = torch.nn.utils.rnn.pad_sequence([i["token_embeddings"] for i in data], batch_first=True)
            label = torch.cat([i["label"] for i in data])
            data = {"token_embeddings": text_input, "label": label}
            # 将数据发送到计算设备
            data = {key: value.to(self.device) for key, value in data.items()}
            position_embed = self.position_embed[:, :data["token_embeddings"].size()[-1], :].to(self.device)
            # 正向传播, 得到预测结果和loss
            predictions, loss = self.model.forward(text_input=data["token_embeddings"],
                                                   positional_enc=position_embed,
                                                   labels=data["label"]
                                                   )
            # 提取预测的结果和标记, 并存到all_predictions, all_labels里用来计算auc
            predictions = predictions.detach().cpu().numpy().reshape(-1).tolist()
            labels = data["label"].cpu().numpy().reshape(-1).tolist()
            all_predictions.extend(predictions)
            all_labels.extend(labels)

            # 计算auc
            fpr, tpr, thresholds = metrics.roc_curve(y_true=all_labels, y_score=all_predictions)
            auc = metrics.auc(fpr, tpr)
            # 反向传播
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                # 用获取的梯度更新模型参数
                self.optimizer.step()
            # 为计算当前epoch的平均loss
            total_loss += loss.item()
            log_dic = {"epoch": epoch, "train_loss": 0, "train_auc": 0, "test_loss": 0, "test_auc": 0, "time": 0}
            if train:
                log_dic["train_loss"] = total_loss / (i + 1)
                log_dic["train_auc"] = auc
                log_dic["time"] = time.time()
            else:
                log_dic["test_loss"] = total_loss / (i + 1)
                log_dic["test_auc"] = auc
                log_dic["time"] = time.time()

            if i % 10 == 0:
                data_iter.write(str({k: v for k, v in log_dic.items() if v != 0}))

        threshold = self.find_best_threshold(all_predictions, all_labels)
        print(str_train + "最佳阈值: " + str(threshold))

        # 将当前epoch的情况记录到DataFrame里
        if train:
            df = pd.read_pickle(log_path)
            df = df.append([log_dic])
            df.reset_index(inplace=True, drop=True)
            df.to_pickle(log_path)
        else:
            log_dic = {k: v for k, v in log_dic.items() if v != 0 and k != "epoch"}
            df = pd.read_pickle(log_path)
            df.reset_index(inplace=True, drop=True)
            for k, v in log_dic.items():
                df.at[epoch, k] = v
            df.to_pickle(log_path)
            # 返回auc, 作为early stop的衡量标准
            return auc

    def find_best_threshold(self, all_predictions, all_labels):
        """寻找最佳的分类边界, 在0到1之间"""
        # 展平所有的预测结果和对应的标记
        all_predictions = np.ravel(all_predictions)  # 将多维数组降为一维
        all_labels = np.ravel(all_labels)
        # 从0到1以0.01为间隔定义99个备选阈值, 分别是从0.01-0.99之间
        thresholds = [i / 100 for i in range(100)]
        all_f1 = []
        for threshold in thresholds:
            # 计算当前阈值的f1 score
            preds = (all_predictions >= threshold).astype(
                "int")  # (array([1, 2, 3, 4]) > 2).astype("int") = array([0, 0, 1, 1])
            f1 = f1_score(y_true=all_labels, y_pred=preds)  # 传入真实值和预测值
            all_f1.append(f1)
        # 找出可以使f1 socre最大的阈值
        best_threshold = thresholds[int(np.argmax(np.array(all_f1)))]
        return best_threshold


if __name__ == '__main__':
    start_epoch = 0
    dynamic_lr = 1e-06
    batch_size = 64
    trainer = SentimentTrainer(300, batch_size, dynamic_lr)
    all_auc = []
    threshold = 999
    patient = 5
    for epoch in range(start_epoch, 9999):
        if epoch == start_epoch and epoch == 0:
            trainer.load_model(trainer.model, dir_path=trainer.config["state_dict_dir"], load_bert=True)
        elif epoch == start_epoch:
            trainer.load_model(trainer.model, dir_path=trainer.config["state_dict_dir"])
        print("epoch[{}]学习率为{}".format(epoch, str(dynamic_lr)))
        trainer.train(epoch)
        trainer.save_state_dict(trainer.model, epoch,
                                state_dict_dir=trainer.config["state_dict_dir"],
                                file_path="sentiment")
        auc = trainer.test(epoch)

        all_auc.append(auc)
        best_auc = max(all_auc)
        if all_auc[-1] < best_auc:
            threshold += 1
            dynamic_lr *= 0.8
            trainer.init_optimizer(lr=dynamic_lr)
        else:
            threshold = 0

        if threshold >= patient:
            print("epoch {}达到最佳阈值 ".format(start_epoch + np.argmax(np.array(all_auc))))
            break

"""
test最佳阈值: 0.2
epoch 133达到最佳阈值 
"""