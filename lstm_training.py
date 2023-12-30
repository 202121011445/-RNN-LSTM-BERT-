import os
import pandas as pd
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
import torch as d2l

from word2vec import *
import numpy as np


def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充


class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 双向循环神经网络
        # self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers, bidirectional=True)
        self.encoder = nn.RNN(embed_size, num_hiddens, num_layers=num_layers, bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 1)
        self.activation = nn.Sigmoid()

    def forward(self, inputs):
        # inputs的形状是（批量大小，时间步数），因为长短期记忆网络要求其输入的第一个维度是时间维，所以在获得词元表示之前，输入会被转置。
        # 输出形状为（时间步数，批量大小，词向量维度）
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()  # 重置参数，在使用NVDIA GPU时处理更快
        # 返回上一个隐藏层在不同时间步的隐状态，
        # outputs的形状是（时间步数，批量大小，2*隐藏单元数）
        outputs, _ = self.encoder(embeddings)
        # 连结初始和最终时间步的隐状态，作为全连接层的输入，
        # 其形状为（批量大小，4*隐藏单元数）
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return self.activation(outs)


def init_weights(m):
    """初始化网络权重"""
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])


def auc(pred, labels):
    """计算AUC"""
    fpr, tpr, thresholds = metrics.roc_curve(y_true=labels, y_score=pred)
    return metrics.auc(fpr, tpr)


def find_best_threshold(all_predictions, all_labels):
    """寻找最佳的分类边界, 在0到1之间"""
    # 展平所有的预测结果和对应的标记
    all_predictions = np.ravel(all_predictions)  # 将多维数组降为一维
    all_labels = np.ravel(all_labels)
    # 从0到1以0.01为间隔定义99个备选阈值, 分别是从0.01-0.99之间
    thresholds = [i / 100 for i in range(100)]
    all_auc = []
    for threshold in thresholds:
        # 计算当前阈值的f1 score
        preds = (all_predictions >= threshold).astype(
            "int")  # (array([1, 2, 3, 4]) > 2).astype("int") = array([0, 0, 1, 1])
        tem_auc = auc(preds, all_labels)  # 传入真实值和预测值
        all_auc.append(tem_auc)
    # 找出可以使f1 socre最大的阈值
    best_threshold = thresholds[int(np.argmax(np.array(all_auc)))]
    return best_threshold, max(all_auc)


# def predict_sentiment(net, vocab, text, threshold):
#     """预测情感"""
#     text = torch.tensor(vocab[jieba_str(text, get_stop_words())], device=try_gpu())
#     pred = net(text.reshape(1, -1)).item()
#     if pred >= threshold:
#         return pred, 1
#     else:
#         return pred, 0


def compute_loss(predictions, labels):
    # 将预测和标记的维度展平
    predictions = predictions.view(-1)
    labels = labels.float().view(-1)
    epsilon = 1e-8
    # 交叉熵
    loss = - labels * torch.log(predictions + epsilon) - (torch.tensor(1.0) - labels) * torch.log(
        torch.tensor(1.0) - predictions + epsilon)
    loss = torch.mean(loss)
    return loss


def transform_preds(preds, threshold):
    """将预测值转化为0,1标签"""
    ans = []
    for pred in preds:
        if pred[0] < threshold:
            ans.append(0)
        else:
            ans.append(1)
    return ans


if __name__ == '__main__':
    # 加载语料
    stop_wprds = get_stop_words()
    train_tokens, train_labels = get_corpus("./corpus/train_sentiment.txt", stop_wprds)  # 分词且去除停用词文本列表
    test_tokens, test_labels = get_corpus("./corpus/test_sentiment.txt", stop_wprds)

    # 构建词典
    vocab = Vocab(train_tokens + test_tokens, min_freq=5, reserved_tokens=['<pad>'])  # 构建字典。len(vocab):10281
    print("字典大小{}".format(len(vocab)))
    # 统计处理后的语料长度
    lens = []
    for tokens in train_tokens + test_tokens:
        lens.append(len(tokens))
    lens = sorted(lens)
    num_steps = lens[int(0.98 * len(lens))]  # 序列长度。将nums_steps的长度设置为98%语句的最大长度
    print('num_steps:{}, max_len:{}'.format(num_steps, len(lens)))
    # 按统一长度将语句张量化
    train_features = torch.tensor([truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])

    # 构建DataLoader。迭代器每次返回一个batch的数据
    bath_size = 1024
    train_iter = d2l.load_array((train_features, torch.tensor(train_labels)), bath_size)  # 获取数据迭代器batch_size是32。
    test_iter = d2l.load_array((test_features, torch.tensor(test_labels)), bath_size)

    # 构造模型并进行初始化
    embed_size = 100
    all_best_auc = {}
    num_hiddens_list = [128,256,512]
    num_layers_list = [1,2,3]
    for num_hiddens in num_hiddens_list:
        for num_layers in num_layers_list:
            for all_epoch in range(2):
                net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
                net.apply(init_weights)

                # 加载预训练的词向量模型，并防止参数更新
                net.embedding.load_state_dict(torch.load('Word2vec.params'))
                net.embedding.weight.requires_grad = False

                # 参数设置
                dynamic_lr = 0.01
                optimizer = torch.optim.Adam(net.parameters(), lr=dynamic_lr)
                device = try_gpu()
                net = net.to(device)
                epoches = 999
                patient = 4

                # 用于记录数据的列表
                train_loss = []
                test_loss = []
                train_auc = []
                test_auc = []
                train_acc = []
                test_acc = []
                train_f1 = []
                test_f1 = []
                all_best_threshold = []

                # 使用pandas DataFrame进行日志存储
                # log_path = "lstm hiddens{0} layers{1} epoch{2}.pickle".format(num_hiddens, num_layers, all_epoch)
                log_path = "rnn hiddens{0} layers{1} epoch{2}.pickle".format(num_hiddens, num_layers, all_epoch)
                print(log_path)
                if os.path.isfile(log_path):
                    os.remove(log_path)
                pd.DataFrame(columns=["epoch", "train_loss", "train_auc", "train_acc",
                                      "test_loss", "test_auc", "test_acc", "time", "threshold"
                                      ]).to_pickle(log_path)

                # 训练
                for epoch in range(epoches):
                    print("epoch[{}]学习率为{}".format(epoch, str(dynamic_lr)))
                    # 初始化记录
                    total_loss = 0
                    train_labels, train_preds = [], []
                    # 训练迭代
                    for i, (features, labels) in tqdm(enumerate(train_iter), desc="train_epoch_:%d" % (epoch),
                                                      total=len(train_iter)):
                        # 前向传播
                        features = features.to(device)
                        labels = labels.to(device)
                        net.train()
                        pred = net(features)
                        loss = compute_loss(pred, labels)
                        # 反向传播
                        optimizer.zero_grad()
                        loss.sum().backward()
                        optimizer.step()
                        # 记录总的损失
                        total_loss += loss.sum().item()
                        train_labels += labels
                        train_preds += pred
                    # 将数据传回CPU
                    train_labels = [i.cpu().numpy().tolist() for i in train_labels]
                    train_preds = [i.detach().cpu().numpy().tolist() for i in train_preds]
                    # 保存模型
                    # torch.save(net.state_dict(),
                    #            'lstm hiddens{0} layers{1} epoch{2}.params'.format(num_hiddens, num_layers, all_epoch))
                    torch.save(net.state_dict(),
                               'rnn hiddens{0} layers{1} epoch{2}.params'.format(num_hiddens, num_layers, all_epoch))
                    # 计算评价指标
                    best_threshold, best_auc = find_best_threshold(train_preds, train_labels)
                    train_preds = transform_preds(train_preds, best_threshold)
                    train_auc.append(best_auc)
                    train_acc.append(accuracy_score(y_pred=train_preds, y_true=train_labels))
                    train_f1.append(f1_score(y_pred=train_preds, y_true=train_labels))
                    train_loss.append(total_loss / labels.shape[0])

                    # 测试
                    total_loss = 0
                    test_labels, test_preds = [], []
                    net.eval()
                    with torch.no_grad():
                        for features, labels in tqdm(test_iter, desc="test_epoch_:%d" % (epoch),
                                                     total=len(test_iter)):
                            features = features.to(device)
                            labels = labels.to(device)
                            pred = net(features)
                            loss = compute_loss(pred, labels)
                            total_loss += loss.sum().item()
                            test_labels += labels
                            test_preds += pred
                    # 将数据穿会CPU
                    test_labels = [i.cpu().numpy().tolist() for i in test_labels]
                    test_preds = [i.detach().cpu().numpy().tolist() for i in test_preds]
                    # 计算评价指标
                    best_threshold, best_auc = find_best_threshold(test_preds, test_labels)
                    test_preds = transform_preds(test_preds, best_threshold)
                    test_auc.append(best_auc)
                    test_acc.append(accuracy_score(y_pred=test_preds, y_true=test_labels))
                    test_f1.append(f1_score(y_pred=test_preds, y_true=test_labels))
                    test_loss.append(total_loss / len(test_labels))

                    # 寻找最佳阈值
                    all_best_threshold.append(best_threshold)

                    # 数据记录
                    log_dic = {"epoch": epoch, "train_loss": train_loss[-1], "train_auc": train_auc[-1],
                               "train_acc": train_acc[-1],
                               "train_f1": train_f1[-1],
                               "test_loss": test_loss[-1], "test_auc": test_auc[-1], "test_acc": test_acc[-1],
                               "test_f1": test_f1[-1], "time": time.time(),
                               "threshold": all_best_threshold[-1]}
                    print(str(log_dic))
                    df = pd.read_pickle(log_path)
                    df = df.append([log_dic])
                    df.reset_index(inplace=True, drop=True)
                    df.to_pickle(log_path)

                    # 更新学习率
                    if test_auc[-1] < max(test_auc):
                        threshold += 1
                        dynamic_lr *= 0.6
                        optimizer = torch.optim.Adam(net.parameters(), lr=dynamic_lr)
                    else:
                        threshold = 0

                    # 早停防止过拟合
                    if threshold >= patient:
                        index = np.argmax(np.array(test_auc))
                        print("epoch {}达到最佳阈值{} ".format(index, all_best_threshold[index]))
                        break
                all_best_auc[
                    "hiddens:{0}-layers:{1}-epoch:{2}".format(int(num_hiddens), int(num_layers), all_epoch)] = int(
                    np.max(test_auc) * 10000)
                print(all_best_auc)




    # test_list = [
    #     '手机拿到手，感觉很好。屏幕细致，手感也不错，运行速度蛮好。还在试用当中。还有一点充电快用电快。',
    #     '值得购买的产品，做工非常精细，比想象中的好，推荐大家尝试',
    #     '辣鸡手机，玩了一个月的样子，打王者有时都掉帧，想玩游戏还是别买小米，真的不行，我同事买的11有一年也不行，用了几个小米手机了，真的不想用了，越来越差。服了。',
    #     "用了快一个月，以前不是很了解小米手机，但是小米手机我个人觉得还是慎入，可能是我还不太会用吧，不到一个月已经卡了三次😧我平时也不玩游戏也按要求升级了系统，感觉玩不明白了",
    #     "不咋地，发烫的很，感觉买后悔了",
    #     "这个手机并没有想象的那么流畅，有的时候很卡顿。有点失望。",
    #     '我也不知道怎么回事，连我原来的那个红米k40还不如，打原神总是卡帧'
    # ]
    # for text in test_list:
    #     print(predict_sentiment(net, vocab, text, all_best_threshold[np.argmax(np.array(test_auc))]))

