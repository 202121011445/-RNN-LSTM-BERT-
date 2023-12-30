import time
import torch
import jieba
from torch import nn
import random
import collections
from tqdm import tqdm


def batchify(data):
    """返回带有负采样的跳元模型的小批量样本"""
    max_len = max(len(contexts) + len(noises) for centers, contexts, noises in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return ((torch.tensor(centers)).reshape(-1, 1), torch.tensor(
        contexts_negatives), torch.tensor(masks), torch.tensor(labels))


class RandomGenerator:
    """根据n个采样权重在{1,...,n}中随机抽取"""

    def __init__(self, sampling_weights):
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # 缓存k个随机采样结果
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]


def get_negatives(all_contexts, vocab, counter, K):
    """返回负采样中的噪声词"""
    # （索引0和1为特殊标记）根据word2vec论文中的建议，将噪声词w的采样概率P(w)设置为其在字典中的相对频率，其幂为0.75
    sampling_weights = [counter[vocab.to_tokens(i)] ** 0.75 for i in range(2, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # 噪声词不能是上下文词
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives


def get_centers_and_contexts(corpus, max_window_size):
    """返回跳元模型中的中心词和上下文词,中心词为一个元素，上下文词为长[1,max_window_size]的列表"""
    centers, contexts = [], []
    for line in corpus:
        # 要形成“中心词-上下文词”对，每个句子至少需要有2个词
        if len(line) < 2:
            continue
        centers += line  # 中心词维度[句子数，每个句子的字典表示（即每个字都会被设为中心词一次）]
        for i in range(len(line)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size), min(len(line), i + 1 + window_size)))
            # 从上下文词中排除中心词
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts


class Vocab:
    """构建情感分析词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        self.unk = 0
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 将词元列表展平成一个列表,并统计每个词元出现频率
        tokens = [token for line in tokens for token in line]
        self.counter = collections.Counter(tokens)
        # 按词元出现频率逆序排序
        self.token_freqs = sorted(self.counter.items(), key=lambda x: x[1], reverse=True)
        # 索引到词列表
        self.idx_to_token = ['<unk>'] + reserved_tokens  # 未知词元以及添加词典先编码
        # 词到索引字典
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        # 使用词构建字典并过滤出现频率较低的词
        for token, freq in self.token_freqs:
            if freq < min_freq:
                break
            if token not in self.idx_to_token:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    """获取词典长度"""

    def __len__(self):
        return len(self.idx_to_token)

    """文本转索引"""

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.token_to_idx.get(token, self.unk) for token in tokens]

    """索引转文本"""

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    def token_freqs(self):
        return self.token_freqs

    def get_counter(self):
        return self.counter


def get_stop_words():
    """获取停用词表"""
    with open('cn_stopwords.txt', 'r', encoding='utf-8') as f:
        stop_words = f.readlines()
    return stop_words


def jieba_str(str, stop_words):
    """对str使用精准模式进行结巴分词，并使用哈工大停用词典进行停用词处理，返回处理后的词语列表"""
    words = jieba.cut(str)
    ans = []
    for word in words:
        if word + '\n' not in stop_words and word != ' ':
            ans.append(word)
    return ans


# def jieba_test():
#     """结巴分词模式测试"""
#     str = "手机内存大，速度快，系统特别好用，流畅的不得了，一直以来都是米粉，" \
#           "12pro简直性价比杠杠的，简直太棒了！"
#     ans1 = jieba.cut(str, cut_all=True)
#     ans2 = jieba.cut(str)
#     ans3 = jieba.cut_for_search(str)
#     print("全模式：" + '/'.join(ans1))
#     print('精确模式：' + '/'.join(ans2))
#     print('搜素引擎模式：' + '/'.join(ans3))


def get_corpus(dir, stop_words):
    """
    加载情感分析语料，并对每句话进行jieba分词以及去除停用词处理
    :param dir:
    :param stop_words:
    :return: words:列表，每个元素为原语句处理后构成的词列表；labels:列表，每个元素为原语句对应的标签
    """
    words = []
    labels = []
    with open(dir, 'r', encoding='utf-8') as f:
        dicts = f.readlines()
    for dic in dicts:
        dic = eval(dic)
        words.append(jieba_str(dic['text'], stop_words))
        labels.append(dic['label'])
    return words, labels


def load_data(batch_size, max_window_size, num_noise_words):
    """加载数据集，返回迭代期间所需的小批量DataLoader以及字典"""

    # 对语料进行结巴分词并使用哈工打停用词典去除停用词
    stop_words = get_stop_words()
    train_words, train_labels = get_corpus('./corpus/train_sentiment.txt', stop_words)
    test_words, test_labels = get_corpus('./corpus/test_sentiment.txt', stop_words)
    words = train_words + test_words

    # 使用语料中所有的词构建词典
    vocab = Vocab(words, min_freq=5, reserved_tokens=['<pad>'])  # 去除出现次数小于5次的词
    counter = vocab.get_counter()

    # 将所有语料转化为词对应的索引
    corpus = [vocab[line] for line in words]

    # 执行跳元模型，返回中心词和上下文词。两个列表相应索引位置构成“中心词-上下文词”对。
    all_centers, all_contexts = get_centers_and_contexts(corpus, max_window_size)

    # 负采样
    all_negatives = get_negatives(all_contexts, vocab, counter, num_noise_words)

    # 内部类Dataset
    class SentimentDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index], self.negatives[index])

        def __len__(self):
            return len(self.centers)

    # 构建dataset
    dataset = SentimentDataset(all_centers, all_contexts, all_negatives)

    # 以指定batch构建数据加载器
    data_iter = torch.utils.data.DataLoader(dataset, batch_size, collate_fn=batchify, num_workers=0)
    return data_iter, vocab


def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    """跳元语法模型"""
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred


class SigmoidBCELoss(nn.Module):
    """带掩码的二元交叉熵损失"""

    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)


def init_weights(m):
    if type(m) == nn.Embedding:
        nn.init.xavier_uniform_(m.weight)


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def save_data(net, epoch):
    torch.save(net[0].state_dict(), 'Word2vec{}.params'.format(epoch))


def train_word2vec(batch_size, window_size, num_noise_words):
    """word2vec预训练"""

    # 训练计时
    start_time = time.time()

    # 获取迭代数据以及字典
    data_iter, vocab = load_data(batch_size, window_size, num_noise_words)

    # 定义损失与嵌入层
    loss = SigmoidBCELoss()
    embed_size = 100
    net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_size),
                        nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_size))

    # 模型训练
    dynamic_lr, num_epochs = 0.002, 200
    optimizer = torch.optim.Adam(net.parameters(), lr=dynamic_lr)
    net.apply(init_weights)
    device = try_gpu()
    net = net.to(device)
    # 规范化的损失之和，规范化的损失数
    a, b = 0, 0
    all_loss = []
    threshold = 200
    for epoch in range(num_epochs):
        for i, batch in tqdm(enumerate(data_iter), desc="epoch_:%d" % (epoch), total=len(data_iter)):
            optimizer.zero_grad()
            center, context_negative, mask, label = [data.to(device) for data in batch]
            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask) / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            a += l.sum()
            b += l.numel()
        all_loss.append(a / b)
        if all_loss[-1] > min(all_loss):
            threshold += 1
            dynamic_lr *= 0.8
            optimizer = torch.optim.Adam(net.parameters(), lr=dynamic_lr)
        else:
            threshold = 0
        print("epoch:{} loss={}".format(epoch, all_loss[-1]))
        print("执行到epoch[{}]所耗时间为：{}".format(epoch, time.time() - start_time))
        save_data(net, epoch)
        if threshold >= 5:
            print("epoch:{} 达到最佳！ loss={}".format(epoch - 5, all_loss[-5]))
            break
    print(all_loss)


# def get_similar_tokens(query_token, k, net, vocab):
#     W = net.weight.data
#     x = W[vocab[query_token]]
#     # 计算余弦相似性。增加1e-9以获得数值稳定性
#     cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
#                                       torch.sum(x * x) + 1e-9)
#     topk = torch.topk(cos, k=k + 1)[1].cpu().numpy().astype('int32')
#     dic = {}
#     for i in topk[1:]:  # 删除输入词
#         dic[vocab.to_tokens(i)] = float(cos[i])
#     return dic


# def get_net():
#     """得到加载参数的模型与对应字典"""
#     # 对语料进行结巴分词并使用哈工打停用词典去除停用词
#     stop_words = get_stop_words()
#     train_words, train_labels = get_corpus('./corpus/train_sentiment.txt', stop_words)
#     test_words, test_labels = get_corpus('./corpus/test_sentiment.txt', stop_words)
#     words = train_words + test_words
#
#     # 使用语料中所有的词构建词典
#     vocab = Vocab(words, min_freq=5, reserved_tokens=['<pad>'])  # 去除出现次数小于5次的词
#     embed_size = 100
#     net = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_size)
#     net.load_state_dict(torch.load("Word2vec.params"))
#     return net, vocab


# def get_features(net, vocab):
#     ans = {}
#     keys = {"服务", "客服", "物流", "快递", "电池", "充电", "待机时间", "续航",
#             "屏幕", "屏", "性能", "速度", "相机", "拍照", "摄像头", "像素", "外观", "外形",
#             "颜色", "手感", "颜值", "音质", "音效", "系统", "信号"}
#     for key in keys:
#         ans[key] = get_similar_tokens(key, 10, net, vocab)
#     return ans


if __name__ == '__main__':
    batch_size, window_size, num_noise_words = 1024, 2, 12
    train_word2vec(batch_size, window_size, num_noise_words)
    # net, vocab = get_net()
