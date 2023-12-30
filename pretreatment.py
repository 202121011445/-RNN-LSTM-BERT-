import random
import numpy as np
import xlrd
import json
import matplotlib.pyplot as plt
import torch as d2l


def load_data(xls_path):
    """
    读取excel文件中的数据，并进行预处理操作以构建用于训练和测试的数据
    :param xls_path: excel文件所在路径
    :return:
    """
    # 打开xls文件对应的表
    sheet = xlrd.open_workbook(xls_path).sheet_by_name('Sheet1')
    # 读取所有数据并分为negative与positive
    pos_texts = []
    neg_texts = []
    for i in range(sheet.nrows):
        row = sheet.row_values(i)
        if str.isdigit(row[1].strip()) or len(row[1]) < 5:
            continue
        row[1] = row[1].replace('\n', '。').replace('\u202f', '。').replace('\xa0', '。')
        if int(row[0]) == 1:
            pos_texts.append(row[1])
        else:
            neg_texts.append(row[1])
    # 去重
    random.shuffle(list(set(pos_texts)))
    random.shuffle(list(set(neg_texts)))
    if len(pos_texts) > len(neg_texts):
        pos_texts = pos_texts[:len(neg_texts)]
    else:
        neg_texts = neg_texts[:len(pos_texts)]
    data = []
    lens = []
    for text in pos_texts:
        data.append({'text': text, 'label': 1})
        lens.append(len(text))
    for text in neg_texts:
        data.append({'text': text, 'label': 0})
        lens.append(len(text))
    random.shuffle(data)
    # 分割训练集和测试集
    test = data[:round(len(data) * 0.1)]
    train = data[round(len(data) * 0.1):]
    # 输出训练集和测试集为txt文件
    with open("train_sentiment.txt", "w", encoding="utf-8") as f:
        for item in train:
            f.write(str(item) + '\n')
    with open("test_sentiment.txt", "w", encoding="utf-8") as f:
        for item in test:
            f.write(str(item) + '\n')
    print("正向评论数量:{:d}".format(len(pos_texts)))
    print("负向评论数量:{:d}".format(len(neg_texts)))
    print("训练集评论数量:{:d}".format(len(train)))
    print("测试集评论数量:{:d}".format(len(test)))
    # 可视化语料序列长度, 可见大部分文本的长度都在300以下
    plt.hist(lens, bins=300)
    plt.xlabel("length")
    plt.ylabel("count")
    plt.show()
    print(np.mean(np.array(lens) < 300))


if __name__ == '__main__':
    load_data("data.csv")
    """
    正向评论数量:28811
    负向评论数量:28811
    训练集评论数量:51860
    测试集评论数量:5762
    0.9860643504217139
    """
